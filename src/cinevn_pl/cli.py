"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import optuna
import torch.cuda
import yaml
from jsonargparse import Namespace
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from cinevn.data import get_mask_type_class

from . import FastMriDataModule, TQDMProgressBarWithoutVersion, VarNetModule


class CineVNCLI(LightningCLI):
    """
    Customized LightningCLI for CineVN
    """
    def __init__(self, overwrite_args: dict | None = None, trial: optuna.Trial | None = None, **kwargs):
        self.overwrite_args = overwrite_args
        self.trial = trial

        parser_kwargs = kwargs.pop('parser_kwargs', {})
        save_config_kwargs = kwargs.pop('save_config_kwargs', {})
        save_config_kwargs.update({'overwrite': True})

        # check if a config file has been passed via command line
        config_found = False
        for i, arg in enumerate(sys.argv):
            if arg in ['-c', '--config']:
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                    config_found = True
                    break
        if not config_found:
            raise RuntimeError('No config file given')

        super().__init__(
            VarNetModule, FastMriDataModule, save_config_kwargs=save_config_kwargs, parser_kwargs=parser_kwargs,
            trainer_defaults={'callbacks': [TQDMProgressBarWithoutVersion()]}, **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        self.add_main_arguments(parser)

        # transform arguments
        group = parser.add_argument_group('Data transform and undersampling options:')
        group.add_argument('--transform.mask_type', type=str, default='gro',
                           choices=['random', 'random_acs', 'vista', 'gro', 'cava', 'equispaced', 'equispaced_acs'],
                           help='Type of k-space mask')
        group.add_argument('--transform.accelerations', type=int | str, nargs='+', default=[4],
                           help='Acceleration rates to use for masks')
        group.add_argument('--transform.random_flips', type=bool, default=True,
                           help='Random flipping of training data')
        group.add_argument('--transform.roll_temporal', type=bool, default=False,
                           help='Random temporal rolling of training data')
        group.add_argument('--transform.augment_phase', type=bool, default=False,
                           help='Data augmentation by random rotation of the phase')
        group.add_argument('--transform.augment_mask', type=bool, default=False,
                           help='Mask augmentation as defined by mask_type')
        group.add_argument('--transform.augment_tres', type=bool, default=False,
                           help='Temporal resolution augmentation')
        group.add_argument('--transform.field_strength', type=float, default=None,
                           help='Add noise to k-space to simulate SNR of a given field strength.')

        # callback arguments
        group = parser.add_argument_group('Callback shortcut options:')
        group.add_argument('--callback.acc_target_epoch', type=int, default=None,
                           help='Target epoch for acceleration scheduling')
        group.add_argument('--callback.patience', type=int, default=30,
                           help='Early stopping patience')
        group.add_argument('--callback.val_log_images', type=int, default=16,
                           help='Number of images to log during validation')
        group.add_argument('--callback.val_log_interval', type=int, default=10,
                           help='Interval for logging validation images')
        group.add_argument('--callback.save_gif', type=bool, default=False,
                           help='Save gif files of reconstructions')
        group.add_argument('--callback.checkpoint_monitor', type=str, default='val_metrics/ssim',
                           help='Metric to monitor for checkpointing')
        group.add_argument('--callback.checkpoint_mode', type=str, default='max', choices=['min', 'max'],
                           help='Mode for checkpointing')

        # float32 matrix multiplication precision
        parser.add_argument('--float32_matmul_precision', type=str, default='highest',
                            choices=['highest', 'high', 'medium'], help='Precision of float32 matrix multiplications')

    def before_instantiate_classes(self) -> None:
        subcommand = self.config['subcommand']
        if subcommand == 'predict':
            raise NotImplementedError('Prediction is not supported, please use `test` subcommand for inference')

        c = self.config[subcommand]
        if c.trainer.callbacks is None:
            c.trainer.callbacks = []  # initialize with empty list

        # overwrite args given via constructor
        if self.overwrite_args:
            for k, v in self.overwrite_args.items():
                c[k] = v

        # parse accelerations given as str
        c.transform.accelerations = [int(acc) for acc in c.transform.accelerations]

        # optuna
        if self.trial is not None:
            c.trainer.num_sanity_val_steps = 0
            c.trainer.callbacks.append(Namespace({
                'class_path': 'cinevn_pl.MyPyTorchLightningPruningCallback',
                'init_args': {'trial': self.trial, 'monitor': 'val_metrics/ssim'},
            }))
            c.trainer.callbacks.append(Namespace({
                'class_path': 'cinevn_pl.MetricMonitor',
                'init_args': {'monitor': 'val_metrics/ssim', 'mode': 'max'},
            }))

        # initialize DDP if all of the following conditions are met:
        # 1. more than one device available
        # 2. `devices` is 'auto', -1, or greater than 1
        if (torch.cuda.device_count() > 1
            and (c.trainer.devices == 'auto' or int(c.trainer.devices) == -1 or int(c.trainer.devices) > 1)
        ):
            c.trainer.strategy = Namespace({
                'class_path': 'pytorch_lightning.strategies.DDPStrategy',
                'init_args': {'find_unused_parameters': False, 'static_graph': True},
            })
            c.data.distributed_sampler = True

        # set default paths based on directory config
        path_config_file = Path('cinevn_dirs.yaml')
        c.data.data_path = fetch_dir('data_path', path_config_file)
        c.data.coil_sens_path = fetch_dir('coil_sens_path', path_config_file)
        c.trainer.default_root_dir = fetch_dir('log_path', path_config_file) / c.name
        c.data.dataset_cache_file = fetch_dir('cache_path', path_config_file) / 'dataset_cache.pkl'

        # set logging interval if not given as cli parameter
        if c.trainer.log_every_n_steps is None:
            if c.trainer.devices == 'auto' or int(c.trainer.devices) == -1:
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = int(c.trainer.devices)
            c.trainer.log_every_n_steps = 50 // max(1, num_gpus)

        # configure checkpointing in checkpoint_dir
        checkpoint_dir = c.trainer.default_root_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        c.trainer.callbacks.append(Namespace({
            'class_path': 'pytorch_lightning.callbacks.ModelCheckpoint',
            'init_args': {'dirpath': str(checkpoint_dir), 'monitor': c.callback.checkpoint_monitor, 'verbose': True,
                          'mode': c.callback.checkpoint_mode},
        }))

        # set default checkpoint if one exists in our checkpoint directory
        if c.ckpt_path is None:
            ckpt_list = sorted(checkpoint_dir.glob('*.ckpt'), key=os.path.getmtime)
            if len(ckpt_list) > 0:
                c.ckpt_path = str(ckpt_list[-1])
        if self.subcommand in ['test', 'predict'] and c.ckpt_path is None:
            raise RuntimeError('No checkpoint available')

        # logger
        c.trainer.logger = Namespace({
            'class_path': 'cinevn_pl.ContinuousTensorBoardLogger',
            'init_args': {'save_dir': c.trainer.default_root_dir},
        })

        # acceleration scheduler callback
        if c.callback.acc_target_epoch is not None:
            if len(c.transform.accelerations) > 1:
                raise NotImplementedError('Acceleration scheduling is not supported for multiple acceleration rates.')
            c.trainer.callbacks.append(Namespace({
                'class_path': 'cinevn_pl.AccelerationScheduler',
                'init_args': {'acc_target': c.transform.accelerations[0], 'target_epoch': c.callback.acc_target_epoch},
            }))

        # early stopping callback
        if c.callback.patience is not None:
            c.trainer.callbacks.append(Namespace({
                'class_path': 'pytorch_lightning.callbacks.EarlyStopping',
                'init_args': {'monitor': 'val_metrics/ssim', 'patience': c.callback.patience, 'mode': 'max'},
            }))

        # logging callback
        c.trainer.callbacks.append(Namespace({
            'class_path': 'cinevn_pl.ValImageLogger',
            'init_args': {'num_log_images': c.callback.val_log_images, 'logging_interval': c.callback.val_log_interval},
        }))

        # save reconstructions callback
        c.trainer.callbacks.append(Namespace({
            'class_path': 'cinevn_pl.SaveReconstructions',
            'init_args': {'save_gif': c.callback.save_gif},
        }))

        # warn about usage of roll_temporal
        if c.transform.roll_temporal:
            warnings.warn(
                'Temporal rolling (roll_temporal) should only be used if start and end of the sequence match (e.g. '
                'exactly one cardiac cycle)!',
                category=UserWarning,
            )

        # mask function and transform objects
        mask_class = get_mask_type_class(c.transform.mask_type)
        mask_func = Namespace({
            'class_path': mask_class.__module__ + '.' + mask_class.__qualname__,
            'init_args': {'accelerations': c.transform.accelerations},
        })
        # use random masks for train transform, fixed masks for val transform
        if c.data.train_transform is None and self.subcommand == 'fit':
            c.data.train_transform = Namespace({
                'class_path': 'cinevn.data.transforms.VarNetDataTransform',
                'init_args': {'mask_func': mask_func, 'use_seed': False, 'random_flips': c.transform.random_flips,
                              'roll_temporal': c.transform.roll_temporal, 'augment_phase': c.transform.augment_phase,
                              'augment_mask': c.transform.augment_mask, 'augment_tres': c.transform.augment_tres,
                              'field_strength': c.transform.field_strength},
            })
        if c.data.val_transform is None and self.subcommand in ['fit', 'validate']:
            c.data.val_transform = Namespace({
                'class_path': 'cinevn.data.transforms.VarNetDataTransform',
                'init_args': {'mask_func': mask_func, 'use_seed': True, 'random_flips': False, 'roll_temporal': False,
                              'augment_phase': False, 'augment_mask': False, 'augment_tres': False,
                              'field_strength': c.transform.field_strength},
            })
        if c.data.test_transform is None and self.subcommand == 'test':
            c.data.test_transform = Namespace({
                'class_path': 'cinevn.data.transforms.VarNetDataTransform',
                'init_args': {'mask_func': None, 'use_seed': True, 'random_flips': False, 'roll_temporal': False,
                              'augment_phase': False, 'augment_mask': False, 'augment_tres': False,
                              'field_strength': None},
            })

        # test split
        if self.subcommand == 'test':
            if c.data.test_split is None:
                # determine test split from mask type and acceleration
                if len(c.transform.accelerations) > 1:
                    raise ValueError('Multiple acceleration rates are not supported for automatic test split naming.')
                c.data.test_split = f'test_{c.transform.mask_type}_{c.transform.accelerations[0]:02d}'
            elif not c.data.test_split.startswith('test'):
                # prepend 'test' to test split name if not already present
                c.data.test_split = f'test_{c.data.test_split}'

        # float32 matrix multiplication precision
        torch.set_float32_matmul_precision(c.float32_matmul_precision)

    @staticmethod
    def add_main_arguments(parser: ArgumentParser | LightningArgumentParser):
        parser.add_argument('--optuna', type=int,
                            help='Number of optuna trials')
        parser.add_argument('--name', type=str, default='dummy',
                            help='Experiment name. If --optuna is given, this is the study name and the experiment '
                                 'names will be composed of the study name and a consecutive number.')


def fetch_dir(key: str, data_config_file: str | Path = 'cinevn_dirs.yaml') -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `data_path`,
    `coil_sens_path`, `log_path`, and `cache_path` and this function will
    retrieve the requested path.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("data_path", "coil_sens_path", "log_path", "cache_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            'data_path': '/path/to/data',
            'coil_sens_path': '/path/to/coil_sensitivities',
            'log_path': '.',
            'cache_path': '.',
        }
        with open(data_config_file, 'w') as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warnings.warn(
            f'Path config at {data_config_file.resolve()} does not exist. A template has been created for you. Please '
            f'enter the directory paths for your system to have defaults.',
            category=UserWarning,
        )
    else:
        with open(data_config_file, 'r') as f:
            data_dir = yaml.safe_load(f)[key]

    # expand environment variables
    data_dir = os.path.expandvars(data_dir)

    return Path(data_dir)
