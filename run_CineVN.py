"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import argparse
import warnings
from typing import Callable, Sequence

import optuna
from optuna.trial import TrialState

from cinevn_pl import cli


def run_cli():
    # ignore `srun` warning from pytorch lightning
    warnings.filterwarnings(
        'ignore', message='The `srun` command is available on your system but is not used.*',
        module='lightning_fabric.plugins.environments.slurm',
    )

    # base argument parser
    parser = argparse.ArgumentParser(add_help=False)
    cli.CineVNCLI.add_main_arguments(parser)

    args = parser.parse_known_args()[0]
    if args.optuna is None:
        # regular run
        _ = cli.CineVNCLI()
    else:
        # hyperparameter tuning
        n_trials = args.optuna if args.optuna > 0 else None
        optimize_hyperparameters(objective, 'maximize', n_trials=n_trials, study_name=args.name)


def optimize_hyperparameters(
        objective: Callable,
        direction: str = 'minimize',
        n_trials: int | None = None,
        study_name: str = 'dummy',
):
    pruner = MultiplePruners(
        pruners=[
            optuna.pruners.MedianPruner(n_warmup_steps=15),
            optuna.pruners.ThresholdPruner(lower=0.9, n_warmup_steps=30),
        ],
        pruning_fun=any,
    )
    study = optuna.create_study(
        storage='sqlite:///optuna.db',
        direction=direction,
        pruner=pruner,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    best_trial = study.best_trial

    print('  Value: ', best_trial.value)

    print('  Params: ')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')


def objective(trial: optuna.Trial):
    params = {
        'model.lr': trial.suggest_float('lr', 1e-4, 1e-1, step=1e-4),
        'model.lr_step_size': trial.suggest_int('lr_step_size', 20, 60, step=5),
        'model.chans': trial.suggest_int('chans', 4, 20, step=2),
        'model.num_cascades': trial.suggest_int('num_cascades', 3, 25),
        'model.pools': trial.suggest_int('pools', 1, 3),
        'model.conv_size': trial.suggest_categorical('conv_size', [3, 5]),
        'model.conv_size_temp': trial.suggest_categorical('conv_size_temp', [3, 5, 7]),
        'model.pool_temporal': trial.suggest_categorical('pool_temporal', [True, False]),
        'name': f'{trial.study.study_name}_{trial.number:03d}',
    }
    mycli = cli.CineVNCLI(trial=trial, overwrite_args=params)

    # look for MetricMonitor callback and return its best value
    for cb in mycli.trainer.callbacks:  # type: ignore
        if type(cb).__name__ == 'MetricMonitor':
            return cb.best.item()


class MultiplePruners(optuna.pruners.BasePruner):
    """
    https://github.com/optuna/optuna/issues/2042
    """

    def __init__(self, pruners: Sequence[optuna.pruners.BasePruner], pruning_fun: Callable = any) -> None:
        self.pruners = pruners
        self.pruning_fun = pruning_fun

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return self.pruning_fun(pruner.prune(study, trial) for pruner in self.pruners)


if __name__ == "__main__":
    __spec__ = None  # required for pdb in combination with ddp
    run_cli()
