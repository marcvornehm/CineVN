#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import argparse
import sys
from pathlib import Path

from helpers.mri import pics_impl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=Path,
                        help='Path to a directory with input .h5 files containing k-space data. Unless this data is '
                             'already undersampled, the optional parameters acceleration and mask_type should be '
                             'provided.')
    parser.add_argument('coil_sens_dir', type=Path,
                        help='Path to a directory with coil sensitivity maps')
    parser.add_argument('regularizer', type=str, nargs='+',
                        help='Regularization term(s) in BART PICS format. Examples: `T:1024:0:1e-5`: temporal total '
                             'variation; `W:3:0:1e-5` for spatial L1-Wavelet regularization. For parameter '
                             'optimization, only the first two elements of the string are required, e.g. `T:1024`.')
    parser.add_argument('--output_dir', type=Path, required='--optuna' not in sys.argv,
                        help='Path to a directory to store the reconstructions. Required if `--optuna` is not given.')
    parser.add_argument('--acceleration', '--accelerations', nargs='*', type=int,
                        help='Acceleration factors. Multiple parameters only allowed if `--optuna` is given.')
    parser.add_argument('--acceleration_range', type=int, default=0,
                        help='Range of acceleration factors to use. If set, acceleration factors are chosen randomly '
                             'from the range given by acceleration and acceleration_range. Mutually exclusive with '
                             'passing more than one parameter to either `--acceleration` or `--mask_type`. Only '
                             'allowed if `--optuna` is given.')
    parser.add_argument('--acceleration_range_step', type=int, default=1,
                        help='Step size for range of acceleration factors to use. Ignored if `--acceleration_range` is '
                             'not given. Only allowed if `--optuna` is given.')
    parser.add_argument('--mask_type', '--mask_types', nargs='*', type=str,
                        choices=['random', 'random_acs', 'vista', 'gro', 'cava', 'equispaced', 'equispaced_acs'],
                        help='Type of k-space mask. Multiple parameters only allowed if `--optuna` is given.')
    parser.add_argument('--optuna', type=str, choices=['individual', 'average'],
                        help='Use optuna to tune the regularization weight for each dataset.')
    parser.add_argument('--study_name', type=str, required='--optuna' in sys.argv,
                        help='Name of the optuna study. Required if `--optuna` is given and ignored otherwise.')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of optuna trials to run. Ignored if `--optuna` is not given.')
    parser.add_argument('--keep_phase_oversampling', action='store_true',
                        help='If set, does not crop phase oversampling. Ignored if `--optuna` is given.')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Enable use of gpu (default).')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                        help='Disable use of gpu.')
    parser.add_argument('--save_gif', action='store_true',
                        help='Save gif. Ignored if `--optuna` is given.')

    # process subsampling arguments
    args = parser.parse_args()
    if args.acceleration is not None and args.mask_type is not None:
        if args.acceleration_range > 0:
            assert len(args.acceleration) == 1
            assert len(args.mask_type) == 1
            acceleration_range = range(-args.acceleration_range, args.acceleration_range + 1, args.acceleration_range_step)
            args.acceleration = [args.acceleration[0] + i for i in acceleration_range]
        num_subsamplings = max(len(args.acceleration), len(args.mask_type))
        assert len(args.acceleration) in [1, num_subsamplings]
        assert len(args.mask_type) in [1, num_subsamplings]
        if len(args.acceleration) == 1:
            args.acceleration *= num_subsamplings
        if len(args.mask_type) == 1:
            args.mask_type *= num_subsamplings
        subsampling_args = []
        for acc, mt in zip(args.acceleration, args.mask_type):
            subsampling_args.append({'acceleration': acc, 'mask_type': mt})
    elif args.acceleration is not None or args.mask_type is not None:
        raise ValueError('Either all or none of `--acceleration` and `--mask_type` must be given.')
    else:
        subsampling_args = None

    if args.optuna is not None:
        assert subsampling_args is not None, 'Optuna requires subsampling parameters to be given.'
        pics_impl.optimize_pics_params(
            args.optuna, args.study_name, args.input_dir, args.coil_sens_dir, args.regularizer, subsampling_args,
            use_gpu=args.gpu, n_trials=args.trials,
        )
    else:
        assert subsampling_args is None or len(subsampling_args) == 1
        subsampling_args_ = subsampling_args[0] if subsampling_args is not None else None
        pics_impl.reconstruct_dir(
            args.input_dir, args.coil_sens_dir, args.regularizer, out_dir=args.output_dir,
            subsampling_options=subsampling_args_, keep_phase_oversampling=args.keep_phase_oversampling,
            use_gpu=args.gpu, write_h5=True, write_gif=args.save_gif,
        )
