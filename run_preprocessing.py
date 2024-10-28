"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import argparse
import logging
from pathlib import Path

from helpers import Preprocessing

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s|%(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('data_source', type=str,
                        help='Type of data that will be processed (only `ocmr` allowed for now)')
    parser.add_argument('raw_dir', type=Path,
                        help='Dir containing raw input files')
    parser.add_argument('target_dir', type=Path,
                        help='Dir where processed data samples are stored')
    parser.add_argument('coil_sens_dir', type=Path,
                        help='Dir where coil sensitivities are located or will be stored, if computed')
    parser.add_argument('--cs_dir', nargs='?', type=Path,
                        help='Dir where CS reconstructions are located or will be stored, if computed')
    parser.add_argument('--csv_query', nargs='*', type=str, default=None,
                        help='Pandas query to filter data set')
    parser.add_argument('--csv_idx_range', nargs='*', type=int, default=[0, None],
                        help='Indices of data samples to process. (start) | (start, stop)')
    parser.add_argument('--accelerations', '--acceleration', nargs='+', type=int, default=None,
                        help='Acceleration rates to use for retrospective undersampling')
    parser.add_argument('--mask_types', '--mask_type', nargs='+', type=str, default=None,
                        choices=['random', 'random_acs', 'vista', 'gro', 'cava', 'equispaced', 'equispaced_acs'],
                        help='Types of k-space masks to use for retrospective undersampling')
    parser.add_argument('--coil_sens_backend', type=str, choices=['sigpy', 'bart'], default='sigpy',
                        help='Which backend to use to calc coil sensitivity maps')
    parser.add_argument('--n_maps', type=int, default=1,
                        help='Number of coil sensitivity maps to estimate')
    parser.add_argument('--no_cuda', dest='try_cuda', action='store_false',
                        help='Disable cuda even if available')
    args = parser.parse_args()

    # process args
    if args.csv_query:
        args.csv_query = ' and '.join(args.csv_query)

    if len(args.csv_idx_range) == 0:
        args.csv_idx_range = [0, None]
    elif len(args.csv_idx_range) == 1:
        args.csv_idx_range = [args.csv_idx_range[0], None]
    elif len(args.csv_idx_range) != 2:
        raise ValueError(f'Invalid csv_idx_range: {args.csv_idx_range}')

    # print values of all arguments
    logging.info(f'Arguments: ')
    for k, v in vars(args).items():
        if '__' in k:
            continue
        logging.info(f'{k}: {v}')

    # run preprocessing
    preprocessing = Preprocessing(
        Path(__file__).parent / 'data_attributes' / f'{args.data_source}.csv',
        args.data_source,
        args.raw_dir,
        args.target_dir,
        args.coil_sens_dir,
        cs_dir=args.cs_dir,
        csv_query=args.csv_query,
        csv_idx_range=args.csv_idx_range,
        accelerations=args.accelerations,
        mask_types=args.mask_types,
        coil_sens_backend=args.coil_sens_backend,
        n_maps=args.n_maps,
        try_cuda=args.try_cuda,
    )
    preprocessing.run()
