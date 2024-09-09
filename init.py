import argparse
import logging
from pathlib import Path

def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir/ log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addFilter(fh)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--model-dir', type=str, default='../models/model/')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=20, help='Size of each batch in training')
    
    parser.add_argument('--mv-feature', type=int, default=512)
    parser.add_argument('--residual-feature', type=int, default=1024)
    parser.add_argument('--partition-feature', type=int, default=1280)
    parser.add_argument('--dcavg-feature', type=int, default=256)
    parser.add_argument('--qp-feature', type=int, default=256)

    #inference
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate',type=int,default=15)
    parser.add_argument('--source',type=str,default=None)
    parser.add_argument('--save-path',type=str, default=None)
    parser.add_argument('--model-path',type=str,default=None)

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args

