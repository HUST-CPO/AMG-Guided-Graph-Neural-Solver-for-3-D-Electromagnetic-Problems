"""DDP entry for the final AMG edge-attention model."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import os
from training.config import DEFAULT_TARGET_GPUS
from training.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description='Final AMG edge-attention DDP training')
    parser.add_argument('--mode', choices=['ddp'], default='ddp')
    parser.add_argument('--gpu', '--gpus', dest='gpus', help='Visible GPU ids, comma separated')
    args = parser.parse_args()
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', DEFAULT_TARGET_GPUS)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('NCCL_TIMEOUT', '3600000')
    os.environ.setdefault('NCCL_IB_TIMEOUT', '3600000')
    os.environ.setdefault('NCCL_HEARTBEAT_TIMEOUT_SEC', '300')
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '0')
    os.chdir(PROJECT_ROOT)
    from training.trainddp import main as train
    train()


if __name__ == '__main__':
    main()
