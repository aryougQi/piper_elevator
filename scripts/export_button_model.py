#!/usr/bin/env python3
"""Export an explicitly trusted PyTorch elevator model to ONNX.

PyTorch checkpoints use Python serialization and can execute code while they
are loaded. Run this utility only for a checkpoint whose source you trust.
"""

import argparse
from pathlib import Path
import shutil
import tempfile

import onnx
from ultralytics import YOLO


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1280,
        help='Fixed square ONNX input size (default: 1280).',
    )
    parser.add_argument(
        '--i-trust-this-pytorch-checkpoint',
        action='store_true',
        help='Required acknowledgement of PyTorch serialization risk.',
    )
    return parser.parse_args()


def main() -> None:
    """Convert the trusted checkpoint and validate the resulting ONNX."""
    arguments = parse_arguments()
    if not arguments.i_trust_this_pytorch_checkpoint:
        raise SystemExit(
            'Refusing to load the checkpoint without the explicit trust flag.'
        )
    if not arguments.source.is_file():
        raise SystemExit(f'Checkpoint does not exist: {arguments.source}')
    if arguments.imgsz <= 0:
        raise SystemExit('--imgsz must be positive')

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='button_model_export_') as folder:
        temporary_checkpoint = Path(folder) / arguments.source.name
        shutil.copy2(arguments.source, temporary_checkpoint)
        model = YOLO(str(temporary_checkpoint))
        exported_path = Path(
            model.export(
                format='onnx',
                imgsz=arguments.imgsz,
                opset=12,
                simplify=False,
                dynamic=False,
                device='cpu',
            )
        )
        exported_model = onnx.load(str(exported_path))
        onnx.checker.check_model(exported_model)
        shutil.copy2(exported_path, arguments.output)

    print(f'Validated ONNX model: {arguments.output}')


if __name__ == '__main__':
    main()
