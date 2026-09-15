#!/usr/bin/env python3
"""Compare identical SAM2 live-frame replays; never connects to robot controls."""
import argparse
from pathlib import Path
import os
import sys
from glob import glob

# Same Torch/cuDNN isolation as the SAM2 ROS launch.
if not os.environ.get('PIPER_SAM2_BENCH_LIBS'):
    libraries = [p for root in sys.path for p in glob(root + '/nvidia/*/lib')]
    env = dict(os.environ, PIPER_SAM2_BENCH_LIBS='1')
    env['LD_LIBRARY_PATH'] = os.pathsep.join(libraries + [env.get('LD_LIBRARY_PATH', '')])
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

import time
import json
import cv2
import numpy as np
import torch
from piper_elevator_app.sam2_button_tracker import _Sam2Backend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=['eager', 'threads', 'encoder', 'vos', 'compiled', 'production'], default='eager')
parser.add_argument('--frames', type=int, default=70)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/sam2_speed')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if args.variant in ('threads', 'encoder', 'vos', 'compiled', 'production'):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
if args.variant in ('encoder', 'vos', 'compiled'):
    import sam2.build_sam as builder
    original = builder.build_sam2_video_predictor
    def build(*a, **kw):
        if args.variant == 'vos':
            kw['vos_optimized'] = True
        elif args.variant == 'encoder':
            kw['hydra_overrides_extra'] = ['++model.compile_image_encoder=true']
        predictor = original(*a, **kw)
        if args.variant == 'compiled':
            for name in ('image_encoder', 'memory_encoder', 'memory_attention', 'sam_prompt_encoder', 'sam_mask_decoder'):
                module = getattr(predictor, name)
                module.forward = torch.compile(module.forward, mode='max-autotune-no-cudagraphs',
                                               fullgraph=True, dynamic=name == 'memory_attention')
        return predictor
    builder.build_sam2_video_predictor = build

source = Path(__file__).resolve().parents[1] / 'data/coarse_detection_loss/color.png'
frame = cv2.imread(str(source))
assert frame is not None
h, w = frame.shape[:2]
started = time.perf_counter()
backend = _Sam2Backend('configs/sam2.1/sam2.1_hiera_t.yaml', '/opt/sam2/checkpoints/sam2.1_hiera_tiny.pt', 'cuda', compile_model=args.variant=='production')
mask = backend.initialize(frame, np.array([474., 270., 510., 308.]))
torch.cuda.synchronize()
init_seconds = time.perf_counter() - started
print(f'INITIALIZED {args.variant} seconds={init_seconds:.3f}', flush=True)
elapsed = []
masks = []
for i in range(args.frames):
    transform = cv2.getRotationMatrix2D((491., 289.), 2*np.sin(i/20), 1 + 0.15*i/args.frames)
    transform[:, 2] += [8*np.sin(i/15), 4*np.sin(i/17)]
    sample = cv2.warpAffine(frame, transform, (w, h))
    torch.cuda.synchronize()
    tick = time.perf_counter()
    if args.variant == 'vos':
        torch.compiler.cudagraph_mark_step_begin()
    mask = backend.track(sample)
    torch.cuda.synchronize()
    elapsed.append(time.perf_counter() - tick)
    masks.append(mask)
    if i % 20 == 0:
        print(f'FRAME {i} ms={1000*elapsed[-1]:.2f}', flush=True)
values = np.array(elapsed[10:])
result = dict(variant=args.variant, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
              threads=torch.get_num_threads(), image_size=backend._predictor.image_size,
              source=str(source), frames=args.frames, warmup_frames=10,
              initialization_seconds=init_seconds, mean_ms=float(values.mean()*1000),
              p95_ms=float(np.percentile(values, 95)*1000), fps=float(1/values.mean()),
              peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,
              elapsed_seconds=elapsed)
baseline = args.output / 'eager_masks.npz'
if baseline.exists() and args.variant != 'eager':
    other = np.load(baseline)['masks']
    ious = []
    center_errors = []
    for a, b in zip(other, masks):
        ious.append(float(np.count_nonzero(a & b)/max(1,np.count_nonzero(a | b))))
        if a.any() and b.any():
            center_errors.append(float(np.linalg.norm(np.array(np.nonzero(a)).mean(axis=1)-np.array(np.nonzero(b)).mean(axis=1))))
    result.update(mask_iou_min=min(ious), mask_iou_mean=float(np.mean(ious)),
                  center_error_max_px=max(center_errors), center_error_mean_px=float(np.mean(center_errors)))
np.savez_compressed(args.output / f'{args.variant}_masks.npz', masks=np.array(masks))
(args.output / f'{args.variant}.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='elapsed_seconds'}, indent=2), flush=True)
