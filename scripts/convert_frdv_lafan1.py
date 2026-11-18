"""Convert FRDV G1 qpos NPZ files to SuperMimic motion NPZ format."""
from pathlib import Path
import argparse
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    args=ap.parse_args(); args.dst.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.src.glob('*.npz')):
        d=np.load(src, allow_pickle=True)
        q=d['qpos'].astype(np.float32); fps=float(d['fps']); dt=1.0/fps
        root_pos=q[:, :3]; root_quat=q[:, 3:7]; joint_pos=q[:, 7:]
        joint_vel=np.gradient(joint_pos, dt, axis=0).astype(np.float32)
        root_lin_vel=np.gradient(root_pos, dt, axis=0).astype(np.float32)
        # FRDV outputs quaternion poses. Angular velocity is unused by the motion
        # loader for targets, so keep a deterministic zero field for compatibility.
        root_ang_vel=np.zeros_like(root_lin_vel)
        out=args.dst/src.name
        np.savez_compressed(out, fps=np.float32(fps),
            joint_names=d['joint_names'][1:], joint_pos=joint_pos,
            joint_vel=joint_vel, root_pos_w=root_pos, root_quat_w=root_quat,
            root_lin_vel_w=root_lin_vel, root_ang_vel_w=root_ang_vel)
    print(f'converted {len(list(args.src.glob("*.npz")))} clips to {args.dst}')

if __name__ == '__main__': main()
