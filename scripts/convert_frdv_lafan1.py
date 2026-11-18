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
        # Derive world-frame angular velocity from consecutive wxyz quaternions.
        q = root_quat / np.linalg.norm(root_quat, axis=1, keepdims=True).clip(1e-8)
        rel = np.empty_like(q)
        rel[0] = np.array([1, 0, 0, 0], dtype=np.float32)
        a, b = q[:-1], q[1:]
        rel[1:, 0] = b[:, 0] * a[:, 0] + np.sum(b[:, 1:] * a[:, 1:], axis=1)
        rel[1:, 1:] = (b[:, 0, None] * a[:, 1:] - a[:, 0, None] * b[:, 1:]
                       - np.cross(b[:, 1:], a[:, 1:]))
        rel /= np.linalg.norm(rel, axis=1, keepdims=True).clip(1e-8)
        sign = np.where(rel[:, :1, 0] < 0, -1.0, 1.0)
        root_ang_vel = (2.0 * rel[:, 1:] * sign / dt).astype(np.float32)
        out=args.dst/src.name
        np.savez_compressed(out, fps=np.float32(fps),
            joint_names=d['joint_names'][1:], joint_pos=joint_pos,
            joint_vel=joint_vel, root_pos_w=root_pos, root_quat_w=root_quat,
            root_lin_vel_w=root_lin_vel, root_ang_vel_w=root_ang_vel)
    print(f'converted {len(list(args.src.glob("*.npz")))} clips to {args.dst}')

if __name__ == '__main__': main()
