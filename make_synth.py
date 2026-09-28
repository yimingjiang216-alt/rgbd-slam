# -*- coding: utf-8 -*-
"""
生成合成测试序列：相机沿直线前进，拍摄一个随机 3D 点云场景。
产生 N 张 640x480 灰度图，用于验证 vo.py 能跑通。
"""
import os
import numpy as np
import cv2

def make_sequence(out_dir, N=40, W=640, H=480, f=500.0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # 生成一个前方空间里的随机 3D 点云（墙壁/地标感）
    n_pts = 800
    X = rng.uniform(-3, 3, n_pts)
    Y = rng.uniform(-2, 2, n_pts)
    Z = rng.uniform(5, 25, n_pts)
    pts3d = np.stack([X, Y, Z], axis=1)

    cx, cy = W / 2.0, H / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    for i in range(N):
        # 相机沿 Z 轴前进，步进 0.3m
        tz = i * 0.3
        img = np.zeros((H, W), dtype=np.uint8)
        # 世界点相对相机：相机在 (0,0,-tz)，所以深度 = Z + tz
        Pc = pts3d + np.array([0, 0, tz])
        # 投影
        u = f * Pc[:, 0] / Pc[:, 2] + cx
        v = f * Pc[:, 1] / Pc[:, 2] + cy
        valid = (Pc[:, 2] > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for uu, vv in zip(u[valid], v[valid]):
            cv2.circle(img, (int(uu), int(vv)), 2, 255, -1)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), img)
    print("生成", N, "张图 ->", out_dir)

if __name__ == "__main__":
    make_sequence("C:/Users/r26304/Documents/codex-shop/slam_vo/synth")
