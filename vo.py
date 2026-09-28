# -*- coding: utf-8 -*-
"""
视觉里程计前端 (Visual Odometry) — 第一版最小可跑
流程：ORB 特征提取 -> 特征匹配 -> 本质矩阵估计位姿 -> 三角化建点
纯 CPU，OpenCV 实现。

用法：
    python vo.py <图片目录> [fx cx cy]
    默认用 KITTI 内参；合成测试用: python vo.py synth 500 320 240
"""
import sys
import os
import glob
import numpy as np
import cv2

print("OpenCV", cv2.__version__)


def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)


def main(img_dir, K):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) < 2:
        print("需要至少 2 张图片，当前:", len(imgs)); return

    orb = cv2.ORB_create(nfeatures=2000)
    pose = np.eye(4, dtype=np.float64)
    poses = [pose.copy()]
    landmarks = []

    prev_img = cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(prev_img, None)

    for i in range(1, len(imgs)):
        cur_img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        kp2, des2 = orb.detectAndCompute(cur_img, None)

        if des1 is None or des2 is None:
            prev_img, kp1, des1 = cur_img, kp2, des2; continue

        matches = match_features(des1, des2)
        if len(matches) < 8:
            prev_img, kp1, des1 = cur_img, kp2, des2; continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None:
            prev_img, kp1, des1 = cur_img, kp2, des2; continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # 三角化（只取 RANSAC 内点）
        inl = mask.ravel() == 1
        p1 = pts1[inl].T.astype(np.float64)   # (2, n)
        p2 = pts2[inl].T.astype(np.float64)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, p1, p2)
        pts3d = (pts4d[:3] / pts4d[3]).T
        landmarks.extend(pts3d.tolist())

        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.ravel()
        pose = pose @ T_rel
        poses.append(pose.copy())

        print(f"[{i}/{len(imgs)-1}] 匹配 {len(matches)}, 内点 {int(mask.sum())}")

        prev_img, kp1, des1 = cur_img, kp2, des2

    traj = np.array([p[:3, 3] for p in poses])
    np.savetxt(os.path.join(img_dir, "trajectory.txt"), traj)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 2], 'b-', linewidth=2, label="estimated")
    plt.scatter(traj[0, 0], traj[0, 2], c='red', s=80, label="start")
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title("VO trajectory"); plt.legend(); plt.grid(True); plt.axis("equal")
    out_png = os.path.join(img_dir, "trajectory.png")
    plt.savefig(out_png, dpi=150)
    print("轨迹图:", out_png, "| 总帧数:", len(poses))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python vo.py <图片目录> [fx cx cy]"); sys.exit(1)
    if len(sys.argv) >= 5:
        fx, cx, cy = map(float, sys.argv[2:5])
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    else:
        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]], dtype=np.float64)
    main(sys.argv[1], K)
