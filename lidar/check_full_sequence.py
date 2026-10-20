"""check_full_sequence.py — 检查完整 TUM 序列是否有闭环

之前只看了前 300 帧（不闭合）。完整序列可能有回访。
"""
import os
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np

    D = os.path.join("tum", "extract", "rgbd_dataset_freiburg1_desk")


    def read_gt(path):
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                v = [float(x) for x in line.split()]
                items.append((v[0], np.array(v[1:4])))
        return items


    def read_rgb_ts(path):
        ts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ts.append(float(line.split()[0]))
        return ts


    gt = read_gt(os.path.join(D, "groundtruth.txt"))
    rgb_ts = read_rgb_ts(os.path.join(D, "rgb.txt"))

    print("=" * 70)
    print("完整 TUM fr1/desk 轨迹分析")
    print("=" * 70)
    print(f"  groundtruth: {len(gt)} 帧")
    print(f"  rgb:         {len(rgb_ts)} 帧")

    # 把 RGB 帧对齐到 gt
    gt_arr = np.array([g[1] for g in gt])
    gt_ts = np.array([g[0] for g in gt])

    P = []
    for t in rgb_ts:
        idx = np.argmin(np.abs(gt_ts - t))
        if abs(gt_ts[idx] - t) < 0.02:
            P.append(gt_arr[idx])
    P = np.array(P)

    print(f"  对齐后:      {len(P)} 帧")

    path = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    print(f"\n  起终点距离:  {np.linalg.norm(P[0]-P[-1])*100:.1f} cm")
    print(f"  轨迹总长:    {path:.2f} m")

    # 回访检测：对每一帧，找 frame_index 更晚、但位置 < 25cm 的帧
    print()
    print("=" * 70)
    print("回访检测（位置 < 25cm，且时间间隔 > 3 秒）")
    print("=" * 70)

    fps_est = len(P) / (rgb_ts[-1] - rgb_ts[0])
    min_gap_frames = int(3.0 * fps_est)
    print(f"  估计帧率: {fps_est:.1f} Hz, 3 秒 = {min_gap_frames} 帧")

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])

    visits = []
    for i in range(len(P)):
        idx = tree.query_ball_point(P[i], r=0.25)
        for j in idx:
            if j >= i + min_gap_frames:
                d = np.linalg.norm(P[i] - P[j])
                path_d = abs(cum[j] - cum[i])
                visits.append((i, j, d, path_d))

    print(f"  回访对数: {len(visits)}")
    if visits:
        ds = [v[2] for v in visits]
        pds = [v[3] for v in visits]
        print(f"  位置距离: 中位 {np.median(ds)*100:.1f} cm")
        print(f"  路程间隔: 中位 {np.median(pds):.2f} m")
        print()
        print(f"{'帧i':>6}{'帧j':>6}{'位置距':>10}{'路程':>10}")
        # 按路程间隔排序，看最"长"的回访
        for v in sorted(visits, key=lambda x: -x[3])[:10]:
            print(f"{v[0]:>6}{v[1]:>6}{v[2]*100:>9.1f}cm{v[3]:>9.2f}m")

    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    if len(visits) > 10:
        print(f"  ✅ 有 {len(visits)} 对回访，可以测回环")
        print(f"     最大路程间隔 {max(v[3] for v in visits):.2f} m")
    else:
        print(f"  ❌ 回访太少（{len(visits)}），不适合测回环")
        print("     需要换 fr2/desk 或自建闭环序列")


if __name__ == "__main__":
    main()
