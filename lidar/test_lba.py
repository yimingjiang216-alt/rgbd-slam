"""test_lba.py — 几何 BA 正确性测试（修正版）

关键认识：
  点面残差 n^T(R pl + t - X) 存在规范自由度：
  位姿和地图点同时平移，残差不变。
  所以：
    - 必须固定第一帧（或加先验）
    - 用「相对位姿误差」评估，不用绝对误差
    - 必须同时优化位姿和点（只优化位姿不够）
"""
import os
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np

    from lba import so3_exp, so3_log, se3_exp, geometry_ba

    rng = np.random.default_rng(0)


    def make_problem(n_kf=5, n_pt=40, n_obs=200, noise_t=0.05, noise_r=0.03):
        """构造 BA 问题：真值 + 噪声初值 + 与真值一致的观测"""
        poses_gt = {}
        for i in range(n_kf):
            T = np.eye(4)
            T[:3, :3] = so3_exp(rng.normal(size=3) * 0.2)
            T[:3, 3] = np.array([i * 0.1, 0, 0]) + rng.normal(size=3) * 0.1
            poses_gt[i] = T

        points_gt = {j: rng.normal(size=3) for j in range(n_pt)}
        normals = {}
        for j in range(n_pt):
            nn = rng.normal(size=3)
            normals[j] = nn / np.linalg.norm(nn)

        obs = []
        for _ in range(n_obs):
            k = int(rng.integers(0, n_kf))
            j = int(rng.integers(0, n_pt))
            T = poses_gt[k]
            R, t = T[:3, :3], T[:3, 3]
            nw = normals[j]
            Xj = points_gt[j]
            # 在过 Xj、法向 nw 的平面上取点，保证残差为 0
            v = rng.normal(size=3)
            v -= nw * (nw @ v)
            v /= (np.linalg.norm(v) + 1e-12)
            P0 = Xj + v * rng.uniform(0.0, 0.5)
            pl = R.T @ (P0 - t)
            obs.append((k, j, pl, nw))

        poses_noisy = {}
        for i in range(n_kf):
            xi = np.zeros(6)
            xi[:3] = rng.normal(size=3) * noise_t
            xi[3:] = rng.normal(size=3) * noise_r
            poses_noisy[i] = se3_exp(xi) @ poses_gt[i]

        return poses_gt, points_gt, poses_noisy, obs


    def rel_pose_errors(P_est, poses_gt, ref=0):
        """相对第 ref 帧的位姿误差（消除规范自由度）"""
        T_ref = P_est[ref]
        Tgt_ref = poses_gt[ref]
        errs = []
        for k in sorted(poses_gt):
            if k == ref:
                continue
            rel_est = np.linalg.inv(T_ref) @ P_est[k]
            rel_gt = np.linalg.inv(Tgt_ref) @ poses_gt[k]
            errs.append(np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3]))
        return np.array(errs)


    def total_cost(poses_d, points_d, obs):
        r = []
        for (k, j, pl, nw) in obs:
            T = poses_d[k]
            pw = T[:3, :3] @ pl + T[:3, 3]
            r.append(nw @ (pw - points_d[j]))
        return float(np.sum(np.array(r) ** 2))


    print("=" * 72)
    print("测试 1: 真值处代价应为 0")
    print("=" * 72)
    poses_gt, points_gt, poses_noisy, obs = make_problem()
    c_true = total_cost(poses_gt, points_gt, obs)
    print(f"  真值代价: {c_true:.3e}")
    ok1 = c_true < 1e-16
    print(f"  {'PASS' if ok1 else 'FAIL'}")


    print()
    print("=" * 72)
    print("测试 2: 位姿误差（相对第一帧）应显著下降")
    print("=" * 72)

    # 需要给点一个初值：用观测反推（更接近真实情况）
    points_init = {}
    for j in range(40):
        obs_j = [o for o in obs if o[1] == j]
        if obs_j:
            k, _, pl, nw = obs_j[0]
            T = poses_noisy[k]
            points_init[j] = T[:3, :3] @ pl + T[:3, 3]
        else:
            points_init[j] = np.zeros(3)

    err_before = rel_pose_errors(poses_noisy, poses_gt)
    c_before = total_cost(poses_noisy, points_init, obs)
    print(f"  优化前: 位姿误差中位 {np.median(err_before)*100:.2f} cm, "
          f"代价 {c_before:.4e}")

    P_opt, X_opt, info = geometry_ba(
        obs, poses_noisy, points_init,
        kf_ids=list(range(5)), fix_first=True, max_nfev=200, min_obs=1)

    err_after = rel_pose_errors(P_opt, poses_gt)
    c_after = total_cost(P_opt, X_opt, obs)
    print(f"  优化后: 位姿误差中位 {np.median(err_after)*100:.2f} cm, "
          f"代价 {c_after:.4e}")
    print(f"  迭代 {info['nfev']}, 状态 {info['status']}")
    print(f"  关键帧 {info['nframes']}, 优化帧 {info['n_free_frames']}, "
          f"点 {info['npoints']}, 残差 {info['nres']}")

    ok2 = np.median(err_after) < np.median(err_before) * 0.5
    print(f"  {'PASS' if ok2 else 'FAIL'}: 位姿误差至少减半")


    print()
    print("=" * 72)
    print("测试 3: 固定第一帧不应改变")
    print("=" * 72)
    d0 = np.linalg.norm(P_opt[0][:3, 3] - poses_noisy[0][:3, 3])
    r0 = np.linalg.norm(P_opt[0][:3, :3] - poses_noisy[0][:3, :3])
    print(f"  第 0 帧: 位移 {d0:.2e}, 旋转 {r0:.2e}")
    ok3 = d0 < 1e-9 and r0 < 1e-9
    print(f"  {'PASS' if ok3 else 'FAIL'}")


    print()
    print("=" * 72)
    print("测试 4: 带噪声观测下的鲁棒性")
    print("=" * 72)
    obs_noisy = [(k, j, pl + nw * rng.normal() * 0.01, nw)
                 for (k, j, pl, nw) in obs]
    P_o, X_o, info_o = geometry_ba(
        obs_noisy, poses_noisy, points_init,
        kf_ids=list(range(5)), fix_first=True, max_nfev=200, min_obs=1)
    err_n = rel_pose_errors(P_o, poses_gt)
    print(f"  1cm 观测噪声下位姿误差中位: {np.median(err_n)*100:.2f} cm")
    print(f"  （优化前 {np.median(err_before)*100:.2f} cm）")
    ok4 = np.median(err_n) <= np.median(err_before) * 1.1
    print(f"  {'PASS' if ok4 else 'FAIL'}: 噪声下不明显恶化")


    print()
    print("=" * 72)
    print("汇总")
    print("=" * 72)
    res = [("真值代价为0", ok1), ("位姿误差下降", ok2), ("固定首帧", ok3),
           ("噪声鲁棒", ok4)]
    for n, ok in res:
        print(f"  {n}: {'PASS' if ok else 'FAIL'}")
    if all(o for _, o in res):
        print("\n  全部通过")
    else:
        print("\n  有失败项")
        sys.exit(1)


if __name__ == "__main__":
    main()
