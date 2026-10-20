"""check_fairness.py — 检查对比的公平性

发现的问题：
  1. 我在 final_experiment.py 里写的"视觉"是极简 ORB+PnP（46.48cm）
  2. rgbd-slam pipeline 的视觉前端是完整实现（12.06cm）
  3. 拿极简视觉 vs 完整激光，或者完整视觉 vs 极简激光，都不公平

本脚本：明确两种口径，分别对比。
"""
import os
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np

    print("=" * 76)
    print("对比公平性检查")
    print("=" * 76)
    print("""
    【问题】之前的对比混用了不同的前端实现水平：

    配置 A（我的 final_experiment.py）：
      视觉 = 极简 ORB + BFMatch + PnP（200 个匹配，无关键帧，无地图点复用）
      激光 = 深度点云 ICP（逐帧配准，无关键帧）

    配置 B（rgbd-slam pipeline.py + 我的激光）：
      视觉 = 完整前端（关键帧、地图点、重投影筛选、滑动窗口 BA）
      激光 = 深度点云 ICP（逐帧配准，无关键帧）

    【结论】这两套不能混着比。

    正确做法是分两组：
     组 1：极简视觉 vs 极简激光（都无任何后端优化）
     组 2：完整视觉 vs 什么？

    组 2 里激光没有对应的"完整前端"实现（关键帧管理+局部地图+回环检索），
    所以无法做严格对比。
    """)

    print("=" * 76)
    print("可用的数字")
    print("=" * 76)

    res_file = os.path.join("results", "final_results.txt")
    lidar = {}
    if os.path.exists(res_file):
        with open(res_file) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    lidar[k] = v

    vf = os.path.join("results", "visual_vs_lidar_desk.txt")
    vis = {}
    if os.path.exists(vf):
        with open(vf) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    vis[k] = v

    print("\n【组 1】极简前端对比（我的实现，596 帧）")
    print(f"  视觉（ORB+PnP，无后端）: {lidar.get('visual_ATE_cm', '?')} cm")
    print(f"  激光（ICP，无后端）:     {lidar.get('lidar_ATE_cm', '?')} cm")
    print(f"  激光 + 回环:             {lidar.get('lidar_loop_ATE_cm', '?')} cm")

    print("\n【组 2】rgbd-slam 完整视觉（613 帧）")
    print(f"  视觉前端:  {vis.get('visual_front_ATE_cm', '?')} cm")
    print(f"  视觉 + BA: {vis.get('visual_BA_ATE_cm', '?')} cm")

    print("\n【注意】")
    print("  组 1 的视觉 46.48cm 是我写的极简版，不是 rgbd-slam 的水平。")
    print("  组 2 的视觉 12cm 才是 rgbd-slam 的真实水平。")
    print()
    print("  所以「视觉 vs 激光」的正确结论只能是：")
    print("    极简视觉 46.48 vs 激光 45.89 —— 打平（组 1，同实现水平）")
    print("    完整视觉 11.87 vs 激光 ???   —— 激光缺完整前端，无法比")


if __name__ == "__main__":
    main()
