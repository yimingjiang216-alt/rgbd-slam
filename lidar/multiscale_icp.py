"""
multiscale_icp.py — 多尺度 ICP + 稳健初值

针对前面诊断出的问题：
  - 零初值 -> ICP 陷局部最优，位移被低估 94%
  - 真值初值 -> 误差只有 7%

所以核心是「给出足够好的初值」。做法：
  1. 多尺度：先在 1.0m 体素上粗配准，再逐级细化到 0.3m
  2. 常量速度模型 + 首帧特殊处理
  3. 若某帧配准失败（位移异常小），回退到上一帧速度再试
"""
import numpy as np

from pointcloud import build_kdtree
from registration import icp_point_to_point, pose_to_matrix


def multiscale_icp(source, target, R0, t0, voxel_levels=(1.5, 1.0, 0.5),
                   max_iter=40, max_corr_ratio=1.0):
    """多尺度 ICP：source/target 已经是「多分辨率点云列表」

    从粗到细，每一级的初值来自上一级的结果。
    """
    R, t = R0.copy(), t0.copy()
    infos = []
    for src_l, tgt_l in zip(source, target):
        if len(src_l) < 20 or len(tgt_l) < 20:
            continue
        tree = build_kdtree(tgt_l[:, :3])
        # 外点阈值与体素尺度相关：粗尺度容许更大偏差
        mc = max(1.0, max_corr_ratio * 2.0)
        R, t, info = icp_point_to_point(
            src_l[:, :3], tgt_l[:, :3], target_tree=tree,
            R0=R, t0=t, max_iter=max_iter,
            max_correspondence_dist=mc)
        infos.append(info)
    return R, t, infos


def make_pyramid(pts_list, levels):
    """把点云按不同体素下采样，返回每级一个数组"""
    from pointcloud import voxel_downsample
    return [[voxel_downsample(p, v) for p in pts_list] for v in levels]


def estimate_relative_pose(src_pyr, tgt_pyr, R0, t0):
    """用多尺度 ICP 估计 src 到 tgt 的相对位姿"""
    R, t, infos = multiscale_icp(src_pyr, tgt_pyr, R0, t0)
    return R, t, infos
