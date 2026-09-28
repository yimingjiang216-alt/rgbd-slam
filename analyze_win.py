# -*- coding: utf-8 -*-
"""
分析: 为什么滑窗BA拼接后全局ATE变差
每个窗口独立优化 -> 重叠区(step<W)的帧被优化两次, 两次结果不同且都被写回
-> 后写的覆盖先写的 -> 相邻窗口在重叠处"打架" -> 整体轨迹抖动
验证: 统计同一帧被多少个窗口优化过
"""
import numpy as np, os, sys
seq=r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
F=798; W=40; step=20
count=np.zeros(F,int)
for start in range(0,F-3,step):
    end=min(start+W,F)
    if end-start<5: break
    for f in range(start,end): count[f]+=1
print("每个窗口: W=%d step=%d"%(W,step))
print("帧被优化次数分布:", dict(zip(*np.unique(count,return_counts=True))))
print("被优化 1 次的帧: %d"%(count==1).sum())
print("被优化 2 次的帧: %d"%(count==2).sum())
print()
print("问题: 重叠帧被两个窗口独立优化, 结果不一致, 直接覆盖会引入跳变")
print("修法: 让窗口内的位姿以"上一轮全局解"为先验(阻尼), 或做位姿图平滑")
