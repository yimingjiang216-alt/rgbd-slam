"""
paths.py — 统一路径配置

所有脚本通过这里取路径，不各自硬编码。
可用环境变量覆盖：
    TUM_DESK_DIR    TUM fr1/desk 数据目录
    KITTI_DIR       KITTI raw drive 0001 目录
    LIDAR_RESULTS   结果输出目录
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)          # rgbd-slam/

# TUM RGB-D 数据（默认放在仓库同级的 tum/extract/ 下）
TUM_DESK = os.environ.get(
    "TUM_DESK_DIR",
    os.path.join(os.path.dirname(_REPO), "tum", "extract",
                 "rgbd_dataset_freiburg1_desk"))

# KITTI raw（默认放在 lidar/data/kitti/ 下）
KITTI = os.environ.get(
    "KITTI_DIR",
    os.path.join(_HERE, "data", "kitti",
                 "2011_09_26", "2011_09_26_drive_0001_sync"))

# 结果输出
RESULTS = os.environ.get("LIDAR_RESULTS", os.path.join(_HERE, "results"))

# 图片输出
FIGURES = os.path.join(_HERE, "figures")


def ensure_dirs():
    """创建输出目录"""
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)


def check_tum():
    """检查 TUM 数据是否就位，返回 (ok, 说明)"""
    if not os.path.isdir(TUM_DESK):
        return False, f"TUM 数据目录不存在: {TUM_DESK}"
    for f in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        if not os.path.exists(os.path.join(TUM_DESK, f)):
            return False, f"缺少 {f}"
    return True, "OK"


def check_kitti():
    """检查 KITTI 数据是否就位"""
    if not os.path.isdir(KITTI):
        return False, f"KITTI 数据目录不存在: {KITTI}"
    if not os.path.isdir(os.path.join(KITTI, "velodyne_points", "data")):
        return False, "缺少 velodyne_points/data"
    return True, "OK"


def tum_rel(*parts):
    """拼 TUM 数据的相对路径（depth/xxx.png -> 完整路径）"""
    return os.path.join(TUM_DESK, *parts)


def imread_any(path, flags=None):
    """跨平台安全读图：绕开 cv2.imread 在 Windows 上读不了含中文/非 ASCII 路径的问题

    cv2.imread 内部用窄字符 fopen，路径里有中文就失败（即使文件存在）。
    np.fromfile 走 Python 的文件 API，能正确打开；再用 imdecode 解码。
    """
    import numpy as np
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return None
    try:
        import cv2
    except ImportError:
        return None
    if flags is None:
        flags = cv2.IMREAD_UNCHANGED
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"仓库根:  {_REPO}")
    print(f"TUM:     {TUM_DESK}  -> {check_tum()}")
    print(f"KITTI:   {KITTI}  -> {check_kitti()}")
    print(f"结果:    {RESULTS}")
    print(f"图片:    {FIGURES}")
