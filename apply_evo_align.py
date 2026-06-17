import numpy as np
import os
import argparse
import sys

# ================= 配置区域 (保持不变) =================

# 2. EVO 对齐参数
# Scale correction
s = 9.197209227952124

# Rotation of alignment (3x3)
R_align = np.array([
    [-0., -0., 1.],
    [-1., -0., -0.],
    [0., -1., -0.]
])

# Translation of alignment (1x3)
t_align = np.array([0., 0., 0.])


# ====================================================

def load_kitti_poses(file_path):
    """读取KITTI格式文件"""
    poses = []
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        sys.exit(1)

    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            # 补全为4x4
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose
            poses.append(pose_4x4)
    return poses


def save_kitti_poses(file_path, poses):
    """保存为KITTI格式"""
    with open(file_path, 'w') as f:
        for pose in poses:
            # 取前3行，展平
            flat = pose[:3, :].flatten()
            line = " ".join([f"{x:.6e}" for x in flat])
            f.write(line + "\n")


def apply_alignment(poses, R, t, s):
    """
    应用 Sim(3) 变换
    """
    aligned_poses = []

    print(f"正在处理 {len(poses)} 帧轨迹...")
    print(f"应用尺度: {s}")
    # print(f"应用旋转:\n{R}")
    # print(f"应用平移: {t}")
    ts = np.eye(4)
    ts[:3, :3] = R
    ts[:3, 3] = t

    for pose in poses:
        # 分离原始旋转和平移


        new_pose = np.linalg.inv(ts) @ pose @ ts
        aligned_poses.append(new_pose)

    return aligned_poses


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="对 KITTI 轨迹文件应用预设的 Sim(3) 变换 (对齐外参和尺度)")
    parser.add_argument("input_path", type=str, help="输入的 KITTI 格式轨迹文件路径 (.txt)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径 (可选，默认在原文件名后加 _aligned)")

    args = parser.parse_args()

    input_file = args.input_path

    # 如果没有指定输出文件名，自动生成
    if args.output is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_aligned{file_ext}"
    else:
        output_file = args.output

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 30)

    # 1. 读取
    raw_poses = load_kitti_poses(input_file)

    # 2. 变换
    aligned_poses = apply_alignment(raw_poses, R_align, t_align, s)

    # 3. 保存
    save_kitti_poses(output_file, aligned_poses)

    print("-" * 30)
    print(f"处理完成！已保存至: {output_file}")

    # 简单验证
    start_old = raw_poses[0][:3, 3]
    start_new = aligned_poses[0][:3, 3]
    print(f"起点 (原): {start_old}")
    print(f"起点 (新): {start_new}")