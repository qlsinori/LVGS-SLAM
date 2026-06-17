#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import numpy as np
import subprocess
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import colorama
import tf.transformations as tfs
class TrajectoryComparator:
    def __init__(self):
        rospy.init_node('trajectory_comparator', anonymous=True)

        # --- 1. 配置参数 ---
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.tf_target_frame = rospy.get_param('~tf_target_frame', 'odom')
        self.tf_source_frame = rospy.get_param('~tf_source_frame', 'base_link')
        self.seq = rospy.get_param('~seq', '1018_00')
        self.poll_rate = rospy.get_param('~poll_rate', 20.0) # 建议提高TF轮询频率
        
        self.odom_output_file = "/slam/bagfile/exp/botanic/limo/"+self.seq+"odom_trajectory.txt"
        self.tf_output_file = "/slam/bagfile/exp/botanic/limo/"+self.seq+"tf_trajectory_matched.txt"
        
        rospy.loginfo("订阅轨迹话题: {}".format(self.odom_topic))
        rospy.loginfo("监听TF变换: 从 {} 到 {}".format(self.tf_source_frame, self.tf_target_frame))
        rospy.loginfo("TF监听频率: {} Hz".format(self.poll_rate))

        # --- 2. 初始化数据存储和ROS工具 ---
        self.odom_poses = []
        self.tf_poses = []
        self.last_tf_stamp = None # 用于避免重复记录TF

        # 【修改】为TransformListener增加缓冲区时长，可以减少ExtrapolationException
        self.tf_listener = tf.TransformListener(True, rospy.Duration(10.0))
        
        self.path_subscriber = rospy.Subscriber(self.odom_topic, Path, self.path_callback)
        self.tf_poll_timer = rospy.Timer(rospy.Duration(1.0 / self.poll_rate), self.tf_poll_callback)
        
        self.evaluation_script_path = rospy.get_param(
            '~evaluation_script_path', 
            '/workspace/limo_ws/src/test_ape/scripts/evaluate.py'
        )
        rospy.on_shutdown(self.process_and_evaluate)

        rospy.loginfo("节点初始化完成，正在记录轨迹...")

    def path_callback(self, msg):
        rospy.loginfo_once("接收到第一个Path消息，正在解析轨迹...")
        self.odom_poses = [] 
        for pose_stamped in msg.poses:
            stamp = pose_stamped.header.stamp
            pos = pose_stamped.pose.position
            ori = pose_stamped.pose.orientation
            trans = (pos.x, pos.y, pos.z)
            rot = (ori.x, ori.y, ori.z, ori.w)
            self.odom_poses.append((stamp, trans, rot))
   

        # print(colorama.Fore.RED + "stamp: " + colorama.Style.RESET_ALL,stamp.to_sec())
        rospy.loginfo("Path消息处理完毕，当前记录了 {} 个里程计位姿。".format(len(self.odom_poses)))

    # --- 【核心修改】 ---
    def tf_poll_callback(self, event):
        """
        定时器回调函数，使用精确的TF时间戳轮询并记录TF变换。
        """
        try:
            # 1. 获取两个坐标系之间最新的可用变换时间戳
            latest_time = self.tf_listener.getLatestCommonTime(self.tf_source_frame, self.tf_target_frame)

            # 2. 检查这个时间戳是否是新的，避免重复记录
            if self.last_tf_stamp and latest_time <= self.last_tf_stamp:
            
                return # 不是新的TF数据，直接返回
            
            # 3. 使用这个精确的时间戳来查询对应的变换
            (trans, rot) = self.tf_listener.lookupTransform(
                self.tf_source_frame, self.tf_target_frame, latest_time
            )
            
            # print("trans:",trans)
            # print("rot:",rot)

            
            # 4. 使用变换的真实时间戳(latest_time)来存储数据
            self.tf_poses.append((latest_time, trans, rot))
            self.last_tf_stamp = latest_time # 更新最后记录的时间戳
  
            # print(colorama.Fore.RED + "latest_time: " + colorama.Style.RESET_ALL,latest_time.to_sec())
        except (tf.Exception) as e: # 捕获所有tf相关的异常
            rospy.logwarn_throttle(5.0, "TF监听失败: {}".format(e))
    # --------------------

    # --- 新增：保存为KITTI格式的函数 ---
    def save_to_kitti_format(self, filepath, poses):
        with open(filepath, 'w') as f:
            for _, trans, rot in poses:
                # 从四元数创建4x4变换矩阵
                matrix_4x4 = tfs.quaternion_matrix(rot)
                # 将平移向量放入矩阵的最后一列
                matrix_4x4[0:3, 3] = trans
                # 获取顶部的3x4部分
                matrix_3x4 = matrix_4x4[0:3, :]
                # 将3x4矩阵展平为12个元素的一维数组
                kitti_values = matrix_3x4.flatten()
                # 将12个数字格式化为字符串并写入文件
                line = ' '.join(['{:.6e}'.format(v) for v in kitti_values])
                f.write(line + '\n')
        rospy.loginfo("轨迹已保存为KITTI格式至: {}".format(filepath))


    def process_and_evaluate(self):
        rospy.loginfo("节点正在关闭，开始处理和评估轨迹...")

        if not self.odom_poses or not self.tf_poses:
            rospy.logerr("数据不足，无法进行评估。里程计或TF轨迹为空。")
            return

        self.tf_poses = self.normalize_trajectory(self.tf_poses)
        rospy.loginfo("正在对齐TF和Path轨迹的时间戳...")
        
        tf_timestamps = np.array([p[0].to_sec() for p in self.tf_poses])
        
        matched_tf_poses = []
        # 【修改】同时保留对齐后的odom轨迹，以确保两者长度完全一致
        aligned_odom_poses = []
        
        for odom_stamp, odom_trans, odom_rot in self.odom_poses:
            odom_time_sec = odom_stamp.to_sec()
            # print("odom_time_sec:",odom_time_sec)
            time_diffs = np.abs(tf_timestamps - odom_time_sec)
            closest_idx = np.argmin(time_diffs)
            
            # 【修改】可以适当放宽时间差阈值，例如半个TF采样周期
            max_time_diff = 0.5 * (1.0 / self.poll_rate)
            if time_diffs[closest_idx] > max_time_diff:
                rospy.logwarn("对于轨迹点时间戳 {}, 未找到足够近的TF匹配，最小时间差为 {:.4f}s。跳过此点.".format(
                    odom_time_sec, time_diffs[closest_idx]))
                continue

            time, tf_trans, tf_rot = self.tf_poses[closest_idx]
            # print("tf_time_sec:\n",tf_timestamps[closest_idx])


            matched_tf_poses.append((odom_stamp, tf_trans, tf_rot))
            # 【新增】将成功匹配的odom位姿也存起来
            aligned_odom_poses.append((odom_stamp, odom_trans, odom_rot))

        if not matched_tf_poses:
            rospy.logerr("时间戳对齐后，没有找到任何匹配的位姿对！无法评估。")
            return
            
        rospy.loginfo("对齐完成，共找到 {} 对匹配的位姿。".format(len(matched_tf_poses)))

        # --- 【核心修改】调用新的保存函数，替换旧的 ---
        self.save_to_kitti_format(self.odom_output_file, aligned_odom_poses)
        self.save_to_kitti_format(self.tf_output_file, matched_tf_poses)
        self.save_to_kitti_format(self.tf_output_file+"_ALL", self.tf_poses)

        rospy.loginfo("正在调用 EVO 计算绝对位姿误差 (APE)...")
        
        # --- 【核心修改】修改evo命令以使用kitti格式 ---
        # command = [
        #     "evo_ape", "kitti", # <-- 从 "tum" 改为 "kitti"
        #     self.odom_output_file,   # 参考轨迹
        #     self.tf_output_file,     # 评估轨迹
        #     "-v",                    # Verbose
        #     "-a",                    # Align
        #     "-p",                    # Plot
        #     "--save_results", "evo_results.zip"
        # ]
        command = [
            "python",
            self.evaluation_script_path,
            "--gt_file", self.tf_output_file,
            "--pred_file",  self.odom_output_file,
        ]
        try:
            rospy.loginfo("执行命令: {}".format(' '.join(command)))
            import subprocess
            subprocess.call(command)
            rospy.loginfo("EVO评估完成！结果图已显示，详细数据已保存到 evo_results.zip。")
        except OSError as e:
            if e.errno == 2:
                rospy.logerr("错误: 'evo_ape' 命令未找到。")
                rospy.logerr("安装指令: pip install evo --upgrade --no-binary evo")
            else:
                rospy.logerr("EVO执行出错: {}".format(e))
        except Exception as e:
             rospy.logerr("EVO执行时发生未知错误: {}".format(e))

    def normalize_trajectory(self,poses):
        """
        将轨迹的所有位姿都乘以第一个位姿的逆，使其从原点开始。
        :param poses: 一个位姿列表 [(stamp, trans, rot), ...]
        :return: 归一化后的位姿列表
        """
        if not poses:
            return []

        # 获取第一个位姿的平移和旋转
        _, first_trans, first_rot = poses[0]
        
        # 计算第一个位姿的逆变换
        # 逆旋转
        first_rot_inv = tfs.quaternion_inverse(first_rot)
        # 逆平移：先将原平移向量用逆旋转进行旋转，然后取反
        first_trans_inv = -np.array(tfs.quaternion_multiply(tfs.quaternion_multiply(first_rot_inv, list(first_trans) + [0]), first_rot)[:3])

        normalized_poses = []
        for stamp, trans, rot in poses:
            # 将当前位姿转换为numpy数组以便计算
            trans_vec = np.array(trans)
            
            # 计算新的旋转: new_rot = first_rot_inv * rot
            new_rot = tfs.quaternion_multiply(first_rot_inv, rot)
            
            # 计算新的平移: new_trans = first_trans_inv + (first_rot_inv * trans)
            # 这里的 (first_rot_inv * trans) 表示用逆旋转来旋转当前的平移向量
            rotated_trans = tfs.quaternion_multiply(tfs.quaternion_multiply(first_rot_inv, list(trans) + [0]), first_rot)[:3]
            new_trans = first_trans_inv + rotated_trans

            normalized_poses.append((stamp, tuple(new_trans), tuple(new_rot)))
            
        return normalized_poses

if __name__ == '__main__':
    try:
        TrajectoryComparator()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断。")