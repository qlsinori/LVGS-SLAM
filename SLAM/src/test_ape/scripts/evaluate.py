# -*- coding: utf-8 -*-
#
# Copyright Qing Li (hello.qingli@gmail.com) 2018. All Rights Reserved.
#
# References: 1. KITTI odometry development kit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#             2. A Geiger, P Lenz, R Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
#

import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
# 确保您的环境中已经安装了这些依赖
# pip install numpy matplotlib
try:
    import tools.transformations as tr
    from tools.pose_evaluation_utils import quat_pose_to_mat
except ImportError:
    print("错误：无法导入 'tools.transformations' 或 'tools.pose_evaluation_utils'。")
    print("请确保这些模块存在于您的项目中，或者从原始代码仓库中获取它们。")
    exit()

# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
plt.switch_backend('agg')


class kittiOdomEval():
    def __init__(self, config):
        # 保存传入的文件路径和配置
        self.gt_file = config.gt_file
        self.pred_file = config.pred_file
        self.output_dir = config.output_dir
        self.epoch = config.epoch
        
        # 从预测文件名中提取序列号
        self.seq = os.path.splitext(os.path.basename(self.pred_file))[0]

        # 检查输入文件是否存在
        assert os.path.exists(self.pred_file), "评价轨迹文件未找到: {}".format(self.pred_file)
        if self.gt_file:
            assert os.path.exists(self.gt_file), "参考轨迹文件未找到: {}".format(self.gt_file)

        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def toCameraCoord(self, pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        R_C2L = np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]])
        inv_R_C2L = np.linalg.inv(R_C2L)
        R = np.dot(inv_R_C2L, pose_mat)
        rot = np.dot(R, R_C2L)
        return rot

    def load_poses_from_txt(self, file_name, toCameraCoord):
        '''
            从KITTI格式的txt文件中加载位姿
        '''
        try:
            f = open(file_name, 'r')
            s = f.readlines()
            f.close()
            poses = {}
            for cnt, line in enumerate(s):
                P = np.eye(4)
                line_split = [float(i) for i in line.split()]
                withIdx = int(len(line_split) == 13)
                if len(line_split) not in [12, 13]:
                    print("警告：文件 {} 第 {} 行格式不正确，已跳过。".format(os.path.basename(file_name), cnt + 1))
                    continue
                
                for row in range(3):
                    for col in range(4):
                        P[row, col] = line_split[row * 4 + col + withIdx]
                
                frame_idx = int(line_split[0]) if withIdx else cnt
                
                if toCameraCoord:
                    poses[frame_idx] = self.toCameraCoord(P)
                else:
                    poses[frame_idx] = P
            return poses
        except Exception as e:
            print("加载文件 {} 时出错: {}".format(file_name, e))
            return {}

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
        '''
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        self.distance = dist[-1] if dist else 0
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    def translationError(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        self.max_speed = 0
        dist = self.trajectoryDistances(poses_gt)
        self.step_size = 10
        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

                if last_frame == -1 or not (last_frame in poses_result.keys()) or not (
                        first_frame in poses_result.keys()):
                    continue

                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)
                if speed > self.max_speed:
                    self.max_speed = speed
                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write + "\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len if seq_len > 0 else 0
        ave_r_err = r_err / seq_len if seq_len > 0 else 0
        return ave_t_err, ave_r_err

    def plot_xyz(self, seq, poses_ref, poses_pred, plot_path_dir):

        def traj_xyz(axarr, positions_xyz, style='-', color='black', title="", label="", alpha=1.0):
            x = range(0, len(positions_xyz))
            xlabel = "index"
            ylabels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
            for i in range(0, 3):
                axarr[i].plot(x, positions_xyz[:, i], style, color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('XYZ')

        fig, axarr = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))

        pred_xyz = np.array([p[:3, 3] for _, p in sorted(poses_pred.items())])
        traj_xyz(axarr, pred_xyz, '-', 'b', title='XYZ', label='Ours', alpha=1.0)
        if poses_ref:
            ref_xyz = np.array([p[:3, 3] for _, p in sorted(poses_ref.items())])
            traj_xyz(axarr, ref_xyz, '-', 'r', label='GT', alpha=1.0)

        name = "{}_xyz".format(seq)
        plt.savefig(plot_path_dir + "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + name + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)

    def plot_rpy(self, seq, poses_ref, poses_pred, plot_path_dir, axes='szxy'):

        def traj_rpy(axarr, orientations_euler, style='-', color='black', title="", label="", alpha=1.0):
            x = range(0, len(orientations_euler))
            xlabel = "index"
            ylabels = ["$roll$ (deg)", "$pitch$ (deg)", "$yaw$ (deg)"]
            for i in range(0, 3):
                axarr[i].plot(x, np.rad2deg(orientations_euler[:, i]), style,
                              color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('PRY')

        fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))

        pred_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _, p in sorted(poses_pred.items())])
        traj_rpy(axarr_rpy, pred_rpy, '-', 'b', title='RPY', label='Ours', alpha=1.0)
        if poses_ref:
            ref_rpy = np.array([tr.euler_from_matrix(p, axes=axes) for _, p in sorted(poses_ref.items())])
            traj_rpy(axarr_rpy, ref_rpy, '-', 'r', label='GT', alpha=1.0)

        name = "{}_rpy".format(seq)
        plt.savefig(plot_path_dir + "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + name + ".pdf")
        fig_rpy.tight_layout()
        pdf.savefig(fig_rpy)
        pdf.close()
        plt.close(fig_rpy)

    def plotPath_2D_3(self, seq, poses_gt, poses_result, plot_path_dir):
        fontsize_ = 10
        plot_keys = ["Ground Truth", "Ours"]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        poses_result_sorted = sorted(poses_result.items())
        x_pred = np.asarray([pose[0, 3] for _, pose in poses_result_sorted])
        y_pred = np.asarray([pose[1, 3] for _, pose in poses_result_sorted])
        z_pred = np.asarray([pose[2, 3] for _, pose in poses_result_sorted])

        fig = plt.figure(figsize=(20, 6), dpi=100)
        
        # Plot 1: X-Z
        plt.subplot(1, 3, 1)
        ax = plt.gca()
        if poses_gt:
            poses_gt_sorted = sorted(poses_gt.items())
            x_gt = np.asarray([pose[0, 3] for _, pose in poses_gt_sorted])
            z_gt = np.asarray([pose[2, 3] for _, pose in poses_gt_sorted])
            plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(x_pred[0], z_pred[0], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        ax.set_aspect('equal', adjustable='box')


        # Plot 2: X-Y
        plt.subplot(1, 3, 2)
        ax = plt.gca()
        if poses_gt:
            y_gt = np.asarray([pose[1, 3] for _, pose in poses_gt_sorted])
            plt.plot(x_gt, y_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, y_pred, style_pred, label=plot_keys[1])
        plt.plot(x_pred[0], y_pred[0], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        ax.set_aspect('equal', adjustable='box')

        # Plot 3: Y-Z
        plt.subplot(1, 3, 3)
        ax = plt.gca()
        if poses_gt:
            plt.plot(y_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(y_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(y_pred[0], z_pred[0], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xlabel('y (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        ax.set_aspect('equal', adjustable='box')

        png_title = "{}_path_2D".format(seq)
        plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + png_title + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)

    def plotPath_3D(self, seq, poses_gt, poses_result, plot_path_dir):
        from mpl_toolkits.mplot3d import Axes3D
        fontsize_ = 8
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        poses_dict = {"Ours": poses_result}
        if poses_gt:
            poses_dict["Ground Truth"] = poses_gt

        fig = plt.figure(figsize=(8, 8), dpi=110)
        # === 修改点 ===
        ax = fig.add_subplot(111, projection='3d')
        # ==============
        
        for key, poses in poses_dict.items():
            plane_point = []
            if not poses: continue
            for frame_idx in sorted(poses.keys()):
                pose = poses[frame_idx]
                plane_point.append([pose[0, 3], pose[1, 3], pose[2, 3]]) # 使用 X, Y, Z
            plane_point = np.asarray(plane_point)
            style = style_pred if key == 'Ours' else style_gt
            ax.plot(plane_point[:, 0], plane_point[:, 1], plane_point[:, 2], style, label=key)
        
        if poses_result:
            start_point = next(iter(sorted(poses_result.items())))[1]
            ax.plot([start_point[0, 3]], [start_point[1, 3]], [start_point[2, 3]], style_O, label='Start Point')

        all_x, all_y, all_z = [], [], []
        for poses in poses_dict.values():
            if not poses: continue
            for p in poses.values():
                all_x.append(p[0,3])
                all_y.append(p[1,3])
                all_z.append(p[2,3])
        
        if not all_x: return
        
        all_x, all_y, all_z = np.array(all_x), np.array(all_y), np.array(all_z)

        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        ax.set_xlabel('x (m)', fontsize=fontsize_)
        ax.set_ylabel('y (m)', fontsize=fontsize_)
        ax.set_zlabel('z (m)', fontsize=fontsize_)
        ax.view_init(elev=20., azim=-35)

        png_title = "{}_path_3D".format(seq)
        plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + png_title + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        pdf.close()
        plt.close(fig)

    def plotError_segment(self, seq, avg_segment_errs, plot_error_dir):
        fontsize_ = 15
        plot_y_t, plot_y_r, plot_x = [], [], []
        for length, value in sorted(avg_segment_errs.items()):
            if not value: continue
            plot_x.append(length)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1] / np.pi * 180 * 100)

        if not plot_x: return

        fig = plt.figure(figsize=(15, 6), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_t) * 1.1 if np.max(plot_y_t) > 0 else 1])
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)

        plt.subplot(1, 2, 2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([100, np.max(plot_x), 0, np.max(plot_y_r) * 1.1 if np.max(plot_y_r) > 0 else 1])
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize_)
        png_title = "{}_error_seg".format(seq)
        plt.savefig(plot_error_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def plotError_speed(self, seq, avg_speed_errs, plot_error_dir):
        fontsize_ = 15
        plot_y_t, plot_y_r, plot_x = [], [], []
        for speed, value in sorted(avg_speed_errs.items()):
            if not value: continue
            plot_x.append(speed * 3.6)
            plot_y_t.append(value[0] * 100)
            plot_y_r.append(value[1] / np.pi * 180 * 100)
        
        if not plot_x: return

        fig = plt.figure(figsize=(15, 6), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(plot_x, plot_y_t, 'ks-')
        plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_t) * 1.1 if np.max(plot_y_t) > 0 else 1])
        plt.xlabel('Speed (km/h)', fontsize=fontsize_)
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)

        plt.subplot(1, 2, 2)
        plt.plot(plot_x, plot_y_r, 'ks-')
        plt.axis([np.min(plot_x), np.max(plot_x), 0, np.max(plot_y_r) * 1.1 if np.max(plot_y_r) > 0 else 1])
        plt.xlabel('Speed (km/h)', fontsize=fontsize_)
        plt.ylabel('Rotation Error (deg/m)', fontsize=fontsize_)
        png_title = "{}_error_speed".format(seq)
        plt.savefig(plot_error_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def computeSegmentErr(self, seq_errs):
        segment_errs = {length: [] for length in self.lengths}
        avg_segment_errs = {}
        for err in seq_errs:
            length, r_err, t_err = err[3], err[1], err[2]
            if length in segment_errs:
                segment_errs[length].append([t_err, r_err])

        for length, errs in segment_errs.items():
            if errs:
                avg_t_err = np.mean(np.asarray(errs)[:, 0])
                avg_r_err = np.mean(np.asarray(errs)[:, 1])
                avg_segment_errs[length] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[length] = []
        return avg_segment_errs

    def computeSpeedErr(self, seq_errs):
        speed_ranges = range(2, 25, 2)
        segment_errs = {s: [] for s in speed_ranges}
        avg_segment_errs = {}

        for err in seq_errs:
            speed, r_err, t_err = err[4], err[1], err[2]
            for s_key in segment_errs:
                if abs(speed - s_key) < 2.0:
                    segment_errs[s_key].append([t_err, r_err])
                    break
        
        for speed, errs in segment_errs.items():
            if errs:
                avg_t_err = np.mean(np.asarray(errs)[:, 0])
                avg_r_err = np.mean(np.asarray(errs)[:, 1])
                avg_segment_errs[speed] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[speed] = []
        return avg_segment_errs

    def eval(self, toCameraCoord):
        eval_run_dir = os.path.join(self.output_dir, '{}_eval_epoch{}'.format(self.seq, self.epoch))
        if not os.path.exists(eval_run_dir):
            os.makedirs(eval_run_dir)
        
        poses_result = self.load_poses_from_txt(self.pred_file, toCameraCoord=toCameraCoord)
        if not poses_result:
            print("未能从 {} 加载任何位姿数据，程序退出。".format(self.pred_file))
            return

        if not self.gt_file:
            self.trajectoryDistances(poses_result)
            print("\n序列: {} (无参考轨迹)".format(self.seq))
            print('轨迹长度 (m): {:.2f}'.format(self.distance))
            self.plot_rpy(self.seq, None, poses_result, eval_run_dir)
            self.plot_xyz(self.seq, None, poses_result, eval_run_dir)
            self.plotPath_3D(self.seq, None, poses_result, eval_run_dir)
            self.plotPath_2D_3(self.seq, None, poses_result, eval_run_dir)
            print("轨迹图已保存至: {}".format(eval_run_dir))
            return

        poses_gt = self.load_poses_from_txt(self.gt_file, toCameraCoord=False)
        if not poses_gt:
            print("未能从 {} 加载任何位姿数据，程序退出。".format(self.gt_file))
            return

        seq_err = self.calcSequenceErrors(poses_gt, poses_result)
        if not seq_err:
            print("警告：无法计算任何分段误差。请检查两条轨迹是否有足够的重叠部分以及长度是否足够。")
            self.plot_rpy(self.seq, poses_gt, poses_result, eval_run_dir)
            self.plot_xyz(self.seq, poses_gt, poses_result, eval_run_dir)
            self.plotPath_3D(self.seq, poses_gt, poses_result, eval_run_dir)
            self.plotPath_2D_3(self.seq, poses_gt, poses_result, eval_run_dir)
            print("轨迹图已保存至: {}".format(eval_run_dir))
            return

        self.saveSequenceErrors(seq_err, os.path.join(eval_run_dir, '{}_error.txt'.format(self.seq)))
        
        ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
        print("\n--- 评估结果: {} ---".format(self.seq))
        print('轨迹长度 (m): {:.2f}'.format(self.distance))
        print('最大速度 (km/h): {:.2f}'.format(self.max_speed * 3.6))
        print("平均平移误差 (%):   {:.4f}".format(ave_t_err * 100))
        print("平均旋转误差 (deg/100m): {:.4f}".format(ave_r_err / np.pi * 180 * 100))
        print("-" * 30)

        summary_txt_path = os.path.join(self.output_dir, 'evaluation_summary.txt')
        with open(summary_txt_path, 'a+') as f:
            f.write('--- 序列: {}, Epoch: {} ---\n'.format(self.seq, self.epoch))
            f.write('轨迹文件: {}\n'.format(os.path.basename(self.pred_file)))
            f.write('参考文件: {}\n'.format(os.path.basename(self.gt_file)))
            f.write('平均平移误差 (%): {:.4f}\n'.format(ave_t_err * 100))
            f.write('平均旋转误差 (deg/100m): {:.4f}\n\n'.format(ave_r_err / np.pi * 180 * 100))

        avg_segment_errs = self.computeSegmentErr(seq_err)
        avg_speed_errs = self.computeSpeedErr(seq_err)

        self.plot_rpy(self.seq, poses_gt, poses_result, eval_run_dir)
        self.plot_xyz(self.seq, poses_gt, poses_result, eval_run_dir)
        self.plotPath_3D(self.seq, poses_gt, poses_result, eval_run_dir)
        self.plotPath_2D_3(self.seq, poses_gt, poses_result, eval_run_dir)
        self.plotError_segment(self.seq, avg_segment_errs, eval_run_dir)
        self.plotError_speed(self.seq, avg_speed_errs, eval_run_dir)
        print("详细评估结果和图表已保存至: {}".format(eval_run_dir))

        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI Odometry Evaluation Tool')
    
    parser.add_argument('--gt_file', type=str, required=False, 
                        help='[可选] KITTI格式的参考轨迹文件路径 (例如: ground_truth/09.txt)')
    parser.add_argument('--pred_file', type=str, required=True, 
                        help='[必需] KITTI格式的评价轨迹文件路径 (例如: results/09_pred.txt)')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='[必需] 保存所有评估结果 (图表, 报告) 的目录')
    
    parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='是否将评价轨迹的位姿转换到相机坐标系 (默认为False)')
    parser.add_argument('--epoch', type=int, default=0, help='当前评估的epoch值, 用于命名输出文件夹')

    args = parser.parse_args()
    
    pose_eval = kittiOdomEval(args)
    pose_eval.eval(toCameraCoord=args.toCameraCoord)