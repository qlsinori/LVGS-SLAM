# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data parser for PandaSet dataset"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type
import numpy.typing as npt
import numpy as np
import pandas as pd
import pyquaternion
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType,transform_points
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.lidar_elevation_mappings import MCD_ELEVATION_MAPPING
from nerfstudio.utils import poses as pose_utils


# PANDASET_ELEVATION_MAPPING = {"Pandar64": PANDAR64_ELEVATION_MAPPING}
MCD_ELEVATION_MAPPING = {"os64":MCD_ELEVATION_MAPPING}

# LIDAR_NAME_TO_INDEX = {
#     "Pandar64": 0,
#     "PandarGT": 1,
# }
LIDAR_NAME_TO_INDEX = {
    "VLP16": 0,
}

DATA_FREQUENCY = 10.0  # 10 Hz
LIDAR_ROTATION_TIME = 1.0 / DATA_FREQUENCY  # 10 Hz

EXTRINSICS_FILE_PATH = os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml")
MAX_RELECTANCE_VALUE = 255.0
MAX_INTENSITY_VALUE = 255.0
BACK_CAMERA_BOTTOM_CROP = 0
ALLOWED_RIGID_CLASSES = (
    "Car",
    "Pickup Truck",
    "Medium-sized Truck",
    "Semi-truck",
    "Towed Object",
    "Motorcycle",
    "Other Vehicle - Construction Vehicle",
    "Other Vehicle - Uncommon",
    "Other Vehicle - Pedicab",
    "Emergency Vehicle",
    "Bus",
    "Personal Mobility Device",
    "Motorized Scooter",
    "Bicycle",
    "Train",
    "Trolley",
    "Tram / Subway",
)
ALLOWED_DEFORMABLE_CLASSES = (
    "Pedestrian",
    "Pedestrian with Object",
)

LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
        "001": -1,
        "011": 1,
        "016": 1,
        "028": -1,
        "053": 1,
        "063": -1,
        "084": -1,
        "106": -1,
        "123": -1,
        "158": -1,
    }
)

MCD_AZIMUTH_RESOLUTION = {"os64": 360.0/1024}
# BOTANIC_IGNORE_REGIONS = {"vlp16": [[-180.0, -157.5, -15.0,15.0],[157.5,180.0,-15.0,15.0]]}
MCD_IGNORE_REGIONS = {"os64": []}

MCD_SKIP_ELEVATION_CHANNELS = {
    "os64": ()
}

HORIZONTAL_BEAM_DIVERGENCE = 3e-3  # radians 和原来的一样
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians

AVAILABLE_CAMERAS = ("rgbt", "rgbb")


def _get_timestamp_from_path(file_path: Path) -> float:
    """Parse timestamp from MCD camera filename like '1666259388_123896122.tif'."""
    name = file_path.stem
    seconds, nanoseconds_str = name.split('_')
    nanoseconds_padded_str = nanoseconds_str.ljust(9, '0')
    return int(seconds) + int(nanoseconds_padded_str) / 1e9


@dataclass
class MCDDataParserConfig(ADDataParserConfig):
    """PandaSet dataset config.
    PandaSet (https://pandaset.org/) is an autonomous driving dataset containing 100+ 8s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras and two lidars.
    It also includes 3D cuboid annotations around objects.
    """

    _target: Type = field(default_factory=lambda: MCD)
    """target class to instantiate"""
    data: Path = Path("/qls/code/dataset/MCD")
    """Directory specifying location of data."""
    sequence: str = "tuhh_04"
    """Name of the scene."""
    cameras: Tuple[Literal["rgbt", "rgbb", "gray0", "gray1", "all"], ...] = (
        "rgbt",
        # "rgbb",
    )
    """Which cameras to use."""
    lidars: Tuple[Literal["os64", "livox", "none"], ...] = ("os64",)
    """Which lidars to use."""
    annotation_interval: float = 0.1
    """Interval between annotations in seconds."""
    correct_cuboid_time: bool = False
    """Whether to correct the cuboid time to match the actual time of observation, not the end of the lidar sweep."""

    """Pandaset lidar is x-right, y-down, z-forward."""
    lidar_elevation_mapping: Dict[str, Dict] = field(default_factory=lambda: MCD_ELEVATION_MAPPING)
    """Elevation mapping for each lidar."""
    skip_elevation_channels: Dict[str, Tuple] = field(default_factory=lambda: MCD_SKIP_ELEVATION_CHANNELS)
    """Channels to skip when adding missing points."""
    lidar_azimuth_resolution: Dict[str, float] = field(default_factory=lambda: MCD_AZIMUTH_RESOLUTION)
    """Azimuth resolution for each lidar."""
    # rolling_shutter_time: float = 0.00
    """The rolling shutter time for the cameras (seconds)."""
    # time_to_center_pixel: float = 0.0
    """In pandaset the image time seems to line up with the final row."""
    index2folder = ['d455t', 'd455b']
    """7a left 0 79 right 1"""
    min_lidar_dist: Tuple[float, float, float,float, float, float] = (-5.0, 2.0, -1.5, 1.5 ,-2.0, 2.0)
    camera_frequency: int = 5
    downsample_camera: bool = False
    use_icp_pose: bool = False
    """Use lidar ICP poses; interpolate to get camera poses."""
    use_camera_pose: bool = False
    """Use external camera poses (camera_poses.csv); interpolate to get lidar poses."""
    camera_gt: bool = True
@dataclass
class MCD(ADDataParser):
    """Botanic DatasetParser"""

    config: MCDDataParserConfig

    def _get_lane_shift_sign(self, sequence: str) -> Literal[-1, 1]:
        return LANE_SHIFT_SIGN.get(sequence, 1)

    def _collect_camera_timestamps(self) -> List[float]:
        """Pre-collect cam0 timestamps from filenames (with frequency filtering).
        Used to pair with KITTI-format camera poses."""
        camera_folder = Path(str(self.config.data)) / self.config.sequence / self.config.index2folder[0]
        all_camera_files = list(camera_folder.glob("*.tif"))
        files_with_timestamps = sorted(
            [(_get_timestamp_from_path(f), f) for f in all_camera_files])

        timestamps = []
        for index, (timestamp, _) in enumerate(files_with_timestamps):
            if self.config.camera_frequency == 5:
                if (index % 5) != 0:
                    continue
            elif self.config.camera_frequency == 10:
                if (index % 3) != 0:
                    continue
            timestamps.append(timestamp)
        return timestamps

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""
        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS

        filenames, times, poses, camera_frame_pose, idxs = [], [], [], [], []
        fx, fy, cx, cy = [], [], [], []
        distortion_param = []
        for camera_idx, cam_name in enumerate(self.config.cameras):
            camera_folder = Path(str(self.config.data)) / self.config.sequence / self.config.index2folder[camera_idx]
            all_camera_files = list(camera_folder.glob("*.tif"))
            files_with_timestamps = sorted(
                [(_get_timestamp_from_path(f), f) for f in all_camera_files])

            for index, (timestamp, camera_file) in enumerate(files_with_timestamps):

                if self.config.camera_frequency == 5:
                    if (index % 5) != 0:
                        continue
                elif self.config.camera_frequency == 10:
                    if (index % 3) != 0:
                        continue
                else:
                    print("ERROR ",self.config.camera_frequency)

                t_val = float(timestamp)
                if self.config.use_icp_pose:
                    if not (self.icp_times[0] <= t_val <= self.icp_times[-1]):
                        continue
                elif self.config.use_camera_pose:
                    if not (self.camera_traj_times[0] <= t_val <= self.camera_traj_times[-1]):
                        continue
                else:
                    if not (self.times_traj[0] < t_val < self.times_traj[-1]):
                        continue

                filenames.append(camera_file)
                times.append(timestamp)
                idxs.append(camera_idx)

                if self.config.use_icp_pose:
                    lidar_pose = interpolate_single_pose(
                        timestamp, self.icp_times, self.icp_traj_arr)
                    cam_pose = lidar_pose @ np.linalg.inv(
                        self.calibs['T_rgb' + str(camera_idx) + '_vlp16'])
                elif self.config.use_camera_pose:
                    cam0_pose = interpolate_single_pose(
                        timestamp, self.camera_traj_times, self.camera_traj)
                    if camera_idx == 0:
                        cam_pose = cam0_pose
                    else:
                        cam_pose = cam0_pose @ np.linalg.inv(
                            self.calibs['T_rgb0']) @ self.calibs['T_rgb' + str(camera_idx)]
                else:
                    ego_pose = interpolate_single_pose(
                        timestamp, self.times_traj, self.traj)
                    cam_pose = ego_pose @ self.calibs['T_rgb' + str(camera_idx)]

                cam_pose[:3, :3] = cam_pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                poses.append(cam_pose)
                # if self.config.downsample_camera :
                #     fx.append(self.calibs['rgb' + str(camera_idx)+'_down']['projection_parameters']['fx'])
                #     fy.append(self.calibs['rgb' + str(camera_idx)+'_down']['projection_parameters']['fy'])
                #     cx.append(self.calibs['rgb' + str(camera_idx)+'_down']['projection_parameters']['cx'])
                #     cy.append(self.calibs['rgb' + str(camera_idx)+'_down']['projection_parameters']['cy'])
                # else:

                intr = np.array(self.calibs['rgb' + str(camera_idx)]['intrinsics'], dtype=np.float32).reshape(4)
                fx.append(intr[0])
                fy.append(intr[1])
                cx.append(intr[2])
                cy.append(intr[3])
                distortion_coeffs = np.array(self.calibs['rgb' + str(camera_idx)]['distortion_coeffs'], dtype=np.float32).reshape(4)
                distortion_param.append(np.array([distortion_coeffs[0],
                                    distortion_coeffs[1],
                                     0.0,0.0,distortion_coeffs[2],
                                    distortion_coeffs[3]]))

        # To tensors
        fx = torch.from_numpy(np.array(fx, dtype=np.float32))
        fy = torch.from_numpy(np.array(fy, dtype=np.float32))
        cx = torch.from_numpy(np.array(cx, dtype=np.float32))
        cy = torch.from_numpy(np.array(cy, dtype=np.float32))
        poses = torch.from_numpy(np.array(poses, dtype=np.float32))
        times = torch.from_numpy(np.array(times, dtype=np.float64))  # need higher precision
        idxs = torch.from_numpy(np.array(idxs, dtype=np.int32)).unsqueeze(-1)
        distortion_param = torch.from_numpy(np.array(distortion_param, dtype=np.float32))

        # if self.config.downsample_camera:
        #     height= self.calibs['rgb' + str(camera_idx)+'_down']['image_height']
        #     width = self.calibs['rgb' + str(camera_idx)+'_down']['image_width']
        # else:
        height = 480
        width =  640

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            distortion_params = distortion_param,
            metadata={"sensor_idxs": idxs,"rcam2lcam": self.calibs["T_rgb0_rgb1"],"lidar2lcam":self.calibs["T_rgb0_vlp16"],
                    "lidar2rcam":self.calibs["T_rgb1_vlp16"]
                # ,"gt_pose":poses[:, :3, :4]
                      },
        )
        return cameras, filenames


    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""
        poses = []
        times = []
        idxs = []
        lidar_filenames = []
        for lidar_index, lidar_name in enumerate(self.config.lidars):

            lidar_folder = Path(str(self.config.data)) / self.config.sequence / "bin_xyzitr"
            lidar_files = sorted(lidar_folder.glob("*.bin"))

            for index, lidar_file in enumerate(lidar_files):
                t = float(lidar_file.stem)
                if self.config.use_icp_pose:
                    pass
                elif self.config.use_camera_pose:
                    if not (self.camera_traj_times[0] <= t <= self.camera_traj_times[-1]):
                        continue
                else:
                    if not (self.times_traj[0] < t < self.times_traj[-1]):
                        continue

                lidar_filenames.append(lidar_file)
                idxs.append(lidar_index)
                times.append(t)

        if self.config.use_icp_pose:
            poses = np.array(self.icp_traj, dtype=np.float64)
            self.icp_times = np.array(times, dtype=np.float64)
            self.icp_traj_arr = poses.copy()
        elif self.config.use_camera_pose:
            cam_to_lidar = self.calibs["T_rgb0_vlp16"]  # lidar→cam0 extrinsic
            for t in times:
                cam_pose = interpolate_single_pose(
                    t, self.camera_traj_times, self.camera_traj)
                lidar_pose = cam_pose @ cam_to_lidar
                poses.append(lidar_pose)
            poses = np.array(poses)
        else:
            for t in times:
                p = interpolate_single_pose(t, self.times_traj, self.traj)
                p = p @ self.calibs["T_vlp16"]
                poses.append(p)
            poses = np.array(poses)

        poses = torch.from_numpy(np.array(poses, dtype=np.float32))
        times = torch.from_numpy(np.array(times, dtype=np.float64))
        idxs = torch.from_numpy(np.array(idxs, dtype=np.int32)).unsqueeze(-1)



        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.OS64,
            times=times,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []
        point_clouds_in_world = []

        for filepath in filepaths:
            #x, y, z, intensity, t, reflec   bin_xyzitr

            try:
                pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 6)
            except:
                print("error:",filepath)
                ValueError
            # print(pc)
            xyz = pc[:, :3]  # N x 3
            pc[:, 5] = pc[:, 5] / MAX_RELECTANCE_VALUE  # N,
            pc[:, 4] = pc[:, 4] * 1.0 / 1e9
             # N, relative timestamps
            # pc = np.hstack((xyz, intensity[:, None], t[:, None]))
            pc = pc[..., [0, 1, 2, 5, 4, 3]]
            point_clouds.append(torch.from_numpy(pc).float())

        times = lidars.times
        poses = lidars.lidar_to_worlds

        if self.config.add_missing_points and not (self.config.use_icp_pose or self.config.use_camera_pose):

            missing_points = []
            ego_times = torch.from_numpy(np.array(self.times_traj))
            times_close_to_lidar = (ego_times > (times.min() - 0.1)) & (ego_times < (times.max() + 0.1))
            ego_times = ego_times[times_close_to_lidar]
            ego_poses = torch.from_numpy(np.array(self.traj,dtype=np.float64))[times_close_to_lidar]
            lidar2ego = torch.from_numpy(np.array(self.calibs["T_vlp16"],dtype=np.float64))
            oxts_lidar_poses = ego_poses @ lidar2ego.unsqueeze(0)

            for point_cloud, l2w, time in zip(point_clouds, poses, times):
                pc = point_cloud.clone().to(torch.float64)
                # absolute time
                pc[:, 4] = pc[:, 4] + time
                # project to world frame
                pc[..., :3] = transform_points(pc[..., :3], l2w.unsqueeze(0).to(pc))
                # remove ego motion compensation and move to sensor frame
                pc, interpolated_poses = self._remove_ego_motion_compensation(pc, oxts_lidar_poses, ego_times)
                # reset time
                pc[:, 4] = point_cloud[:, 4].clone()
                # transform to common lidar frame again
                interpolated_poses = torch.matmul(
                    pose_utils.inverse(l2w.unsqueeze(0)).float(), pose_utils.to4x4(interpolated_poses).float()
                )
                # move channel from index 5 to 3
                pc = pc[..., [0, 1, 2, 5, 3, 4]]
                # add missing points
                mp = self._get_missing_points(
                    pc, interpolated_poses, "os64", dist_cutoff=0.0, ignore_regions=MCD_IGNORE_REGIONS["os64"]
                ).float()
                # move channel from index 3 to 5
                mp = mp[..., [0, 1, 2, 4, 5, 3]]
                missing_points.append(mp)
            # add missing points to point clouds
            point_clouds = [torch.cat([pc, missing], dim=0) for pc, missing in zip(point_clouds, missing_points)]

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds


    def _get_actor_trajectories(self) -> List[Dict]:
        """Returns a list of actor trajectories."""
        allowed_classes = ALLOWED_RIGID_CLASSES
        if self.config.include_deformable_actors:
            allowed_classes += ALLOWED_DEFORMABLE_CLASSES
        cuboids = []
        for i in range(PANDASET_SEQ_LEN):
            curr_cuboids = self.sequence.cuboids[i]
            # Remove invalid cuboids
            is_allowed_class = np.array([label in allowed_classes for label in curr_cuboids["label"]])
            valid_mask = (~curr_cuboids["stationary"]) & is_allowed_class
            curr_cuboids = curr_cuboids[valid_mask]
            if not len(curr_cuboids):
                continue

            uuid = np.array(curr_cuboids["uuid"])
            label = np.array(curr_cuboids["label"])

            yaw = curr_cuboids["yaw"].astype(np.float32)
            rot = _yaw_to_rotation_matrix(yaw)

            stationary = np.array(curr_cuboids["stationary"], dtype=np.bool8)  # True for static objects
            pos_x = curr_cuboids["position.x"].astype(np.float32)  # x position of cuboid in world coords
            pos_y = curr_cuboids["position.y"].astype(np.float32)  # y position of cuboid in world coords
            pos_z = curr_cuboids["position.z"].astype(np.float32)  # z position of cuboid in world coords
            pos = np.vstack([pos_x, pos_y, pos_z]).T

            cuboid_poses = np.eye(4)[None].repeat(len(uuid), axis=0)
            cuboid_poses[:, :3, :3] = rot
            cuboid_poses[:, :3, 3] = pos

            width = curr_cuboids["dimensions.x"].astype(np.float32)  # width of cuboid in world coords
            length = curr_cuboids["dimensions.y"].astype(np.float32)  # length of cuboid in world coords
            height = curr_cuboids["dimensions.z"].astype(np.float32)  # height of cuboid in world coords
            dims = np.vstack([width, length, height]).T

            # if dynamic and visible in both 360 and front facing, two cuboids are annoated. 0 for 360, 1 for front facing, -1 otherwise
            sensor_id = np.array(curr_cuboids["cuboids.sensor_id"], dtype=np.int32)
            sibling_id = np.array(
                curr_cuboids["cuboids.sibling_id"]
            )  # uuid of sibling cuboid, i.e., if sensor_id != -1

            if self.config.correct_cuboid_time:
                # correct the cuboid time to match the actual time of observation, not the end of the lidar sweep
                lidpose = _pandaset_pose_to_matrix(self.sequence.lidar.poses[i])
                posinlid = pos @ lidpose[:3, :3].T + lidpose[:3, 3]
                angle = np.arctan2(posinlid[:, 0], posinlid[:, 1]) - np.pi / 2
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                timediff = angle / (2 * np.pi) * np.diff(self.sequence.lidar.timestamps).mean()
                cuboid_times = self.sequence.camera["front_camera"].timestamps[i] + timediff
            else:
                # assume the cuboid time matches the sequence time
                cuboid_times = np.repeat(self.sequence.camera["front_camera"].timestamps[i], len(uuid))

            for cuboid_index in range(len(uuid)):
                cuboids.append(
                    {
                        "uuid": uuid[cuboid_index],
                        "label": label[cuboid_index],
                        "poses": cuboid_poses[cuboid_index],
                        "stationary": stationary[cuboid_index],
                        "dims": dims[cuboid_index],
                        "sensor_ids": sensor_id[cuboid_index],
                        "sibling_id": sibling_id[cuboid_index] if sensor_id[cuboid_index] != -1 else None,
                        "timestamps": np.array(cuboid_times[cuboid_index]),
                    }
                )
        return _cuboids_to_trajectories(cuboids)

    def _generate_dataparser_outputs(self, split="train") -> DataparserOutputs:
        assert not (self.config.use_icp_pose and self.config.use_camera_pose), \
            "use_icp_pose and use_camera_pose are mutually exclusive"

        seq_folder = Path(str(self.config.data)) / self.config.sequence

        if self.config.use_icp_pose:
            # icp_pose_path = str(seq_folder / "bin_xyzitr_poses_kitti.txt")
            icp_pose_path = str(seq_folder / "ego_lidar_poses_kitti.txt")
            self.icp_traj = read_kitti_trajectory(icp_pose_path)
        elif self.config.use_camera_pose:
            camera_pose_path = str(seq_folder / "camera_poses_kitti.txt")
            camera_poses_list = read_kitti_trajectory(camera_pose_path)
            camera_timestamps = self._collect_camera_timestamps()
            assert len(camera_poses_list) == len(camera_timestamps), \
                f"camera_poses_kitti.txt has {len(camera_poses_list)} poses but found {len(camera_timestamps)} camera frames"
            self.camera_traj_times = np.array(camera_timestamps, dtype=np.float64)
            self.camera_traj = np.array(camera_poses_list, dtype=np.float64)
        else:
            ego_pose_path = str(seq_folder / "pose_inW.csv")
            self.times_traj, self.traj = load_trajectory_to_poses(ego_pose_path)

        self.calibs = get_calib(str(self.config.data))

        out = super()._generate_dataparser_outputs(split=split)
        for attr in ('times_traj', 'traj', 'calibs', 'icp_traj',
                      'icp_times', 'icp_traj_arr', 'camera_traj_times', 'camera_traj'):
            self.__dict__.pop(attr, None)
        return out


    def _add_channel_info(self, point_cloud: torch.Tensor, dim: int = -1, lidar_name: str = "") -> torch.Tensor:
        """Infer channel id from point cloud, and add it to the point cloud.

        Args:
            point_cloud: Point cloud to add channel id to (in sensor frame). Shape: [num_points, 3+x] x,y,z (timestamp, intensity, etc.)

        Returns:
            Point cloud with channel id. Shape: [num_points, 3+x+1] x,y,z (timestamp, intensity, etc.), channel_id
            channel_id is added to dim
        """
        # these are limits where channels are equally spaced
        ELEV_HIGH_IDX = 5
        ELEV_LOW_IDX = -11
        ELEV_LOW_IDX_ABS = len(self.config.lidar_elevation_mapping[lidar_name]) + ELEV_LOW_IDX

        dist = torch.norm(point_cloud[:, :3], dim=-1)
        elevation = torch.arcsin(point_cloud[:, 2] / dist)
        elevation = torch.rad2deg(elevation)

        middle_elev_mask = (elevation < (self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX] + 0.2)) & (
            elevation > (self.config.lidar_elevation_mapping[lidar_name][ELEV_LOW_IDX_ABS] - 0.2)
        )
        middle_elev = elevation[middle_elev_mask]

        histc, bin_edges = torch.histogram(middle_elev, bins=2000)

        # channels should be equally spaced
        expected_channel_edges = (bin_edges[-1] - bin_edges[0]) / 49 * torch.arange(50) + bin_edges[0]

        res = (
            self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX]
            - self.config.lidar_elevation_mapping[lidar_name][ELEV_HIGH_IDX + 1]
        )

        # find consecutive empty bins in histogram
        empty_bins = []
        empty_bin = []
        empty_bins_edges = []
        for i in range(len(histc)):
            if histc[i] == 0:
                empty_bin.append(i)
            else:
                if len(empty_bin) > 0:
                    empty_bins.append(empty_bin)
                    empty_bins_edges.append((bin_edges[empty_bin[0]], bin_edges[empty_bin[-1] + 1]))
                    empty_bin = []

        # find channel edges, use first expected for init
        found_channel_edges = [expected_channel_edges[0].tolist()]
        empty_bins_edges = torch.tensor(empty_bins_edges)
        for i, edge in enumerate(expected_channel_edges[1:-1]):
            found_edge = False
            for empty_bin in empty_bins_edges:
                # if edge is in empty bin, keep the edge as is
                if edge > empty_bin[0] and edge < empty_bin[1]:
                    found_channel_edges.append(edge.tolist())
                    found_edge = True
                    break
            if found_edge:
                continue
            distances = torch.abs(edge - empty_bins_edges)
            min_dist_idx = distances.argmin()
            if distances.flatten()[min_dist_idx] < 0.03:
                found_channel_edges.append(empty_bins_edges.flatten()[min_dist_idx].tolist())
                continue

        found_channel_edges.append(expected_channel_edges[-1].tolist())
        found_channel_edges = torch.tensor(found_channel_edges)

        if len(found_channel_edges) < len(expected_channel_edges):
            # we have missing channels, interpolate edges
            while (num_missing_edges := len(expected_channel_edges) - len(found_channel_edges)) > 0:
                distances = found_channel_edges.diff().abs()
                max_dist_idx = distances.argmax()
                num_edges_to_insert = max((distances[max_dist_idx] / res).round().int() - 1, 1)
                num_edges_to_insert = min(num_missing_edges, num_edges_to_insert)
                new_edges = torch.linspace(
                    found_channel_edges[max_dist_idx], found_channel_edges[max_dist_idx + 1], num_edges_to_insert + 2
                )[1:-1]
                found_channel_edges = torch.cat(
                    [found_channel_edges[: max_dist_idx + 1], new_edges, found_channel_edges[max_dist_idx + 1 :]]
                )  # insert new edges

        # add remaining edges
        for i in range(len(self.config.lidar_elevation_mapping[lidar_name])):
            if i >= ELEV_HIGH_IDX and i <= len(self.config.lidar_elevation_mapping[lidar_name]) + ELEV_LOW_IDX:
                continue
            current_elevation = self.config.lidar_elevation_mapping[lidar_name][i]
            if i == 0:
                new_edge = 100
            elif i == len(self.config.lidar_elevation_mapping[lidar_name]) - 1:
                new_edge = -100
            elif i < ELEV_HIGH_IDX:
                dist_to_prev_elevation = abs(current_elevation - self.config.lidar_elevation_mapping[lidar_name][i - 1])
                new_edge = current_elevation + dist_to_prev_elevation * 0.22
            elif i > len(self.config.lidar_elevation_mapping[lidar_name]) + ELEV_LOW_IDX:
                dist_to_next_elevation = (
                    abs(current_elevation - self.config.lidar_elevation_mapping[lidar_name][i + 1])
                    if i < len(self.config.lidar_elevation_mapping[lidar_name]) - 1
                    else 1000.0
                )
                new_edge = current_elevation - dist_to_next_elevation * 0.22

            found_channel_edges = torch.cat([found_channel_edges, torch.tensor([new_edge]).float()])

        found_channel_edges, _ = torch.sort(found_channel_edges, descending=True)
        channel_id = torch.full((point_cloud.shape[0], 1), -1, device=point_cloud.device)

        # assign channel id
        for i in range(len(self.config.lidar_elevation_mapping[lidar_name])):
            elevation_mask = (elevation >= found_channel_edges[i + 1]) & (elevation < found_channel_edges[i])
            channel_id[elevation_mask] = i

        point_cloud = torch.cat([point_cloud[:, :dim], channel_id, point_cloud[:, dim:]], dim=-1)
        return point_cloud


def _pandaset_pose_to_matrix(pose):
    translation = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]])
    quaternion = np.array([pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]])
    pose = np.eye(4)
    pose[:3, :3] = pyquaternion.Quaternion(quaternion).rotation_matrix
    pose[:3, 3] = translation
    return pose


def _yaw_to_rotation_matrix(yaw: np.ndarray):
    """Converts array of yaw angles to rotation matrices."""
    rotation_matrices = np.zeros((yaw.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = np.cos(yaw)
    rotation_matrices[:, 0, 1] = -np.sin(yaw)
    rotation_matrices[:, 1, 0] = np.sin(yaw)
    rotation_matrices[:, 1, 1] = np.cos(yaw)
    rotation_matrices[:, 2, 2] = 1
    return rotation_matrices


def _cuboids_to_trajectories(cuboids):
    """Connects cuboids into trajectories."""
    trajs = []
    trajs_dict = {}
    for cuboid in cuboids:
        if cuboid["sensor_ids"] == 1:  # TODO: allow for cuboids from front-facing lidar
            continue  # skip cuboids from front-facing lidar

        if cuboid["uuid"] not in trajs_dict:
            trajs_dict[cuboid["uuid"]] = []

        trajs_dict[cuboid["uuid"]] += [cuboid]

    for uuid, traj in trajs_dict.items():
        trajs_dict[uuid] = sorted(traj, key=lambda x: x["timestamps"])
        trajs.append(
            {
                # "uuid": uuid,
                "poses": torch.from_numpy(np.stack([t["poses"] for t in traj])).float(),
                "timestamps": torch.from_numpy(np.stack([t["timestamps"] for t in traj])),
                "dims": torch.from_numpy(np.array([t["dims"] for t in traj]).astype(np.float32).max(axis=0)),
                "label": traj[0]["label"],
                "stationary": traj[0]["stationary"],
                "symmetric": "Pedestrian" not in traj[0]["label"],
                "deformable": "Pedestrian" in traj[0]["label"],
            }
        )
    return trajs

def get_calib(data_root: str = "/qls/code/dataset/MCD") -> Dict[str, npt.NDArray[np.float32]]:
    extrinsics = os.path.join(data_root, "hhs_calib.yaml")
    with open(extrinsics) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)






    T_rgb0 = np.array(data['body']['d455t_color']['T'], dtype=np.float32).reshape(4, 4)
    T_rgb1 = np.array(data['body']['d455b_color']['T'], dtype=np.float32).reshape(4, 4)
    T_vlp16 = np.array(data['body']['os_sensor']['T'], dtype=np.float32).reshape(4, 4)

    T_rgb0_vlp16 = np.linalg.inv(T_rgb0) @ T_vlp16
    T_rgb0_rgb1 = np.linalg.inv(T_rgb0) @ T_rgb1
    T_rgb1_vlp16 = np.linalg.inv(T_rgb1) @ T_vlp16


    rgb0 = data['body']['d455t_color']
    rgb1 = data['body']['d455b_color']



    return {
        "rgb0": rgb0,  # type: ignore
        "rgb1": rgb1,  # type: ignore
        "T_rgb0":T_rgb0,
        "T_rgb1": T_rgb1,
        "T_vlp16": T_vlp16,
        "T_rgb0_rgb1":T_rgb0_rgb1,
        "T_rgb0_vlp16":T_rgb0_vlp16,
        "T_rgb1_vlp16":T_rgb1_vlp16,
    }


def load_trajectory_to_poses(csv_filepath_or_buffer):
    """
    从一个逗号分隔的CSV文件中读取轨迹数据，并将其转换为归一化的4x4位姿矩阵。

    Args:
        csv_filepath_or_buffer (str or buffer): CSV文件的路径或一个文件缓冲区。
                                               文件应有标题行，且使用逗号作为分隔符。

    Returns:
        tuple: 一个包含两个列表的元组:
               - timestamps (list): 与每个位姿对应的时间戳列表。
               - poses (list): 一个由归一化后的4x4 NumPy数组组成的列表。
    """
    try:
        # 【核心修正】: 将分隔符从空格 sep=r'\s+' 改为逗号 sep=','
        # 同时，因为文件现在有正确的标题行，我们使用 header=0 让pandas自动识别。
        df = pd.read_csv(
            csv_filepath_or_buffer,
            sep=',',
            header=0
        )
    except FileNotFoundError:
        print(f"错误: 文件未找到 at path: {csv_filepath_or_buffer}")
        return [], []
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
        return [], []

    absolute_poses = []
    timestamps = []

    # --- 步骤 1: 读取所有位姿为绝对位姿 ---
    for index, row in df.iterrows():
        try:
            # 确保列名与CSV标题行中的完全一致
            timestamps.append(row['t'])
            translation = row[['x', 'y', 'z']].values.astype(float)
            quaternion = row[['qx', 'qy', 'qz', 'qw']].values.astype(float)

            rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = translation

            absolute_poses.append(pose_matrix)
        except (KeyError, ValueError) as e:
            # 如果标题行有任何不可见的字符或空格，这里仍然可能出错
            # 例如 ' t' 而不是 't'。可以打印 df.columns 来调试。
            # print("DataFrame columns:", df.columns)
            print(f"警告: 处理第 {index} 行时出错，已跳过。错误: {e}")
            continue

    # --- 步骤 2: 对所有位姿进行归一化 ---
    if not absolute_poses:
        return [], []

    inv_first_pose = np.linalg.inv(absolute_poses[0])
    normalized_poses = [inv_first_pose @ p for p in absolute_poses]


    timestamps = np.array(timestamps, dtype=np.float64)
    normalized_poses  = np.array(normalized_poses, dtype=np.float32)

    return timestamps, normalized_poses

def read_kitti_trajectory(file_path):
    """
    读取KITTI格式的轨迹文件
    格式: 每行 = [ r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz]
    实际构成4×4变换矩阵:
        [[r11, r12, r13, tx],
         [r21, r22, r23, ty],
         [r31, r32, r33, tz],
         [0,   0,   0,   1]]
    """
    trajectories = []


    if not os.path.exists(file_path):
        raise FileNotFoundError(f"轨迹文件不存在: {file_path}")

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # 跳过空行和注释行
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 分割行数据
            parts = line.split()
            if len(parts) != 12:
                print(f"警告: 第 {line_num} 行包含 {len(parts)} 个元素 (应为12个), 跳过")
                continue

            try:


                # 解析变换矩阵元素
                elements = list(map(float, parts[:]))

                # 构建4×4变换矩阵
                transform = np.array([
                    [elements[0], elements[1], elements[2], elements[3]],  # 第一行: r11, r12, r13, tx
                    [elements[4], elements[5], elements[6], elements[7]],  # 第二行: r21, r22, r23, ty
                    [elements[8], elements[9], elements[10], elements[11]],  # 第三行: r31, r32, r33, tz
                    [0.0, 0.0, 0.0, 1.0]  # 第四行
                ])

                # 存储结果

                trajectories.append(transform)

            except ValueError as e:
                print(f"错误: 第 {line_num} 行解析失败 - {str(e)}")
                print(f"问题行内容: {line}")

    return trajectories
def read_times(file_path):
    timestamps = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"轨迹文件不存在: {file_path}")

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # 跳过空行和注释行
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 分割行数据
            parts = line.split()
            if len(parts) != 1:
                print(f"警告: 第 {line_num} 行包含 {len(parts)} 个元素 (应为1个), 跳过")
                continue

            try:
                # 解析时间戳
                timestamp = float(parts[0])

                timestamps.append(timestamp)


            except ValueError as e:
                print(f"错误: 第 {line_num} 行解析失败 - {str(e)}")
                print(f"问题行内容: {line}")

    return timestamps

def interpolate_poses(rgb_times, traj_times, traj_poses):
    """
    根据RGB时间戳和轨迹数据（时间和位姿），通过插值获得RGB图像对应的位姿

    参数:
        rgb_times: RGB图像的时间戳列表 (M,)
        traj_times: 轨迹的时间戳列表 (N,)
        traj_poses: 轨迹的位姿列表 (N, 4, 4)

    返回:
        interp_poses: 插值后的位姿 (M, 4, 4)
    """
    # 将输入转换为numpy数组
    traj_times = np.array(traj_times)
    traj_poses = np.array(traj_poses)
    rgb_times = np.array(rgb_times)

    # 确保时间序列是单调递增的
    sort_idx = np.argsort(traj_times)
    traj_times = traj_times[sort_idx]
    traj_poses = traj_poses[sort_idx]

    # 提取旋转和平移分量
    rotations = traj_poses[:, :3, :3]  # (N, 3, 3)
    translations = traj_poses[:, :3, 3]  # (N, 3)

    # 创建旋转插值器（使用球面线性插值）
    def check_rotation_validity(R):
        """检查旋转矩阵是否有效（正交且行列式≈1）"""
        # 检测奇异矩阵
        if np.linalg.det(R) < 1e-6:
            return False

        # 检测正交性：R@R.T 应接近单位矩阵
        ortho_check = R @ R.T
        if not np.allclose(ortho_check, np.eye(3), atol=1e-4):
            return False

        return True

    # 遍历检查所有旋转矩阵
    invalid_indices = []
    for i, R in enumerate(rotations):
        if not check_rotation_validity(R):
            invalid_indices.append(i)

    if invalid_indices:
        raise ValueError(
            f"found {len(invalid_indices)} useless rot (index at: {invalid_indices})。"
        )

    rot_objects = Rotation.from_matrix(rotations)
    slerp = Slerp(traj_times, rot_objects)

    # 创建平移插值器（使用线性插值）
    trans_interp_x = interp1d(traj_times, translations[:, 0], kind='linear',
                              fill_value="extrapolate", assume_sorted=True)
    trans_interp_y = interp1d(traj_times, translations[:, 1], kind='linear',
                              fill_value="extrapolate", assume_sorted=True)
    trans_interp_z = interp1d(traj_times, translations[:, 2], kind='linear',
                              fill_value="extrapolate", assume_sorted=True)

    # 对每个RGB时间戳进行插值
    interp_rots = slerp(rgb_times).as_matrix()  # (M, 3, 3)

    # 插值平移
    interp_trans_x = trans_interp_x(rgb_times)
    interp_trans_y = trans_interp_y(rgb_times)
    interp_trans_z = trans_interp_z(rgb_times)
    interp_trans = np.vstack([interp_trans_x, interp_trans_y, interp_trans_z]).T  # (M, 3)

    # 构建完整的4x4位姿矩阵
    interp_poses = np.zeros((len(rgb_times), 4, 4))
    interp_poses[:, :3, :3] = interp_rots
    interp_poses[:, :3, 3] = interp_trans
    interp_poses[:, 3, 3] = 1.0  # 设置齐次坐标

    return interp_poses

def interpolate_single_pose(rgb_time: float, traj_times: np.ndarray, traj_poses: np.ndarray) -> np.ndarray:
    """
    根据单个RGB时间戳和轨迹数据，通过插值获得对应的单个位姿 (精简、快速版)。

    此版本针对单个时间点进行优化，并省略了旋转矩阵的有效性检查以最大化速度。
    它假定输入的轨迹位姿是有效的（即旋转矩阵是正交的）。

    参数:
        rgb_time: 需要插值位姿的单个浮点数时间戳。
        traj_times: 轨迹的时间戳数组 (N,)。
        traj_poses: 轨迹的位姿数组 (N, 4, 4)。

    返回:
        interp_pose: 插值后的单个位姿，是一个 (4, 4) 的 NumPy 数组。
    """
    # 【注意】: 为了性能，此函数假定 traj_times 已经是单调递增的。
    # 如果不确定，应在调用此函数前对 traj_times 和 traj_poses 进行排序。
    # sort_idx = np.argsort(traj_times)
    # traj_times = traj_times[sort_idx]
    # traj_poses = traj_poses[sort_idx]

    # 提取旋转和平移分量
    rotations = traj_poses[:, :3, :3]
    translations = traj_poses[:, :3, 3]

    # --- 旋转插值 (Slerp) ---
    try:
        # 从矩阵创建Scipy旋转对象
        rot_objects = Rotation.from_matrix(rotations)
    except ValueError as e:
        # 即使省略了检查，如果矩阵格式错误，scipy依然会报错
        raise ValueError(f"从矩阵创建Rotation对象失败，请检查输入位姿。Scipy错误: {e}")

    # 创建球面线性插值器
    slerp = Slerp(traj_times, rot_objects)
    # 对单个时间点进行插值，直接得到一个 (3, 3) 矩阵
    interp_rot_matrix = slerp(rgb_time).as_matrix()

    # --- 平移插值 (线性) ---
    # 创建一个可以处理整个 (N, 3) 平移向量的插值器
    trans_interp = interp1d(
        traj_times,
        translations,
        axis=0,  # 关键：沿着时间轴进行插值
        kind='linear',
        fill_value="extrapolate",
        assume_sorted=True # 假定时间已排序
    )
    # 对单个时间点进行插值，直接得到一个 (3,) 向量
    interp_translation = trans_interp(rgb_time)

    # --- 组合成最终的位姿矩阵 ---
    # 创建一个4x4的单位矩阵作为基础
    interp_pose = np.eye(4)
    # 填入插值得到的旋转和平移
    interp_pose[:3, :3] = interp_rot_matrix
    interp_pose[:3, 3] = interp_translation

    return interp_pose

def get_mock_timestamps(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Get mock relative timestamps for the velodyne points."""
    # the velodyne has x forward, y left, z up and the sweep is split behind the car.
    # it is also rotating counter-clockwise, meaning that the angles close to -pi are the
    # first ones in the sweep and the ones close to pi are the last ones in the sweep.
    angles = np.arctan2(points[:, 1], points[:, 0])  # N, [-pi, pi]
    angles += np.pi  # N, [0, 2pi]
    # see how much of the rotation have finished
    fraction_of_rotation = angles / (2 * np.pi)  # N, [0, 1]
    # get the pseudo timestamps based on the total rotation time
    timestamps = fraction_of_rotation * LIDAR_ROTATION_TIME
    return timestamps

if __name__ == "__main__":
    # 读取 YAML 文件
    # rgb_path = "/qls/code/dataset/Botanic_Garden/1018_00/1018_00_dalsa_cams/1018_00_rgb_exposed_stamp.txt"
    #
    #
    # print(read_times(rgb_path)[1])
    #
    # rgb_times= [-0.25,0.5,0.75]
    # traj_times = [0,1]
    # traj_poses = [[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0,0,10],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]]
    # print(interpolate_poses(rgb_times, traj_times, traj_poses))
    pc = np.fromfile("/qls/code/dataset/Botanic_Garden/1018_00/bin/1666059817.936109000.bin", dtype=np.float32).reshape(-1, 6)
    print(pc)
    # b = BotanicDataParserConfig()
    # bb = b.setup()
    # print(bb._generate_dataparser_outputs())