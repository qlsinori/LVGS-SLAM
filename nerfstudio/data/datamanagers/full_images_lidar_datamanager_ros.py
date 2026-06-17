# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Data manager that outputs cameras / images and lidars / point clouds instead of raybundles

Good for things like gaussian splatting which require full sensors instead of the standard ray
paradigm
"""

from __future__ import annotations
import shutil
import math
import random
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin,Callable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import map_points_to_lidar_tiles, points_mapping_offset_encode, populate_image_from_points
from rich.progress import track
from typing_extensions import assert_never
import sys
from scipy.spatial.transform import Rotation
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import (
    Lidars,
    LidarType,
    get_lidar_azimuth_resolution,
    get_lidar_elevation_mapping,
    transform_points,
)
from collections import deque
import matplotlib.pyplot as plt
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset ,SimpleDataset
from nerfstudio.data.datasets.lidar_dataset import LidarDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.poses import inverse
from nerfstudio.utils.rich_utils import CONSOLE
from collections import defaultdict
from nerfstudio.utils import writer
from contextlib import contextmanager
import cv2
AZIM_CHANNELS_PER_TILE = 32
ELEV_CHANNELS_PER_TILE = 8
import os
import quaternion
import threading

ROS_AVAILABLE = False
rospy = None  # type: ignore
RosImage = None  # type: ignore
PointCloud2 = None  # type: ignore
CameraInfo = None  # type: ignore
PoseStamped = None  # type: ignore
CvBridge = None  # type: ignore
message_filters = None  # type: ignore

try:
    import rospy as _rospy  # type: ignore
    from sensor_msgs.msg import Image as _RosImage, PointCloud2 as _PointCloud2, CameraInfo as _CameraInfo  # type: ignore
    from geometry_msgs.msg import PoseStamped as _PoseStamped  # type: ignore
    from cv_bridge import CvBridge as _CvBridge  # type: ignore
    import message_filters as _message_filters  # type: ignore
    rospy = _rospy
    RosImage = _RosImage
    PointCloud2 = _PointCloud2
    CameraInfo = _CameraInfo
    PoseStamped = _PoseStamped
    CvBridge = _CvBridge
    message_filters = _message_filters
    ROS_AVAILABLE = True
except ImportError:
    print("[WARN] ROS packages not found. ROS integration disabled.")
# print(f"Current Working Directory is: {os.getcwd()}")

# import pypose as pp
# sys.path.append(r"../MAC-VO-main/DataLoader")
# sys.path.append(r"../MAC-VO-main")
# from Interface  import StereoInertialFrame, StereoFrame, StereoData, IMUData, AttitudeData
# from MACVO_QLS import run_macvo_system
# EDN2NED = pp.from_matrix(torch.tensor([
#     [0., 0., 1., 0.],
#     [1., 0., 0., 0.],
#     [0., 1., 0., 0.],
#     [0., 0., 0., 1.],
# ]), pp.SE3_type)
# NED2EDN = EDN2NED.Inv()


@contextmanager
def in_directory(path: Path):
    """一个上下文管理器，可以临时将当前工作目录切换到指定路径。"""
    original_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)




@dataclass
class FullImageLidarDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: FullImageLidarDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=DataParserConfig) # in splatad PandaSetDataParserConfig(add_missing_points=True)
    eval_num_lidars_to_sample_from: int = -1
    """Number of lidars to sample during eval iteration."""
    eval_num_times_to_repeat_lidars: int = -1
    """When not evaluating on all lidars, number of iterations before picking
    new lidars. If -1, never pick new lidars."""
    eval_lidar_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the lidar indices to use during eval; if None, uses all."""
    cache_lidars: Literal["cpu", "gpu"] = "gpu"
    """Whether to cache lidars in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images and lidars. If None, uses all available threads."""
    downsample_factor: float = 1
    """Downsample factor for the lidar. If <1, downsample will be used."""
    paint_points: bool = True
    """Whether to project points into image and store their RGB values."""
    paint_points_topk: int = 9
    """Number of top cameras to use for painting points.
    For example, 2 means the two closest of every camera (2 front, 2 left, 2 right, etc)."""
    train_lidar_only: bool = False
    """Whether to only train on lidar data."""
    train_image_only: bool = False
    """Whether to only train on image data."""
    slam: bool = False
    bev:bool = False
    windows_size: int = 15 # 1hz 对应 *3的sensor
    windows_iter:int = 100
    init_frames: int = 50000000 # 1hz 对应 *3的sensor
    macvo2opencv:torch.tensor = torch.tensor([[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0]])
    use_macvo: bool = False
    select_every_k_frame: int = 4
    slam_cameara_only: bool = True
    win_all_split:int = 4
    thre_angle_scene: float = 8.5 # 2.0 for bot
    thre_angle: float = 3.0
    thre_dis: float = 1.0
    noval_eval_width: int = -1
    kiss_gs_diffix: bool = False
    eval_image: bool = False
    noval_rate: int = 2
    noval_ag: int = 0
    noval_tr: int = 1
    use_ros: bool = False
    """Enable ROS integration: subscribe to keyframes from SLAM and publish depth."""
    ros_depth_topic: str = "/sensor/camera/depth/image_rect"
    ros_keyframe_pose_topic: str = "/keyframe_pose"
    ros_keyframe_image_topic: str = "/keyframe_image"
    ros_keyframe_cloud_topic: str = "/keyframe_cloud"

class FullImageLidarDatamanager(FullImageDatamanager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: FullImageLidarDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    noval_dataset: List[SimpleDataset]
    train_lidar_dataset: LidarDataset
    eval_lidar_dataset: LidarDataset
    slam_dataset: List
    slam_train_dataset:List
    slam_eval_dataset: List
    slam_for_depth :deque
    noval_dataset: List
    windows_right: int = 0
    windows_right_train: int = -1
    camera_index : int = 0
    count: int = 0
    slam_system = None
    init_pose_MACVO = None
    skip_pose_MACVO = None
    init_pose_NF = None

    finished = False
    step_for_schedule: int = 0
    delta_angle: float = 0
    scene_it: Dict = {}
    scene_left: Dict = {}
    ALL_scene:bool = False
    ALL_map:bool = False
    diffix_one: Optional[Callable] = None
    depth_index:int = 0

    def _has_eval_cameras(self) -> bool:
        """True if the dataparser config defines eval_cameras and it's non-empty."""
        eval_cameras = getattr(self.dataparser.config, "eval_cameras", ()) or ()
        return len(eval_cameras) > 0

    def _eval_camera_sensor_idx(self) -> Optional[int]:
        """Sensor idx for the first eval camera stream in slam_dataset.

        In get_sorted_train_stream(), eval cameras are appended after train cameras, so their sensor_idx starts at
        len(dataparser.config.cameras).
        """
        if not self._has_eval_cameras():
            return None
        try:
            return len(self.dataparser.config.cameras)
        except Exception:
            return None

    def _find_camera_frame_by_time_and_sensor(
        self, timestamp: float, sensor_idx: int, time_tolerance_s: float = 0.05
    ) -> Optional[Tuple[Cameras, Dict, float]]:
        """Find closest camera frame in `self.slam_dataset` by timestamp & sensor_idx."""
        if not hasattr(self, "slam_dataset") or self.slam_dataset is None:
            return None

        best_item = None
        best_dt = None
        for item in self.slam_dataset:
            # item: [timestamp, sensor_type, sensor_obj, data_dict, sensor_idx, delta_angle, slam_index]
            if len(item) < 5 or item[1] != "camera":
                continue
            if int(item[4]) != int(sensor_idx):
                continue
            dt = abs(float(item[0]) - float(timestamp))
            if dt > time_tolerance_s:
                continue
            if best_dt is None or dt < best_dt:
                best_item = item
                best_dt = dt
                if best_dt == 0.0:
                    break

        if best_item is None:
            return None
        return best_item[2], best_item[3], float(best_item[0])

    def __init__(
        self,
        config: FullImageLidarDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        diffix_one: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

        # if len(self.train_lidar_dataset) > 500 and self.config.cache_lidars == "gpu":
        #     CONSOLE.print(
        #         "Lidar train dataset has over 500 point clouds, overriding cache_lidars to cpu",
        #         style="bold yellow",
        #     )
        #     self.config.cache_lidars = "cpu"
        self.current_time_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if self.config.paint_points:
            CONSOLE.log("Painting lidar points")
            self.paint_points()
            CONSOLE.log("Finish painting lidar points")
        self.diffix_one = diffix_one
        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_lidars = [i for i in range(len(self.train_lidar_dataset))]
        self.eval_unseen_lidars = [i for i in range(len(self.eval_lidar_dataset))]

        assert len(self.train_unseen_lidars) > 0, "No data found in dataset"
        self.train_unseen_noval = []

        if self.config.slam:
            self.slam_dataset,self.slam_train_dataset ,self.slam_eval_dataset,self.slam_for_depth= self.get_sorted_train_stream()
            self.noval_dataset = []
            # with in_directory("../MAC-VO-main"):
            #     self.slam_system = run_macvo_system()

        self.slam_timing_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})

        # --- ROS Integration ---
        self._ros_enabled = False
        self._ros_keyframe_buffer = []
        self._ros_buffer_lock = threading.Lock()
        self._ros_kf_index_counter = 0
        self._depth_pub = None
        self._cv_bridge = None
        self._camera_template = None
        self._ros_waiting_for_keyframe = False

        if self.config.use_ros and ROS_AVAILABLE:
            self._init_ros()
    # ==================== ROS Integration Methods ====================

    def _init_ros(self):
        """Initialize ROS node, publishers, and subscribers."""
        if not ROS_AVAILABLE:
            print("[ERROR] ROS not available, cannot initialize ROS integration.")
            return

        if not rospy.core.is_initialized():
            rospy.init_node('gs_mapping_node', anonymous=True, disable_signals=True)

        self._cv_bridge = CvBridge()

        # Extract camera template from first camera in dataset for constructing Camera objects
        if hasattr(self, 'slam_train_dataset') and len(self.slam_train_dataset) > 0:
            for item in self.slam_train_dataset:
                if item[1] == 'camera':
                    self._camera_template = deepcopy(item[2])
                    break

        if self._camera_template is None and hasattr(self, 'slam_dataset') and len(self.slam_dataset) > 0:
            for item in self.slam_dataset:
                if item[1] == 'camera':
                    self._camera_template = deepcopy(item[2])
                    break

        if self._camera_template is None:
            print("[WARN] No camera template found. ROS keyframe reception may not work correctly.")

        self._depth_pub = rospy.Publisher(
            self.config.ros_depth_topic,
            RosImage,
            queue_size=10
        )

        sub_pose = message_filters.Subscriber(self.config.ros_keyframe_pose_topic, PoseStamped)
        sub_image = message_filters.Subscriber(self.config.ros_keyframe_image_topic, RosImage)
        sub_cloud = message_filters.Subscriber(self.config.ros_keyframe_cloud_topic, PointCloud2)

        self._ros_sync = message_filters.ApproximateTimeSynchronizer(
            [sub_pose, sub_image, sub_cloud],
            queue_size=30,
            slop=0.1
        )
        self._ros_sync.registerCallback(self._ros_keyframe_callback)

        self._ros_enabled = True
        self._ros_kf_index_counter = len(self.slam_train_dataset) if hasattr(self, 'slam_train_dataset') else 0
        print(f"[ROS] 3DGS mapping node initialized.")
        print(f"[ROS]   Subscribing: {self.config.ros_keyframe_pose_topic}, "
              f"{self.config.ros_keyframe_image_topic}, {self.config.ros_keyframe_cloud_topic}")
        print(f"[ROS]   Publishing:  {self.config.ros_depth_topic}")

    def _ros_keyframe_callback(self, pose_msg, image_msg, cloud_msg):
        """Synchronized callback: receive keyframe (pose + image + cloud) from SLAM module."""
        try:
            timestamp = pose_msg.header.stamp.to_sec()

            # --- Convert image ---
            cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
            if cv_image.dtype == np.uint8:
                image_tensor = torch.from_numpy(cv_image.copy()).float() / 255.0
            elif cv_image.dtype == np.uint16:
                image_tensor = torch.from_numpy(cv_image.copy()).float() / 65535.0
            else:
                image_tensor = torch.from_numpy(cv_image.copy()).float()

            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
            elif image_tensor.shape[-1] == 1:
                image_tensor = image_tensor.repeat(1, 1, 3)

            # --- Convert pose (PoseStamped -> 4x4 matrix) ---
            pose_4x4 = self._pose_msg_to_4x4(pose_msg)

            # --- Construct Camera object using template ---
            camera_obj = self._construct_camera_from_pose(pose_4x4, timestamp)
            if camera_obj is None:
                print("[ROS] Failed to construct camera from keyframe pose. Skipping.")
                return

            data_dict = {'image': image_tensor}

            # --- Compute delta angle ---
            delta_angle = 0.0
            if hasattr(self, 'slam_train_dataset') and len(self.slam_train_dataset) > 0:
                last_cam = None
                for item in reversed(self.slam_train_dataset):
                    if item[1] == 'camera':
                        last_cam = item[2]
                        break
                if last_cam is not None:
                    try:
                        from scipy.spatial.transform import Rotation as R
                        cur_rot = pose_4x4[:3, :3]
                        last_rot = last_cam.camera_to_worlds.squeeze(0)[:3, :3].cpu().numpy()
                        r_delta = R.from_matrix(last_rot).inv() * R.from_matrix(cur_rot)
                        delta_angle = np.degrees(r_delta.magnitude())
                    except Exception:
                        delta_angle = 0.0

            with self._ros_buffer_lock:
                idx = self._ros_kf_index_counter
                self._ros_kf_index_counter += 1
                entry = [timestamp, 'camera', camera_obj, data_dict, 0, delta_angle, idx]
                self._ros_keyframe_buffer.append(entry)

            print(f"[ROS] Received keyframe: t={timestamp:.3f}, idx={idx}, "
                  f"image={image_tensor.shape}, delta_angle={delta_angle:.2f}")

        except Exception as e:
            print(f"[ROS] Error in keyframe callback: {e}")
            import traceback
            traceback.print_exc()

    def _pose_msg_to_4x4(self, pose_msg):
        """Convert geometry_msgs/PoseStamped to 4x4 numpy matrix."""
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot
        mat[:3, 3] = [p.x, p.y, p.z]
        return mat

    def _construct_camera_from_pose(self, pose_4x4_np, timestamp):
        """Build a Cameras object by cloning the template and applying a new pose + timestamp."""
        if self._camera_template is None:
            return None

        camera = deepcopy(self._camera_template)

        opencv_to_nerfstudio_4x4 = np.eye(4, dtype=np.float32)
        opencv_to_nerfstudio_4x4[:3, :3] = OPENCV_TO_NERFSTUDIO

        pose_nerfstudio = pose_4x4_np @ opencv_to_nerfstudio_4x4
        camera.camera_to_worlds = torch.from_numpy(
            pose_nerfstudio[:3, :4]
        ).float().unsqueeze(0).to(camera.device)

        camera.times = torch.tensor([[timestamp]], dtype=torch.float32, device=camera.device)

        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["scene_index"] = 0

        return camera

    def _process_ros_keyframes(self):
        """Pop buffered ROS keyframes and append them to slam_train_dataset."""
        if not self._ros_enabled:
            return 0

        with self._ros_buffer_lock:
            new_entries = list(self._ros_keyframe_buffer)
            self._ros_keyframe_buffer.clear()

        if len(new_entries) == 0:
            return 0

        for entry in new_entries:
            cam = entry[2]
            if cam.metadata is None:
                cam.metadata = {}
            scene_idx = 0
            if len(self.slam_train_dataset) > 0:
                last_scene = self.slam_train_dataset[-1][2].metadata.get("scene_index", 0)
                scene_idx = last_scene
            cam.metadata["scene_index"] = scene_idx

            self.slam_train_dataset.append(entry)
            self.slam_for_depth.append(entry)

            if scene_idx not in self.scene_it:
                self.scene_it[scene_idx] = 0
            if scene_idx not in self.scene_left:
                self.scene_left[scene_idx] = 0

        print(f"[ROS] Added {len(new_entries)} keyframes. "
              f"Total slam_train_dataset: {len(self.slam_train_dataset)}")
        return len(new_entries)

    def publish_depth(self, model, camera, timestamp):
        """Render depth from the 3DGS model and publish as ROS Image (16UC1, depth in mm)."""
        if not self._ros_enabled or self._depth_pub is None:
            return

        try:
            model.eval()
            with torch.no_grad():
                outputs = model.get_outputs(camera)

            depth_tensor = outputs["depth"].squeeze()
            depth_in_mm = depth_tensor * 1000.0
            depth_image_16uc1 = depth_in_mm.cpu().numpy().astype(np.uint16)

            msg = self._cv_bridge.cv2_to_imgmsg(depth_image_16uc1, encoding='16UC1')
            msg.header.stamp = rospy.Time.from_sec(timestamp)
            msg.header.frame_id = "sensor/camera"
            self._depth_pub.publish(msg)

        except Exception as e:
            print(f"[ROS] Error publishing depth: {e}")

    def has_pending_ros_keyframes(self):
        """Check if there are buffered ROS keyframes waiting to be processed."""
        if not self._ros_enabled:
            return False
        with self._ros_buffer_lock:
            return len(self._ros_keyframe_buffer) > 0

    # ==================== End ROS Integration Methods ====================

    @cached_property
    def cached_lidar_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_lidars("train", cache_lidars_device=self.config.cache_lidars)

    @cached_property
    def cached_lidar_eval(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_lidars("eval", cache_lidars_device=self.config.cache_lidars)

    def _lidar_to_raster_pts(
        self,
        point_cloud,
        lidar,
        elevation_boundaries,
        elevation_mapping,
        azimuth_resolution,
        did_return_threshold,
        is_eval,
    ):
        if not is_eval:
            # shuffle points
            point_cloud = point_cloud[torch.randperm(point_cloud.shape[0])]

        # Remove ego motion
        rs_adjusted_point_cloud = point_cloud[:, :3] - lidar.metadata["linear_velocities_local"] * point_cloud[..., 4:5]

        azimuth = torch.rad2deg(torch.atan2(rs_adjusted_point_cloud[:, 1], rs_adjusted_point_cloud[:, 0]))
        distance = torch.linalg.vector_norm(rs_adjusted_point_cloud[:, :3], dim=1)
        elevation = torch.rad2deg(torch.asin(rs_adjusted_point_cloud[:, 2] / distance))

        intensity = point_cloud[:, 3]
        point_cloud_time = point_cloud[:, 4]
        spherical_coords_time_intensity = torch.stack(
            [azimuth, elevation, distance, point_cloud_time, intensity], dim=1
        ).cuda()
        points_tile_ids, flatten_ids = map_points_to_lidar_tiles(
            spherical_coords_time_intensity[None, :, :2],
            elevation_boundaries,
            azimuth_resolution * AZIM_CHANNELS_PER_TILE,
            -180.0,
        )
        tile_width = math.ceil(360 / (azimuth_resolution * AZIM_CHANNELS_PER_TILE))
        tile_height = len(elevation_boundaries) - 1
        tile_offsets = points_mapping_offset_encode(
            points_tile_ids,
            1,
            tile_width,
            tile_height,
        )

        image_width = tile_width * AZIM_CHANNELS_PER_TILE
        image_height = len(elevation_mapping)

        if is_eval:
            points_per_tile = torch.cat(
                [tile_offsets.flatten(), torch.tensor([point_cloud.shape[0]], device=tile_offsets.device)]
            ).diff()
            max_points_per_tile = ELEV_CHANNELS_PER_TILE * AZIM_CHANNELS_PER_TILE
            n_batches = (points_per_tile // (max_points_per_tile + 1)).max() + 1
            raster_pts_image = torch.zeros((n_batches, image_height, image_width, 5), device=point_cloud.device)
            for batch_idx in range(n_batches):
                flatten_ids_batch = torch.cat(
                    [
                        flatten_ids[s : (s + n)]
                        for s, n in zip(
                            (tile_offsets.flatten() + max_points_per_tile * batch_idx),
                            points_per_tile.clamp_max(max_points_per_tile),
                        )
                    ]
                )
                tile_offsets_batch = (
                    torch.cat(
                        [
                            torch.tensor([0], device=points_per_tile.device),
                            points_per_tile.clamp_max(max_points_per_tile).cumsum(dim=0)[:-1],
                        ]
                    )
                    .view(tile_offsets.shape)
                    .int()
                )
                points_per_tile = (points_per_tile - max_points_per_tile).clamp_min(0)

                raster_pts_image[batch_idx] = populate_image_from_points(  # (azimuth, elev, depth, time, intensity)
                    spherical_coords_time_intensity[None],
                    image_width=image_width,
                    image_height=image_height,
                    tile_width=AZIM_CHANNELS_PER_TILE,
                    tile_height=ELEV_CHANNELS_PER_TILE,
                    tile_offsets=tile_offsets_batch,
                    flatten_id=flatten_ids_batch,
                )

        else:
            raster_pts_image = populate_image_from_points(  # (azimuth, elev, depth, time, intensity)
                spherical_coords_time_intensity[None],
                image_width=image_width,
                image_height=image_height,
                tile_width=AZIM_CHANNELS_PER_TILE,
                tile_height=ELEV_CHANNELS_PER_TILE,
                tile_offsets=tile_offsets,
                flatten_id=flatten_ids,
            )

        return raster_pts_image

    def _add_metadata(self, lidar, data, num_cameras):
        data["lidar"] = data["lidar"].to(self.device)
        data["elevation_boundaries"] = data["elevation_boundaries"].to(self.device)
        data["elevation_mapping"] = data["elevation_mapping"].to(self.device)
        lidar.metadata["elevation_boundaries"] = data["elevation_boundaries"]
        lidar.metadata["azimuth_resolution"] = data["azimuth_resolution"]
        lidar.metadata["cam_idx"] = lidar.metadata["lidar_idx"] + num_cameras
        raster_pts_image = self._lidar_to_raster_pts(
            data["lidar"],
            lidar,
            data["elevation_boundaries"],
            data["elevation_mapping"],
            data["azimuth_resolution"],
            lidar.valid_lidar_distance_threshold,
            data["is_eval"],
        )
        data["raster_pts"] = raster_pts_image
        data["raster_pts_did_return"] = raster_pts_image[..., 2] <= lidar.valid_lidar_distance_threshold
        data["raster_pts_valid_depth_and_did_return"] = (
            (data["raster_pts_did_return"] & (raster_pts_image[..., 2] > 0)).flatten().nonzero().squeeze()
        )
        data["raster_pts_valid_depth_and_did_not_return"] = (
            (~data["raster_pts_did_return"] & (raster_pts_image[..., 2] > 0)).flatten().nonzero().squeeze()
        )
        lidar.metadata["raster_pts"] = raster_pts_image
        data["lidar_pts_did_return"] = data["lidar"].norm(dim=-1) <= lidar.valid_lidar_distance_threshold
        data["linear_velocities_local"] = lidar.metadata["linear_velocities_local"]

    def _load_lidars(self, split, cache_lidars_device):
        # Which dataset?
        if split == "train":
            dataset = self.train_lidar_dataset
        elif split == "eval":
            dataset = self.eval_lidar_dataset
        else:
            assert_never(split)

        def process_data(idx):
            data = dataset.get_data(idx)
            lidar = dataset.lidars[idx : idx + 1]
            lidar_type = LidarType(lidar.lidar_type.item())
            elevation_mapping = get_lidar_elevation_mapping(lidar_type)
            elevation_mapping = torch.tensor(sorted(elevation_mapping.values())).float()
            elevation_boundaries = torch.cat(
                [
                    elevation_mapping[0:1] - 1.0,
                    (
                        elevation_mapping[ELEV_CHANNELS_PER_TILE::ELEV_CHANNELS_PER_TILE]
                        + elevation_mapping[ELEV_CHANNELS_PER_TILE - 1 : -1 : ELEV_CHANNELS_PER_TILE]
                    )
                    / 2,
                    elevation_mapping[-1:] + 1.0,
                ]
            )
            azimuth_resolution = get_lidar_azimuth_resolution(lidar_type)
            data["elevation_boundaries"] = elevation_boundaries.cpu()
            data["elevation_mapping"] = elevation_mapping.cpu()
            data["azimuth_resolution"] = azimuth_resolution
            data["is_eval"] = split == "eval"
            return data

        CONSOLE.log(f"Caching {split} lidars")
        with ThreadPoolExecutor(max_workers=2) as executor:
            cached_data = list(
                track(
                    executor.map(
                        process_data,
                        range(len(dataset)),
                    ),
                    description=f"Caching {split} lidars",
                    transient=True,
                    total=len(dataset),
                )
            )

        if cache_lidars_device == "gpu":
            for cache in cached_data:
                cache["lidar"] = cache["lidar"].to(self.device)
                cache["elevation_boundaries"] = cache["elevation_boundaries"].to(self.device)
                cache["elevation_mapping"] = cache["elevation_mapping"].to(self.device)
                self.train_lidars = self.train_lidar_dataset.lidars.to(self.device)
        else:
            for cache in cached_data:
                cache["lidar"] = cache["lidar"].pin_memory()
                cache["elevation_boundaries"] = cache["elevation_boundaries"].pin_memory()
                cache["elevation_mapping"] = cache["elevation_mapping"].pin_memory()
                self.train_lidars = self.train_lidar_dataset.lidars

        CONSOLE.log(f"Finish caching {split} lidars")
        return cached_data

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""

        self.noval_dataset = []
        self.train_lidar_dataset = LidarDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            downsample_factor=self.config.downsample_factor,
        )
        return super().dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        # self.eval_lidar_dataset = None
        self.eval_lidar_dataset = LidarDataset(
            dataparser_outputs=eval_dataparser_outputs,
            downsample_factor=self.config.downsample_factor,
        )
        return super().dataset_type(
            dataparser_outputs=eval_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_lidar_dataloader(self) -> List[Tuple[Lidars, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (lidar, data) tuples
        """
        lidar_indices = [i for i in range(len(self.eval_lidar_dataset))]
        data = [d.copy() for d in self.cached_lidar_eval]
        _lidars = deepcopy(self.eval_lidar_dataset.lidars).to(self.device)
        lidars = []
        for i in lidar_indices:
            data[i]["lidar"] = data[i]["lidar"].to(self.device)
            _lidar = _lidars[i : i + 1]
            _lidar.metadata["lidar_idx"] = i
            self._add_metadata(_lidar, data[i], len(self.eval_dataset))
            lidars.append(_lidar)
        assert len(self.eval_lidar_dataset.lidars.shape) == 1, "Assumes single batch dimension"
        return list(zip(lidars, data))

    @property
    def fixed_indices_train_lidar_dataloader(self) -> List[Tuple[Lidars, Dict]]:
        """
        Pretends to be the dataloader for train, it returns a list of (lidar, data) tuples
        """
        lidar_indices = [i for i in range(len(self.train_lidar_dataset))]
        data = [d.copy() for d in self.cached_lidar_train]
        _lidars = deepcopy(self.train_lidar_dataset.lidars).to(self.device)
        lidars = []
        for i in lidar_indices:
            data[i]["lidar"] = data[i]["lidar"].to(self.device)
            _lidar = _lidars[i : i + 1]
            _lidar.metadata["lidar_idx"] = i
            self._add_metadata(_lidar, data[i], len(self.train_dataset))
            lidars.append(_lidar)
        assert len(self.train_lidar_dataset.lidars.shape) == 1, "Assumes single batch dimension"
        return list(zip(lidars, data))

    def next_train_lidar(self, step: int) -> Tuple[Lidars, Dict]:
        """Returns the next training batch"""
        lidar_idx = self.train_unseen_lidars.pop(random.randint(0, len(self.train_unseen_lidars) - 1))

        assert len(self.train_lidars.shape) == 1, "Assumes single batch dimension"
        lidar = self.train_lidars[lidar_idx : lidar_idx + 1].to(self.device)
        if lidar.metadata is None:
            lidar.metadata = {}
        lidar.metadata["lidar_idx"] = lidar_idx

        data = self.cached_lidar_train[lidar_idx]
        data = data.copy()

        self._add_metadata(lidar, data, len(self.train_dataset))

        return lidar, data

    def next_train_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))

        data = self.cached_train[image_idx]
        # We're going to copy to make sure we don't mutate the cached dictionary.
        # This can cause a memory leak: https://github.com/nerfstudio-project/nerfstudio/issues/3335
        data = data.copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_noval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        ndataset = self.noval_dataset[-1]

        try :
            image_idx = self.train_unseen_noval.pop(random.randint(0, len(self.train_unseen_noval) - 1))
        except:
            print(" len(ndataset):", len(ndataset),self.train_unseen_noval)


        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = ndataset.cameras[image_idx].to(self.device)
        data = ndataset.data[image_idx]

        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data
    def next_train(self, step: int,model = None) -> Tuple[Union[Cameras, Lidars], Dict]:
        """Returns the next training batch

        Returns a Camera or Lidar instead of raybundle"""

        if self.config.slam:
            return self.next_slam(step,model)

        if (len(self.train_unseen_cameras) + len(self.train_unseen_lidars)) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
            self.train_unseen_lidars = [i for i in range(len(self.train_lidar_dataset))]

        if self.train_unseen_noval == [] and len(self.noval_dataset)!=0:
            self.train_unseen_noval = [i for i in range(len(self.noval_dataset[-1]))]

        if self.config.train_lidar_only:
            self.train_unseen_cameras = []
        if self.config.train_image_only:
            self.train_unseen_lidars = []

        # return self.next_train_image(step)
        # if len(self.noval_dataset) != 0 and step < 35000:
        if len(self.noval_dataset) != 0 :
            if random.randint(1, 10) < 3:
                a, b = self.next_noval_image(step)
                return a, b
        if random.randint(0, len(self.train_unseen_cameras) + len(self.train_unseen_lidars) - 1) < len(
            self.train_unseen_cameras
        ):
            return self.next_train_image(step)
        else:
            return self.next_train_lidar(step)

    def next_slam(self, step: int,model = None) -> Tuple[Union[Cameras, Lidars], Dict]:
        # print("   self.count :",   self.count)

        # --- [ROS] Process any pending keyframes from SLAM ---
        if self._ros_enabled:
            self._process_ros_keyframes()

        timers = {
            "Mapping_Extend": [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)],
            "Diffix_Gen": [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)],  # 通常只有Camera
            "Mapping_Extend1": [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)],
            "Mapping_Extend2": [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)],
            "Mapping_Extend3": [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)],
            "Mapping_Extend4": [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)],
        } ###qls

        mapping_type_str = None  # 建图发生时的传感器类型 ('Camera' / 'Lidar')###qls

        if(self.count <= 0):
            # --- [ROS] When using ROS, wait for new keyframes if we've caught up ---
            if self._ros_enabled and self.windows_right_train + 1 >= len(self.slam_train_dataset):
                if not self.has_pending_ros_keyframes():
                    self.count = 1
                    if len(self.slam_train_dataset) > 0:
                        self._ros_waiting_for_keyframe = True
                        if step % 300 == 0:
                            print(f"[ROS] Waiting for new keyframes... "
                                  f"(current: {len(self.slam_train_dataset)})")
                        window_right = min(self.windows_right_train, len(self.slam_train_dataset) - 1)
                        if window_right >= 0:
                            index = random.randint(0, window_right)
                            t, s, sensor_obj, data_dict, s_id, d_r, slam_frame_idx = self.slam_train_dataset[index]
                            return sensor_obj, data_dict
                else:
                    self._process_ros_keyframes()
                    self._ros_waiting_for_keyframe = False

            self.windows_right_train = self.windows_right_train + 1
            if step == 0 :
                # self.save_pose(step, Path("./"))
                pass
            # if self.windows_right_train < len(self.slam_train_dataset) and False:
            #     for self.windows_right in range(self.slam_train_dataset[self.windows_right_train][-1],
            #                                     self.slam_train_dataset[self.windows_right_train + 1][-1]
            #                                     if self.windows_right_train + 1 < len(self.slam_train_dataset) else len(self.slam_train_dataset) ):
            #     #slam part
            #
            #         if (self.windows_right < self.config.init_frames):
            #             #use gt
            #             pass
            #         elif self.config.kiss_gs_diffix and   self.config.use_macvo and self.slam_dataset[self.windows_right][1] == 'lidar'  :
            #
            #             if self.windows_right + 100 <  len(self.slam_dataset) and  self.slam_dataset[self.windows_right][2].metadata["scene_index"] == \
            #                 self.slam_dataset[self.windows_right + 100][2].metadata["scene_index"]:
            #                 continue
            #
            #             lidar_obj = deepcopy(self.slam_dataset[self.windows_right][2])
            #
            #             for i in range(self.windows_right,-1,-1):
            #                 if self.slam_dataset[i][4] ==0:
            #                     nearest_camera = deepcopy(self.slam_dataset[i][2])
            #                     break
            #
            #             ltw = lidar_obj.lidar_to_worlds
            #
            #             bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=ltw.device,
            #                                       dtype=ltw.dtype)
            #             ltw = torch.cat((ltw, bottom_row), dim=1)
            #
            #             lidar2lcam_tensor = torch.from_numpy(nearest_camera.metadata["lidar2lcam"]).to(
            #                 device=ltw.device,
            #                 dtype=ltw.dtype
            #             )
            #             # T_rgb0_vlp16
            #             opencv_to_nerfstudio_4x4 = np.eye(4, dtype=np.float32)
            #             opencv_to_nerfstudio_4x4[:3, :3] = OPENCV_TO_NERFSTUDIO
            #
            #             ctw = ltw.squeeze(0) @ torch.inverse(lidar2lcam_tensor)
            #             device =  ltw.device
            #             nearest_camera.camera_to_worlds =(ctw @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device))[:3,:].unsqueeze(0)
            #
            #             interpolated_pose_4x4_tensor = lidar2lcam_tensor @ ltw.squeeze(
            #                 0) @ torch.inverse(lidar2lcam_tensor)
            #
            #
            #
            #             camera_obj_single = nearest_camera
            #             camera_obj_next = deepcopy(camera_obj_single)
            #             camera_obj_next.camera_to_worlds =( ctw @ torch.from_numpy(nearest_camera.metadata["rcam2lcam"]).to(device) @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device) )[:3,:].unsqueeze(0)
            #
            #             img1 = self.diffix_one(camera_obj_single, self.slam_dataset[i-60][3]['image'])
            #             data_dict = {'image': img1}
            #
            #             img2 = self.diffix_one(camera_obj_next, self.slam_dataset[i-60+1][3]['image']) # 确保两帧时间戳一样
            #             data_next = {'image': img2}
            #
            #             new_stereo_frame,_ = self.odom(interpolated_pose_4x4_tensor, camera_obj_single, camera_obj_next,
            #                                          data_dict, data_next)
            #
            #             if self.windows_right_train == len(self.slam_train_dataset) - 1:
            #                 mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=True)
            #             else:
            #                 mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=False)
            #
            #             self.camera_index += 1
            #
            #             # lcam_pose = torch.inverse(lidar2lcam_tensor) @ (
            #             #             self.init_pose @ torch.from_numpy(mac_vo_pose)) @ lidar2lcam_tensor
            #             # lcam_pose = lcam_pose @ torch.inverse(lidar2lcam_tensor)
            #
            #             # self.slam_dataset[self.windows_right][2].camera_to_worlds = (lcam_pose @ torch.from_numpy(
            #             #     opencv_to_nerfstudio_4x4)).unsqueeze(0).to(self.device)[:, :3, :]
            #             #
            #             # for j in range(self.windows_right + 1, len(self.slam_dataset)):
            #             #     if self.slam_dataset[self.windows_right][0] == self.slam_dataset[j][0]:
            #             #         rcam_pose = lcam_pose @ torch.from_numpy(
            #             #             camera_obj_single.metadata["rcam2lcam"]) @ torch.from_numpy(
            #             #             opencv_to_nerfstudio_4x4)
            #             #         self.slam_dataset[j][2].camera_to_worlds = rcam_pose.unsqueeze(0).to(self.device)[:, :3,
            #             #                                                     :]
            #             #         break
            #
            #         elif self.slam_dataset[ self.windows_right][4] ==0:
            #             #use lidar inter
            #             lidar1_data, lidar2_data = self._find_bracketing_lidars(self.windows_right)
            #
            #             if lidar1_data is None or lidar2_data is None:
            #                 print("lidar1_data or lidar2_data is None")
            #                 return
            #
            #             interpolated_pose_tensor = self._interpolate_pose(
            #                 camera_timestamp=self.slam_dataset[ self.windows_right][0],
            #                 lidar1_data=lidar1_data,
            #                 lidar2_data=lidar2_data,
            #             )
            #             opencv_to_nerfstudio_4x4 = np.eye(4, dtype=np.float32)
            #             opencv_to_nerfstudio_4x4[:3, :3] = OPENCV_TO_NERFSTUDIO
            #
            #
            #             # interpolated_pose_tensor = self.slam_dataset[self.windows_right][2].camera_to_worlds
            #             test0 = self.slam_dataset[self.windows_right][2].camera_to_worlds
            #             camera_obj_single = self.slam_dataset[self.windows_right][2]
            #             camera_obj_next = self.slam_dataset[self.windows_right + 1][2]
            #             data_dict = deepcopy(self.slam_dataset[self.windows_right][3])
            #             data_next = deepcopy(self.slam_dataset[self.windows_right + 1][3])
            #             device = camera_obj_single.device
            #             interpolated_pose_tensor = interpolated_pose_tensor.to(device)
            #             # metadata = {"sensor_idxs": idxs, "extrinsic": self.calibs["T_rgb0_rgb1"],
            #             #             "lidar2lcam": self.calibs["T_rgb0_vlp16"]},
            #             bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=interpolated_pose_tensor.device,
            #                                       dtype=interpolated_pose_tensor.dtype)
            #             interpolated_pose_4x4_tensor = torch.cat((interpolated_pose_tensor, bottom_row), dim=1)
            #
            #             # interpolated_pose_4x4_tensor =  interpolated_pose_4x4_tensor @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device)
            #
            #             lidar2lcam_tensor = torch.from_numpy(camera_obj_single.metadata["lidar2lcam"]).to(
            #                 device=interpolated_pose_4x4_tensor.device,
            #                 dtype=interpolated_pose_4x4_tensor.dtype
            #             )
            #             #相机坐标系相机位姿                                #T_rgb0_vlp16
            #             test1 = interpolated_pose_4x4_tensor
            #
            #
            #             # interpolated_pose_4x4_tensor = lidar2lcam_tensor @ (interpolated_pose_4x4_tensor.squeeze(
            #             #     0)@ lidar2lcam_tensor) @ torch.inverse(lidar2lcam_tensor)
            #
            #
            #
            #
            #
            #             if self.config.use_macvo:
            #                     interpolated_pose_4x4_tensor = lidar2lcam_tensor @ interpolated_pose_4x4_tensor.squeeze(
            #                     0) @ torch.inverse(lidar2lcam_tensor)
            #
            #                     new_stereo_frame,delta_pose = self.odom(interpolated_pose_4x4_tensor,camera_obj_single,camera_obj_next,data_dict,data_next)
            #                     if self.windows_right_train == len(self.slam_train_dataset) -1 :
            #                         mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame,finish=True)
            #                     else:
            #                         mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=False)
            #
            #
            #                     mac_vo_pose_tensor = torch.from_numpy(mac_vo_pose)
            #                     # print("mac_vo_pose:", mac_vo_pose_tensor)
            #                     # 2. 将 CPU 张量移动到目标设备
            #                     mac_vo_pose_tensor = mac_vo_pose_tensor.to(device)
            #
            #                     if self.camera_index >= 4 and self.init_pose_NF == None:
            #                         self.init_pose_NF = interpolated_pose_4x4_tensor.to(device)
            #                         self.skip_pose_MACVO = mac_vo_pose_tensor.to(device)
            #                     # == test1
            #
            #                     if self.init_pose_NF == None:
            #                         lcam_pose = torch.inverse(lidar2lcam_tensor) @ (
            #                                 self.init_pose_MACVO.to(device) @ delta_pose.to(device)) @ lidar2lcam_tensor
            #                     else:
            #                         lcam_pose = torch.inverse(lidar2lcam_tensor) @ (self.init_pose_NF @ torch.inverse(self.skip_pose_MACVO) @ mac_vo_pose_tensor) @ lidar2lcam_tensor
            #
            #
            #
            #                     lcam_pose = lcam_pose @ torch.inverse(lidar2lcam_tensor)
            #
            #                     # == test0
            #                     # print("windows_right:", self.windows_right, "camera_to_worlds:",
            #                     #       self.slam_dataset[self.windows_right][2].camera_to_worlds[:, :, 3])
            #                     self.slam_dataset[self.windows_right][2].camera_to_worlds = (lcam_pose.to(device) @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device)).unsqueeze(0)[:,:3,:]
            #                     # print("windows_right:",self.windows_right,"camera_to_worlds:",self.slam_dataset[self.windows_right][2].camera_to_worlds[:,:,3])
            #                     for j in range(self.windows_right+1,len(self.slam_dataset)):
            #                         if  self.slam_dataset[self.windows_right][0] == self.slam_dataset[j][0]:
            #                             rcam_pose = lcam_pose.to(device) @ torch.from_numpy(camera_obj_single.metadata["rcam2lcam"] ).to(device) @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device)
            #                             self.slam_dataset[j][2].camera_to_worlds = rcam_pose.unsqueeze(0)[:,:3,:]
            #                             break
            #             else:
            #                 interpolated_pose_4x4_tensor = (interpolated_pose_4x4_tensor.squeeze(0) @ torch.inverse(lidar2lcam_tensor)).to(device)
            #                 self.slam_dataset[self.windows_right][2].camera_to_worlds = (
            #                         interpolated_pose_4x4_tensor.to(device) @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device) ).unsqueeze(0).to(device)[:,:3,:]
            #                 self.slam_dataset[self.windows_right + 1][2].camera_to_worlds = (
            #                         interpolated_pose_4x4_tensor.to(device) @ torch.from_numpy(camera_obj_single.metadata["rcam2lcam"] ).to(device)
            #                         @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device)).unsqueeze(0).to(device)[:,:3,:]
            #
            #
            # if self.ALL_scene and False :
            #     for self.windows_right in range(self.slam_train_dataset[self.scene_left[
            #         self.slam_train_dataset[self.windows_right_train - 1][2].metadata["scene_index"]]][-1],
            #                                     self.slam_train_dataset[self.windows_right_train - 1][-1]):
            #         # slam part
            #
            #         if (self.windows_right < self.config.init_frames):
            #             # use gt
            #             pass
            #         elif self.config.kiss_gs_diffix and self.config.use_macvo and self.slam_dataset[self.windows_right][
            #             1] == 'lidar':
            #
            #             # if self.windows_right + 100 < len(self.slam_dataset) and \
            #             #         self.slam_dataset[self.windows_right][2].metadata["scene_index"] == \
            #             #         self.slam_dataset[self.windows_right + 100][2].metadata["scene_index"]:
            #             #     continue
            #
            #             lidar_obj = deepcopy(self.slam_dataset[self.windows_right][2])
            #
            #             for i in range(self.windows_right, -1, -1):
            #                 if self.slam_dataset[i][4] == 0:
            #                     nearest_camera = deepcopy(self.slam_dataset[i][2])
            #                     break
            #
            #             ltw = lidar_obj.lidar_to_worlds
            #
            #             bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=ltw.device,
            #                                       dtype=ltw.dtype)
            #             ltw = torch.cat((ltw, bottom_row), dim=1)
            #
            #             lidar2lcam_tensor = torch.from_numpy(nearest_camera.metadata["lidar2lcam"]).to(
            #                 device=ltw.device,
            #                 dtype=ltw.dtype
            #             )
            #             # T_rgb0_vlp16
            #             opencv_to_nerfstudio_4x4 = np.eye(4, dtype=np.float32)
            #             opencv_to_nerfstudio_4x4[:3, :3] = OPENCV_TO_NERFSTUDIO
            #
            #             ctw = ltw.squeeze(0) @ torch.inverse(lidar2lcam_tensor)
            #             device = ltw.device
            #             nearest_camera.camera_to_worlds = (ctw @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device))[
            #                                               :3, :].unsqueeze(0)
            #
            #             interpolated_pose_4x4_tensor = lidar2lcam_tensor @ ltw.squeeze(
            #                 0) @ torch.inverse(lidar2lcam_tensor)
            #
            #             camera_obj_single = nearest_camera
            #             camera_obj_next = deepcopy(camera_obj_single)
            #             camera_obj_next.camera_to_worlds = (ctw @ torch.from_numpy(
            #                 nearest_camera.metadata["rcam2lcam"]).to(device) @ torch.from_numpy(
            #                 opencv_to_nerfstudio_4x4).to(device))[:3, :].unsqueeze(0)
            #
            #             img1 = self.diffix_one(camera_obj_single, self.slam_dataset[max(0,i - 15)][3]['image'])
            #             data_dict = {'image': img1}
            #
            #             img2 = self.diffix_one(camera_obj_next, self.slam_dataset[max(0,i - 15 + 1)][3]['image'])  # 确保两帧时间戳一样
            #             data_next = {'image': img2}
            #
            #             new_stereo_frame, _ = self.odom(interpolated_pose_4x4_tensor, camera_obj_single,
            #                                             camera_obj_next,
            #                                             data_dict, data_next)
            #
            #             if self.windows_right_train == len(self.slam_train_dataset) - 1:
            #                 mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=True)
            #             else:
            #                 mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=False)
            #
            #             self.camera_index += 1
            #
            #
            #         elif self.slam_dataset[self.windows_right][4] == 0:
            #             # use lidar inter
            #             lidar1_data, lidar2_data = self._find_bracketing_lidars(self.windows_right)
            #
            #             if lidar1_data is None or lidar2_data is None:
            #                 print("lidar1_data or lidar2_data is None")
            #                 return
            #
            #             interpolated_pose_tensor = self._interpolate_pose(
            #                 camera_timestamp=self.slam_dataset[self.windows_right][0],
            #                 lidar1_data=lidar1_data,
            #                 lidar2_data=lidar2_data,
            #             )
            #             opencv_to_nerfstudio_4x4 = np.eye(4, dtype=np.float32)
            #             opencv_to_nerfstudio_4x4[:3, :3] = OPENCV_TO_NERFSTUDIO
            #
            #             # interpolated_pose_tensor = self.slam_dataset[self.windows_right][2].camera_to_worlds
            #             test0 = self.slam_dataset[self.windows_right][2].camera_to_worlds
            #             camera_obj_single = self.slam_dataset[self.windows_right][2]
            #             camera_obj_next = self.slam_dataset[self.windows_right + 1][2]
            #             data_dict = deepcopy(self.slam_dataset[self.windows_right][3])
            #             data_next = deepcopy(self.slam_dataset[self.windows_right + 1][3])
            #             device = camera_obj_single.device
            #             interpolated_pose_tensor = interpolated_pose_tensor.to(device)
            #             # metadata = {"sensor_idxs": idxs, "extrinsic": self.calibs["T_rgb0_rgb1"],
            #             #             "lidar2lcam": self.calibs["T_rgb0_vlp16"]},
            #             bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=interpolated_pose_tensor.device,
            #                                       dtype=interpolated_pose_tensor.dtype)
            #             interpolated_pose_4x4_tensor = torch.cat((interpolated_pose_tensor, bottom_row), dim=1)
            #
            #             # interpolated_pose_4x4_tensor =  interpolated_pose_4x4_tensor @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device)
            #
            #             lidar2lcam_tensor = torch.from_numpy(camera_obj_single.metadata["lidar2lcam"]).to(
            #                 device=interpolated_pose_4x4_tensor.device,
            #                 dtype=interpolated_pose_4x4_tensor.dtype
            #             )
            #             # 相机坐标系相机位姿                                #T_rgb0_vlp16
            #             test1 = interpolated_pose_4x4_tensor
            #
            #             # interpolated_pose_4x4_tensor = lidar2lcam_tensor @ (interpolated_pose_4x4_tensor.squeeze(
            #             #     0)@ lidar2lcam_tensor) @ torch.inverse(lidar2lcam_tensor)
            #
            #             if self.config.use_macvo:
            #                 interpolated_pose_4x4_tensor = lidar2lcam_tensor @ interpolated_pose_4x4_tensor.squeeze(
            #                     0) @ torch.inverse(lidar2lcam_tensor)
            #
            #                 new_stereo_frame, delta_pose = self.odom(interpolated_pose_4x4_tensor, camera_obj_single,
            #                                                          camera_obj_next, data_dict, data_next)
            #                 if self.windows_right_train == len(self.slam_train_dataset) - 1:
            #                     mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=True)
            #                 else:
            #                     mac_vo_pose = self.slam_system.receive_3dgs_frames(new_stereo_frame, finish=False)
            #
            #                 mac_vo_pose_tensor = torch.from_numpy(mac_vo_pose)
            #                 # print("mac_vo_pose:", mac_vo_pose_tensor)
            #                 # 2. 将 CPU 张量移动到目标设备
            #                 mac_vo_pose_tensor = mac_vo_pose_tensor.to(device)
            #
            #                 if self.camera_index >= 4 and self.init_pose_NF == None:
            #                     self.init_pose_NF = interpolated_pose_4x4_tensor.to(device)
            #                     self.skip_pose_MACVO = mac_vo_pose_tensor.to(device)
            #                 # == test1
            #
            #                 if self.init_pose_NF == None:
            #                     lcam_pose = torch.inverse(lidar2lcam_tensor) @ (
            #                             self.init_pose_MACVO.to(device) @ delta_pose.to(device)) @ lidar2lcam_tensor
            #                 else:
            #                     lcam_pose = torch.inverse(lidar2lcam_tensor) @ (self.init_pose_NF @ torch.inverse(
            #                         self.skip_pose_MACVO) @ mac_vo_pose_tensor) @ lidar2lcam_tensor
            #
            #                 lcam_pose = lcam_pose @ torch.inverse(lidar2lcam_tensor)
            #
            #                 # == test0
            #                 # print("windows_right:", self.windows_right, "camera_to_worlds:",
            #                 #       self.slam_dataset[self.windows_right][2].camera_to_worlds[:, :, 3])
            #                 self.slam_dataset[self.windows_right][2].camera_to_worlds = (lcam_pose.to(
            #                     device) @ torch.from_numpy(opencv_to_nerfstudio_4x4).to(device)).unsqueeze(0)[:, :3, :]
            #                 # print("windows_right:",self.windows_right,"camera_to_worlds:",self.slam_dataset[self.windows_right][2].camera_to_worlds[:,:,3])
            #                 for j in range(self.windows_right + 1, len(self.slam_dataset)):
            #                     if self.slam_dataset[self.windows_right][0] == self.slam_dataset[j][0]:
            #                         rcam_pose = lcam_pose.to(device) @ torch.from_numpy(
            #                             camera_obj_single.metadata["rcam2lcam"]).to(device) @ torch.from_numpy(
            #                             opencv_to_nerfstudio_4x4).to(device)
            #                         self.slam_dataset[j][2].camera_to_worlds = rcam_pose.unsqueeze(0)[:, :3, :]
            #                         break
            #             else:
            #                 interpolated_pose_4x4_tensor = (
            #                             interpolated_pose_4x4_tensor.squeeze(0) @ torch.inverse(lidar2lcam_tensor)).to(
            #                     device)
            #                 self.slam_dataset[self.windows_right][2].camera_to_worlds = (
            #                                                                                     interpolated_pose_4x4_tensor.to(
            #                                                                                         device) @ torch.from_numpy(
            #                                                                                 opencv_to_nerfstudio_4x4).to(
            #                                                                                 device)).unsqueeze(0).to(
            #                     device)[:, :3, :]
            #                 self.slam_dataset[self.windows_right + 1][2].camera_to_worlds = (
            #                                                                                         interpolated_pose_4x4_tensor.to(
            #                                                                                             device) @ torch.from_numpy(
            #                                                                                     camera_obj_single.metadata[
            #                                                                                         "rcam2lcam"]).to(
            #                                                                                     device)
            #                                                                                         @ torch.from_numpy(
            #                                                                                     opencv_to_nerfstudio_4x4).to(
            #                                                                                     device)).unsqueeze(
            #                     0).to(device)[:, :3, :]
            #     self.save_pose(step, Path("./"))

            if self.windows_right_train > 0 and self.windows_right_train != len(self.slam_train_dataset)  and model is not None:
                new_keyframe_data = self.slam_train_dataset[self.windows_right_train]

                raw_type = new_keyframe_data[1]  # 'camera' or 'lidar' ###qls
                mapping_type_str = "Camera" if raw_type == 'camera' else None ###qls

                if new_keyframe_data[1] == 'camera':
                    new_keyframe_cam = new_keyframe_data[2]
                    slam_frame_idx =  self.windows_right_train

                    # 将 slam_frame_idx 添加到 metadata，以便可视化函数可以获取
                    if new_keyframe_cam.metadata is None:
                        new_keyframe_cam.metadata = {}
                    new_keyframe_cam.metadata["slam_frame_idx"] = slam_frame_idx
                    ###qls
                    cnt = 0
                    for cn in range(self.windows_right_train, 0, -1):
                        if self.slam_train_dataset[cn][1] == 'camera':
                            cnt += 1
                            if cnt == 25:
                                break


                    if  self.config.bev or ( self.diffix_one is not None  and self.windows_right_train > 50 and  cnt == 25) :
                        try:
                            # 1. 复制当前相机并构建旋转60度的新位姿
                            # 假设场景坐标系 Z 轴向上 (LiDAR dataset common practice)
                            from scipy.spatial.transform import Rotation as R


                            novel_data = deepcopy(self.slam_train_dataset[cn])
                            novel_cam = deepcopy(self.slam_train_dataset[cn][2])

                            # --- [NEW] If eval_cameras is enabled, try using the eval camera directly for diffix ---
                            paired_used = False
                            if (not self.config.bev) and self._has_eval_cameras():
                                try:
                                    t_ref = float(self.slam_train_dataset[cn][0])
                                    eval_sensor_idx = self._eval_camera_sensor_idx()
                                    if eval_sensor_idx is not None:
                                        matched = self._find_camera_frame_by_time_and_sensor(
                                            timestamp=t_ref,
                                            sensor_idx=eval_sensor_idx,
                                            time_tolerance_s=0.05,
                                        )
                                        if matched is not None:
                                            eval_cam_obj, _eval_data_dict, t_eval = matched
                                            novel_cam = deepcopy(eval_cam_obj).to(self.device)
                                            novel_data[0] = t_eval
                                            novel_data[4] = eval_sensor_idx
                                            # Keep train image as diffusion reference.
                                            # We only use eval camera pose as novel-view target.
                                            ref_image_tensor = self.slam_train_dataset[cn][3]['image']
                                            paired_used = True
                                            print(
                                                f"[diffix] eval_camera paired: "
                                                f"t_ref={t_ref:.6f} -> t_eval={t_eval:.6f} "
                                                f"(dt={abs(t_eval - t_ref):.6f}s), "
                                                f"eval_sensor_idx={eval_sensor_idx}"
                                            )
                                except Exception:
                                    paired_used = False

                            # 获取当前位姿 [3, 4] -> [4, 4]

                            if not paired_used:
                                pose_4x4 = self.camera_to_worlds_to_numpy(novel_cam.camera_to_worlds)

                                if self.config.bev:
                                    new_pose_4x4 = self.apply_bev(
                                        pose_4x4,
                                        -15.0,  # 旋转角度
                                        [0.0, 2.5, -2.5]  # 平移向量
                                    )
                                elif random.randint(1, 2) <= 1  :
                                    new_pose_4x4 = self.apply_z_rotation_and_translation(
                                        pose_4x4,
                                        self.config.noval_ag*1.0,  # 旋转角度
                                        [0.0, 0.0,0.0 ]  # 平移向量
                                    )
                                else:
                                    new_pose_4x4 = self.apply_z_rotation_and_translation(
                                        pose_4x4,
                                        0.0,  # 旋转角度
                                        [-self.config.noval_tr * 1.0, 0.0, 0.0]  # 平移向量
                                    )

                                # 更新相机位姿
                                novel_cam.camera_to_worlds = torch.from_numpy(new_pose_4x4[0,:3, :4]).float().to(
                                    novel_cam.device).unsqueeze(0)

                            # 2. 获取参考图像 (当前帧的 GT image)
                            if not paired_used:
                                cnt = 0
                                for cn2 in range(cn, 0, -1):
                                    if self.slam_train_dataset[cn2][1] == 'camera':
                                        cnt += 1
                                        if cnt == 1:
                                            break

                                ref_image_tensor = self.slam_train_dataset[cn2][3]['image']

                            # 3. 调用 diffix_one
                            # 注意：diffix_one 内部会使用 model 对 novel_cam 进行渲染，然后结合 ref_image_tensor 进行扩散修复
                            print(
                                f"Generating novel view ({self.config.noval_ag} deg rot {self.config.noval_tr} trans) and running diffusion fix for frame {slam_frame_idx}...")

                            base_output_dir = f"outputs/{self.current_time_str}_gene/"



                            refined_image_tensor = self.diffix_one(novel_cam, ref_image_tensor,timers,save_dir = base_output_dir,frame_idx = cn,bev = self.config.bev)

                            # 确保返回的是 tensor 格式 [C, H, W] 或 [H, W, C]，SimpleDataset 通常处理 tensor
                            if isinstance(refined_image_tensor, np.ndarray):
                                refined_image_tensor = torch.from_numpy(refined_image_tensor)

                            # 如果维度是 HWC (uint8 numpy转换来的), 可能需要 permute 到 CHW 或保持原样取决于 Dataset 实现
                            # 这里的 SimpleDataset 并没有复杂的转换，通常保持与 model 输出一致即可

                            novel_data[2] = novel_cam
                            novel_data[3]['image'] = refined_image_tensor
                            print("refined_image_tensor:",refined_image_tensor.shape)
                            if not(self.config.bev):
                                self.noval_dataset.append(novel_data)

                            # 更新 train_unseen_noval 以便下一次 next_noval_image 可以采样到
                            # 注意：这里逻辑是根据最新的 noval_dataset 更新索引列表


                            print(
                                f"Added generated view to noval_dataset. Total noval datasets: {len(self.noval_dataset)}")

                        except Exception as e:
                            print(f"Error in diffix_one generation: {e}")
                            import traceback
                            traceback.print_exc()



                    print(f"\nAttempting to extend map with new keyframe (SLAM train index: {slam_frame_idx})")

                    # --- 在这里控制可视化和保存 ---
                    # 例如，每 10 个关键帧进行一次可视化并保存
                    show_visualization = (self.windows_right_train % 1 == 0)
                    save_path = None
                    if show_visualization:
                        # 构建保存路径
                        # 你需要一个基础路径，这里我们假设它在 pipeline 配置中
                        # 或者你可以硬编码一个


                        base_output_dir = Path(f"outputs/{self.current_time_str}_vis/")
                        if step == 1 and base_output_dir.exists():
                            os.chmod(base_output_dir, 0o777)
                            shutil.rmtree(base_output_dir)
                        slam_frame_str = f"{slam_frame_idx:06d}"
                        save_path = base_output_dir / f"extend_vis_step_{step}_frame_{slam_frame_str}.png"

                    timers["Mapping_Extend"][0].record()
                    model.extend_map_with_new_frame(
                        new_camera=new_keyframe_cam,
                        img = new_keyframe_data[3],
                        datamanager=self,
                        lidar_cnt=10,#1.21
                        visualize=show_visualization,
                        save_vis_path=save_path, # 传递保存路径
                        index = new_keyframe_data[-1],
                        timers = timers,
                        bigger_fov = True #1.21
                    )
                    timers["Mapping_Extend"][1].record()
                    while( len(self.slam_for_depth) > 0  and new_keyframe_data[0] - self.slam_for_depth[0][0] > 1.0):
                        # print("slam_for_depth")
                        c =  self.slam_for_depth.popleft()
                        new_camera = c[2]
                        depth_timestamp = c[0]

                        if self._ros_enabled:
                            self.publish_depth(model, new_camera, depth_timestamp)
                        else:
                            base_output_dir = Path(f"outputs/{self.current_time_str}_slam_depth/")
                            slam_frame_idx = self.depth_index
                            self.depth_index+=1
                            slam_frame_str = f"{slam_frame_idx:06d}"
                            save_path = base_output_dir / f"{slam_frame_str}.png"
                            model.save_depth(new_camera,save_path)

            if self.windows_right_train < self.config.windows_size and False:
                # self.count = self.windows_right_train * 10
                self.count = self.config.windows_iter  # 100 gslic
                self.ALL_scene = False
                self.ALL_map = False

            elif self.windows_right_train == len(self.slam_train_dataset) - 1:
                if self._ros_enabled:
                    self.count = self.config.windows_iter
                    self.ALL_scene = False
                    self.ALL_map = False
                else:
                    self.ALL_map = True
                    print("self.ALL_map = True")
                    max_key = max(self.scene_it)
                    self.scene_it[max_key+1] = 0
                    self.count = 1

            elif False and self.windows_right_train < len(self.slam_train_dataset) and self.scene_left[self.slam_train_dataset[self.windows_right_train ][2].metadata["scene_index"]]    \
                != self.scene_left[self.slam_train_dataset[self.windows_right_train + 1][2].metadata["scene_index"]]  :

                print("self.ALL_scene = True")
                # self.count = self.windows_right_train * 200
                self.count = 1
                self.ALL_scene = True

            else:
                self.count = self.config.windows_iter # 100 gslic
                self.ALL_scene = False
                self.ALL_map = False
            assert self.count > 0





        window_left = max(0,  self.windows_right_train - self.config.windows_size)
        window_right = min(self.windows_right_train , len(self.slam_train_dataset) - 1)
        s_lf = self.scene_left[self.slam_train_dataset[window_right][2].metadata["scene_index"]]

        if self.ALL_map:

            index = random.randint(0, window_right)
        elif self.ALL_scene:
            # index = random.randint(s_lf, self.windows_right_train)
            # index = self.weighted_random_interpolated(s_lf,  self.windows_right_train, 0.0)
            index = random.randint(s_lf,  window_right)
            # index = random.randint(0, window_right)
        else:
            if ( random.randint(1, 10) <= 2 ) :
                index = random.randint(0, max(0,window_left-1))
            else:
                index = random.randint(window_left, window_right)


        # elif s_lf < window_left and  self.count % self.config.win_all_split == 0:
        #     index = random.randint(s_lf, window_left)
        # else:
        #     index = random.randint( max(window_left,s_lf),  window_right)

        # ct = 0
        # while self.slam_train_dataset[index][2].metadata["cnt"] > 1.4 * ((step + 1) /(self.windows_right_train+1)) :
        #     ct += 1
        #     print("index:",index,"cnt:",self.slam_train_dataset[index][2].metadata["cnt"])
        #     if self.count % self.config.win_all_split == 0 and ct < 300:
        #         index = random.randint(0, window_left)
        #     else:
        #         if ct > 300:
        #             print("ct > 300,train too much past!")
        #         index = random.randint(window_left, self.windows_right_train)
        # self.slam_train_dataset[index][2].metadata["cnt"] += 1
        if len(self.noval_dataset) > 10 and (random.randint(1, 10) <= self.config.noval_rate):
            tmp = random.randint(0, len(self.noval_dataset)-1)
            t, s, sensor_obj, data_dict, s_id, d_r, slam_frame_idx = self.noval_dataset[tmp]
        else:
            self.count -= 1
            t,s, sensor_obj, data_dict,s_id,d_r,slam_frame_idx = self.slam_train_dataset[index]

        if self.ALL_map:
            max_key = max(self.scene_it)
            self.scene_it[max_key] += 1
        if self.ALL_map:
            max_key = max(self.scene_it)
            self.scene_it[max_key] += 1
        else:
            self.scene_it[sensor_obj.metadata["scene_index"]] += 1
        if step % 300 == 0:

            if self.ALL_map:
                print("train ALL_map")
            if self.ALL_scene:
                print("train ALL_scene")

            print("step:", step, "SLAM Progress:", self.windows_right_train, "/", len(self.slam_train_dataset) )
            print("scene_index:",sensor_obj.metadata["scene_index"],"scene_left_slam_index:", self.slam_train_dataset[s_lf][-1] , \
                  "scene_right_slam_index:",self.slam_train_dataset[self.windows_right_train][-1])
        self.delta_angle = d_r

        self.step_for_schedule += 1

        if self.windows_right_train == len(self.slam_train_dataset) -1 and  self.count == 0:
            self.finished = True

        # 1. 总时间结束


        # --- [NEW] 统计并区分 Lidar/Camera ---
        torch.cuda.synchronize()  # 确保时间准确

        # 确定返回数据的类型 (用于 Data_Selection 和 Total)
        returned_type_str = "Camera" if isinstance(sensor_obj, Cameras) else "Lidar"

        # 辅助函数：安全记录时间
        def safe_record(key_name, type_suffix, timer_pair):
            try:
                # elapsed_time 仅在两次 record 都被调用后有效
                # 我们可以简单通过判断 elapsed_time > 0 或者依靠 try-catch
                # 但为了严谨，如果你确信某些路径没跑 record，这里的值可能是无意义的
                # 由于我们上面逻辑覆盖了主要路径，这里直接取值
                duration = timer_pair[0].elapsed_time(timer_pair[1])

                # 如果 duration 极小且不合理（比如没 record），在某些 pytorch 版本会报错或返回 0
                # 这里假设 record 只要调用了就是有效的
                full_key = f"{key_name}_{type_suffix}"
                self.slam_timing_stats[full_key]["sum"] += duration
                self.slam_timing_stats[full_key]["count"] += 1
            except RuntimeError:
                pass  # 事件未记录

        # 1. 记录 Mapping (仅当 mapping_type_str 存在时)
        if mapping_type_str is not None:
            safe_record("Mapping_Extend", mapping_type_str, timers["Mapping_Extend"])
            safe_record("Mapping_Extend1", mapping_type_str, timers["Mapping_Extend1"])
            safe_record("Mapping_Extend2", mapping_type_str, timers["Mapping_Extend2"])
            safe_record("Mapping_Extend3", mapping_type_str, timers["Mapping_Extend3"])
            safe_record("Mapping_Extend4", mapping_type_str, timers["Mapping_Extend4"])
            # Diffix 肯定是 Camera
            if mapping_type_str == "Camera":
                safe_record("Diffix_Gen", "Camera", timers["Diffix_Gen"])

        # 2. 记录 Selection 和 Total (基于返回的数据类型)

        # --- 定期写入日志 ---
        if step % 2000 == 0:
            for key, stats in self.slam_timing_stats.items():
                if stats["count"] > 0:
                    avg_time = stats["sum"] / stats["count"]
                    writer.put_scalar(f"SLAM_Timing/{key}_avg_ms", avg_time, step)
                    stats["sum"] = 0.0
                    stats["count"] = 0
        return sensor_obj, data_dict


    # def odom(self,interpolated_pose_4x4_tensor,camera_obj_single,camera_obj_next,data_dict,data_next):
    #
    #
    #
    #     if self.init_pose_MACVO == None:
    #         self.init_pose_MACVO = interpolated_pose_4x4_tensor
    #
    #     delta_pose = torch.inverse(self.init_pose_MACVO.cpu()) @ interpolated_pose_4x4_tensor.cpu()
    #
    #     macvo_delta_pose = delta_pose
    #
    #
    #
    #     pose_3x4 = macvo_delta_pose[:3, :4].cpu()  # 形状变为 [3, 4]
    #
    #     q = quaternion.from_rotation_matrix(pose_3x4[:3, :3])
    #     liepose = torch.tensor([pose_3x4[0, 3], pose_3x4[1, 3], pose_3x4[2, 3], q.x, q.y, q.z, q.w])
    #     K = torch.from_numpy(camera_obj_single.get_intrinsics_matrices().cpu().numpy())
    #
    #     T_BS_lcam = np.array([
    #         [1.0, 0.00000, 0.00000, 0.00000],
    #         [0.00000, 1.0, 0.00000, 0.00000],
    #         [0.00000, 0.00000, 1.0, 0.00000],
    #         [0.00000, 0.00000, 0.00000, 1.0]
    #     ]).reshape(4, 4)  # body frame -> camera frame
    #
    #     T_BS = pp.from_matrix(
    #         torch.tensor(T_BS_lcam, dtype=torch.float32).unsqueeze(0), pp.SE3_type
    #     ) @ NED2EDN.unsqueeze(0)  # body frame -> NED frame
    #
    #     l_data = BTMonocularDataset(
    #         camera_obj_single.get_intrinsics_matrices().cpu().numpy()[0],
    #         camera_obj_single.distortion_params.cpu().numpy()[0, :5], camera_obj_single.metadata["lidar2lcam"],
    #         camera_obj_single.width.item(),
    #         camera_obj_single.height.item())
    #     r_data = BTMonocularDataset(
    #         camera_obj_next.get_intrinsics_matrices().cpu().numpy()[0],
    #         camera_obj_next.distortion_params.cpu().numpy()[0, :5], camera_obj_next.metadata["lidar2rcam"],
    #         camera_obj_next.width.item(),
    #         camera_obj_next.height.item())
    #
    #     rectified_K = sync_LR(l_data, r_data)
    #     K = torch.tensor(rectified_K[:3, :3], dtype=torch.float).unsqueeze(0)
    #     new_width = camera_obj_single.width.item()
    #
    #     if self.config.noval_eval_width != -1:
    #         K[0, 0, 2] = self.config.noval_eval_width / new_width * K[0, 0, 2]
    #         new_width = self.config.noval_eval_width
    #
    #     a = pad_image_to_size(l_data.undistort(data_dict['image'].cpu().numpy()),
    #                       camera_obj_single.height.item(), new_width) if data_dict['image'].shape[
    #                                                                          1] < new_width else l_data.undistort(
    #         data_dict['image'].cpu().numpy())
    #     new_stereo_frame = StereoFrame(
    #         idx=[self.camera_index],
    #         stereo=StereoData(
    #             T_BS=T_BS,
    #             K=K,
    #             baseline=torch.tensor([0.25374], device=self.device),  # !!!!!!记得改
    #             time_ns=[int(camera_obj_single.times.item() * 1e9)],
    #             height=camera_obj_single.height.item(),
    #             width=new_width,
    #             imageL=pad_image_to_size(l_data.undistort(data_dict['image'].cpu().numpy()),
    #                                      camera_obj_single.height.item(), new_width) if data_dict['image'].shape[1] <  new_width else l_data.undistort(data_dict['image'].cpu().numpy()),  # HW3->
    #             imageR=pad_image_to_size(r_data.undistort(data_next['image'].cpu().numpy()),
    #                                      camera_obj_single.height.item(), new_width) if data_next['image'].shape[1] <  new_width else r_data.undistort(data_next['image'].cpu().numpy()),  # 假设dataparser会加载右图
    #
    #             gt_depth=None,
    #             gt_flow=None,
    #             flow_mask=None,
    #         ),
    #         time_ns=[int(camera_obj_single.times.item() * 1e9)],
    #         gt_pose=cast(pp.LieTensor, pp.SE3(liepose)) if (
    #                 liepose is not None) else None,
    #     )
    #     self.camera_index += 1
    #     return new_stereo_frame,delta_pose
    def get_sorted_train_stream(self) -> Tuple[
        List[Tuple[float, str, Union[Cameras, Lidars], Dict, int, float, int]], List[
            Tuple[float, str, Union[Cameras, Lidars], Dict, int, float, int]]]:
        """
        读取训练用的相机和激光雷达数据，将它们合并，并返回一个按时间戳排序的列表。
        同时计算每一帧和上一帧的角度变换（弧度制），并将其添加到列表中。

        Args:
            train_cameras (Cameras): 包含所有训练相机帧的对象。
            cached_cam_data (List[Dict]): 与相机帧对应的缓存数据（如图像）。
            train_lidars (Lidars): 包含所有训练激光雷达扫描的对象。
            cached_lidar_data (List[Dict]): 与激光雷达扫描对应的缓存数据。
            device (str): 计算设备。

        Returns:
            一个按时间顺序排列好的列表。
            列表中的每个元素都是一个元组：(timestamp, sensor_type, sensor_object, metadata_dict, sensor_idx, delta_angle, original_index)。
        """
        from scipy.spatial.transform import Rotation as R

        def get_rotation_matrix_from_sensor_obj( sensor_obj: Union[Cameras, Lidars]) -> np.ndarray:
            """从 Cameras 或 Lidars 对象中提取 3x3 旋转矩阵，并返回 NumPy 数组。"""
            if isinstance(sensor_obj, Cameras):
                # Cameras 的 camera_to_worlds 是 [N, 3, 4] 或 [3, 4]
                # 我们需要提取旋转部分 [3, 3]
                pose = sensor_obj.camera_to_worlds.squeeze(0)  # 移除 batch 维度
                return pose[:3, :3].cpu().numpy()  # 转换为 NumPy 数组
            elif isinstance(sensor_obj, Lidars):
                # Lidars 的 lidar_to_worlds 是 [N, 3, 4] 或 [3, 4]
                # 我们需要提取旋转部分 [3, 3]
                pose = sensor_obj.lidar_to_worlds.squeeze(0)  # 移除 batch 维度
                return pose[:3, :3].cpu().numpy()  # 转换为 NumPy 数组
            else:
                raise TypeError("Unsupported sensor object type for rotation matrix extraction.")

        print("\n正在合并与排序传感器数据流并计算角度变换...")
        combined_stream = []

        # 1. 调用数据加载器获取并处理所有相机数据
        camera_data_list = self.fixed_indices_train_dataloader
        for camera_obj, data_dict in camera_data_list:
            timestamp = camera_obj.times.item()
            sensor_type = "camera"
            sensor_idx = camera_obj.metadata["sensor_idxs"].item()
            combined_stream.append((timestamp, sensor_type, camera_obj, data_dict, sensor_idx))  # 初始角度差为 0

        # 2. 调用数据加载器获取并处理所有激光雷达数据
        lidar_data_list = self.fixed_indices_train_lidar_dataloader
        for lidar_obj, data_dict in lidar_data_list:
            timestamp = lidar_obj.times.item()
            sensor_type = "lidar"
            sensor_idx = lidar_obj.metadata["sensor_idxs"].item()
            combined_stream.append((timestamp, sensor_type, lidar_obj, data_dict, sensor_idx))  # 初始角度差为 0

        # 3. 根据时间戳 (元组的第一个元素) 和 sensor_idx (保证相同时间戳下传感器顺序稳定) 对合并后的列表进行排序
        combined_stream.sort(key=lambda item: (item[0], item[4]))



        # 4. 计算并更新角度变换

        new_combined_stream = []


        for i, item in enumerate(combined_stream):
            timestamp, sensor_type, sensor_obj, data_dict, sensor_idx = item

            # 从传感器对象中提取旋转矩阵，并转换为 SciPy 的 Rotation 对象
            current_rotation_matrix_np = get_rotation_matrix_from_sensor_obj(sensor_obj)
            current_rotation_obj = R.from_matrix(current_rotation_matrix_np)
            previous_rotation_obj: Optional[R] = None

            for j in range(i-1,-1,-1):
            # for j in range(i , 0):
                if abs(item[0] - combined_stream[j][0]) >0.05 and item[4] == combined_stream[j][4]:
                    previous_rotation_obj =  R.from_matrix(get_rotation_matrix_from_sensor_obj(combined_stream[j][2]))
                    break

            if previous_rotation_obj is not None:
                # 计算从上一帧到当前帧的相对旋转
                # R_delta = R_prev.inv() * R_current
                R_delta_obj = previous_rotation_obj.inv() * current_rotation_obj

                # 获取旋转的轴角表示，其中 angle 是旋转角度（弧度）
                # `magnitude` 返回旋转的欧几里得范数（弧度）
                delta_angle = np.degrees(R_delta_obj.magnitude())  # 获取旋转角度的绝对值（弧度）
            else:
                delta_angle = 0.0

            new_combined_stream.append([timestamp, sensor_type, sensor_obj, data_dict, sensor_idx, delta_angle, i])

        if False:
            indices = [i for i, item in enumerate(new_combined_stream)]
            delta_angles = [item[-2] for item in new_combined_stream]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(indices, delta_angles, label='帧间角度变换 (Delta Angle)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            plt.show()
        combined_stream = new_combined_stream

        scene_index = 0
        self.scene_it[scene_index] = 0

        #test
        # a = [i[-2]>self.config.thre_angle for i in combined_stream]
        #test

        #计算 scene id
        last = None
        for i, item in enumerate(combined_stream):
            timestamp, sensor_type, sensor_obj, data_dict,sensor_idx,delta_angle,slam_index = item

            if last == None:
                last = sensor_obj

            if isinstance(item[2], Cameras):
                pose = item[2].camera_to_worlds[:, :, 3].cpu().numpy()
            elif isinstance(item[2], Lidars):
                pose = item[2].lidar_to_worlds[:, :, 3].cpu().numpy()

            if isinstance(last, Cameras):
                last_pose = last.camera_to_worlds[:, :, 3].cpu().numpy()
            elif isinstance(last, Lidars):
                last_pose = last.lidar_to_worlds[:, :, 3].cpu().numpy()

            if np.linalg.norm(pose - last_pose) > 40.0:
                print("straight scene")
                scene_index += 1
                sensor_obj.metadata["scene_index"] = scene_index
                self.scene_it[scene_index] = 0
                last = sensor_obj

            elif delta_angle > self.config.thre_angle_scene and np.linalg.norm(pose - last_pose) > 20.0:

                sensor_obj.metadata["scene_index"] = -1

                for j in range(i-1,max(i-20,-1),-1):
                    if combined_stream[j][5] > self.config.thre_angle_scene :
                        sensor_obj.metadata["scene_index"] = combined_stream[j][2].metadata["scene_index"]
                        break

                if sensor_obj.metadata["scene_index"] == -1:
                    for j in range(i + 1, min(i + 20, len(combined_stream))):
                        if combined_stream[j][5] > self.config.thre_angle_scene:
                            print("wang scene")
                            scene_index += 1
                            sensor_obj.metadata["scene_index"] = scene_index
                            last = sensor_obj
                            self.scene_it[scene_index] = 0
                            break
                    if sensor_obj.metadata["scene_index"] == -1:
                        sensor_obj.metadata["scene_index"] = scene_index #孤立值
            else:
                sensor_obj.metadata["scene_index"] = scene_index

        # 根据配置筛选训练数据流
        train_cam_count = len(self.dataparser.config.cameras)
        has_eval_cameras = len(getattr(self.dataparser.config, 'eval_cameras', ())) > 0

        if has_eval_cameras:
            combined_train_stream = [item for item in combined_stream if item[1] == 'camera' and item[4] < train_cam_count]
            eval_stream = [item for item in combined_stream if item[1] == 'camera' and item[4] >= train_cam_count]
            print(f"eval_cameras enabled: train cameras={train_cam_count}, eval frames={len(eval_stream)}")
        else:
            combined_train_stream = [item for item in combined_stream if item[1] == 'camera']
            eval_stream = []

        train_stream = []

        angle_cnt = 0
        use_linear_keyframe_select = False
        #关键帧
        for i, item in enumerate(combined_train_stream):

            # eval_stream.append(item)
            if use_linear_keyframe_select:
                if i % self.config.select_every_k_frame == 0:
                    train_stream.append(item)
                continue


            if len(train_stream) == 0:
                train_stream.append(item)
                continue

            if isinstance(item[2], Cameras):
                pose = item[2].camera_to_worlds[:, :, 3].cpu().numpy()
            elif isinstance(item[2], Lidars):
                pose = item[2].lidar_to_worlds[:, :, 3].cpu().numpy()

            if isinstance(train_stream[-1][2], Cameras):
                last_pose = train_stream[-1][2].camera_to_worlds[:, :, 3].cpu().numpy()
            elif isinstance(train_stream[-1][2], Lidars):
                last_pose = train_stream[-1][2].lidar_to_worlds[:, :, 3].cpu().numpy()

            if np.linalg.norm(pose - last_pose) > self.config.thre_dis:
                print("find trans:", int(item[-1]))
                train_stream.append(item)
                continue

            elif item[5] > self.config.thre_angle :
                angle_cnt += 1
                if  angle_cnt % 3 == 0:
                    print("find big rot:",int(item[-1]) )
                    train_stream.append(item)
                    continue

            if not has_eval_cameras:
                eval_stream.append(item)



        print("train camera stream size:", len(train_stream),"combined_train_stream size:", len(combined_train_stream))

        app = []
        lidar_fre = 1 #1 hz
        if not(self.config.slam_cameara_only):

            lidar_stream = [item for item in combined_stream if item[1] == 'lidar']
            step = max(1, int(10 / lidar_fre))  # ensure integer and avoid 0
            for i in range(0, len(lidar_stream), step):
                app.append(lidar_stream[i])

            print("lidar num:",len(app))
            train_stream.extend(app)
            train_stream.sort(key=lambda item: (item[0], item[4]))
        else:
            print("slam_cameara_only")
        for i, item in enumerate(train_stream):
            if i == 0 or item[2].metadata["scene_index"] != train_stream[i-1][2].metadata["scene_index"]:
                print("scene_left:",train_stream[max(0,i-1)][-1])
                self.scene_left[item[2].metadata["scene_index"]] = i



        print("排序和角度变换计算完成。")

        for it in combined_stream:
            # print(it[0],it[1],it[-1])
            pass

        queue = deque()
        for i, item in enumerate(combined_train_stream):
            queue.append(item)

        return combined_stream, train_stream, eval_stream,queue


    def weighted_random_interpolated(self,start, end, ratio=0.2):
        """
        在一个范围内生成一个随机整数，数值越大概率呈线性关系越大。

        可以通过 ratio 参数控制最大概率与最小概率的比率。

        参数:
        start (int): 范围的起始值 (包含)。
        end (int): 范围的结束值 (包含)。
        ratio (float): 最高概率比最低概率高出的部分。例如，0.2 表示最高概率是最低的1.2倍。

        返回:
        int: 根据线性插值权重随机选择的整数。
        """
        # 处理边界情况
        if start >= end:
            return end
        if ratio < 0:
            raise ValueError("Ratio 不能为负数。")

        # 创建数值范围
        population = range(start, end + 1)
        n = len(population)

        # 如果只有一个元素，直接返回
        if n == 1:
            return start

        # --- 核心逻辑：线性插值计算权重 ---
        # 总增量为 ratio，共有 n-1 个间隔
        step = ratio / (n - 1)
        # 使用列表推导式高效生成权重列表
        weights = [1 + i * step for i in range(n)]

        # random.choices 返回一个列表，我们只取一个值 (k=1)，所以获取第一个元素
        return random.choices(population, weights=weights, k=1)[0]

    def _find_bracketing_lidars(self, target_index: int) :
        """
        在给定的窗口内，为目标索引查找时间上前后相邻的LIDAR数据。
        """
        lidar1_data, lidar2_data = None, None

        # 向后搜索，找到前一个LIDAR
        for i in range(target_index, 0, -1):
            if self.slam_dataset[i][1] == 'lidar':
                lidar1_data = self.slam_dataset[i]
                break

        # 向前搜索，找到后一个LIDAR
        for i in range(target_index, len(self.slam_dataset)):
            if self.slam_dataset[i][1] == 'lidar':
                lidar2_data = self.slam_dataset[i]
                break

        return lidar1_data, lidar2_data

    def _interpolate_pose(self, camera_timestamp: float, lidar1_data: Tuple, lidar2_data: Tuple) -> torch.Tensor:
        """
        在两个LIDAR位姿之间进行球形插值(SLERP)。
        """
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        # 1. 提取时间戳和位姿
        t1, _, lidar1_obj, _,_,_,_ = lidar1_data
        t2, _, lidar2_obj, _,_,_,_ = lidar2_data

        # 将 PyTorch 张量位姿转换为 NumPy 4x4 矩阵
        pose1_3x4 = lidar1_obj.lidar_to_worlds.squeeze(0).cpu().numpy()
        pose2_3x4 = lidar2_obj.lidar_to_worlds.squeeze(0).cpu().numpy()

        pose1_4x4 = np.vstack([pose1_3x4, [0, 0, 0, 1]])
        pose2_4x4 = np.vstack([pose2_3x4, [0, 0, 0, 1]])

        # 2. 计算插值因子 (t)
        time_interval = t2 - t1
        if time_interval < 1e-6:  # 避免除以零
            interpolation_factor = 0.0
        else:
            interpolation_factor = (camera_timestamp - t1) / time_interval

        # 确保因子在 [0, 1] 范围内
        interpolation_factor = np.clip(interpolation_factor, 0.0, 1.0)

        # 3. 对平移部分进行线性插值 (LERP)
        trans1 = pose1_4x4[:3, 3]
        trans2 = pose2_4x4[:3, 3]
        interp_trans = (1 - interpolation_factor) * trans1 + interpolation_factor * trans2

        # 4. 对旋转部分进行球形插值 (SLERP)
        rotations = R.from_matrix([pose1_4x4[:3, :3], pose2_4x4[:3, :3]])
        slerp = Slerp([t1, t2], rotations)
        interp_rotation = slerp([camera_timestamp]).as_matrix()[0]

        # 5. 组合成新的 4x4 位姿矩阵
        new_pose_4x4 = np.identity(4)
        new_pose_4x4[:3, :3] = interp_rotation
        new_pose_4x4[:3, 3] = interp_trans

        # 6. 转换回 PyTorch 张量 [1, 3, 4] 格式
        new_pose_tensor = torch.from_numpy(new_pose_4x4[:3, :]).float().unsqueeze(0)

        return new_pose_tensor





    def next_eval(self, step: int) -> Tuple[Union[Cameras, Lidars], Dict]:
        """Returns the next evaluation batch

        Returns a Camera or Lidar instead of raybundle"""
        # repopulate unseen cameras and lidars if they are empty
        if (len(self.eval_unseen_cameras) + len(self.eval_unseen_lidars)) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
            self.eval_unseen_lidars = [i for i in range(len(self.eval_lidar_dataset))]

        if random.randint(0, len(self.eval_unseen_cameras) + len(self.eval_unseen_lidars) - 1) < len(
            self.eval_unseen_cameras
        ):
            return self.next_eval_image(step)
        else:
            return self.next_eval_lidar(step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval_lidar(self, step: int) -> Tuple[Lidars, Dict]:
        """Returns the next evaluation batch

        Returns a Lidar instead of raybundle"""
        if len(self.eval_unseen_lidars) == 0:
            self.eval_unseen_lidars = [i for i in range(len(self.eval_lidar_dataset))]
        lidar_idx = self.eval_unseen_lidars.pop(random.randint(0, len(self.eval_unseen_lidars) - 1))
        assert len(self.eval_lidar_dataset.lidars.shape) == 1, "Assumes single batch dimension"
        lidar = self.eval_lidar_dataset.lidars[lidar_idx : lidar_idx + 1].to(self.device)
        if lidar.metadata is None:
            lidar.metadata = {}
        lidar.metadata["lidar_idx"] = lidar_idx

        data = self.cached_lidar_eval[lidar_idx]
        data = data.copy()

        self._add_metadata(lidar, data, len(self.eval_dataset))

        return lidar, data

    def paint_points(self):
        cameras = self.train_dataset.cameras
        lidars = self.train_lidar_dataset.lidars
        image_cache = self.cached_train
        lidar_cache = self.cached_lidar_train
        point_clouds_rgb = []
        topk = len(cameras.metadata["sensor_idxs"].unique()) * self.config.paint_points_topk
        topk = min(topk, len(cameras))

        for lidar_i, lidar_data in enumerate(lidar_cache[:10]):
            pc = lidar_data["lidar"].to("cpu")
            lidar = lidars[lidar_i]
            pc_in_world = transform_points(pc[:, :3], lidar.lidar_to_worlds)
            # point_cloud_rgb = torch.rand_like(pc[:, :3]) * 255
            point_cloud_rgb = torch.zeros_like(pc[:, :3])
            lidar_time = lidar.times.squeeze(-1)
            top_k_cam_idx = torch.topk(
                (cameras.times - lidar_time).abs().squeeze(),
                topk,
                largest=False,
            ).indices
            for cam_idx in top_k_cam_idx.flip(0)[::3]:
                camera = cameras[cam_idx]
                pc_in_camera = transform_points(pc_in_world, inverse(camera.camera_to_worlds.squeeze(0)))
                # Flip the y and z axis because of nerfstudio conventions
                pc_in_camera[:, 1] = -pc_in_camera[:, 1]
                pc_in_camera[:, 2] = -pc_in_camera[:, 2]
                # Only paint points in front of the camera
                valid_points = pc_in_camera[:, 2] > 0.3
                # Normalize the points
                pc_in_camera = pc_in_camera / pc_in_camera[:, 2:]

                intrinsics = camera.get_intrinsics_matrices().squeeze(0)
                pc_in_image = (torch.matmul(intrinsics, pc_in_camera[:, :3].T).T).to(torch.int64)

                # Only paint points that are within the image
                valid_points = (
                    valid_points
                    & (pc_in_image[:, 0] >= 0)
                    & (pc_in_image[:, 0] < camera.width)
                    & (pc_in_image[:, 1] >= 0)
                    & (pc_in_image[:, 1] < camera.height)
                )

                image = image_cache[cam_idx]["image"].to("cpu")
                point_cloud_rgb[valid_points] = (
                    image[pc_in_image[valid_points, 1], pc_in_image[valid_points, 0]]
                ).float()
            point_clouds_rgb.append(point_cloud_rgb)

        self.train_dataparser_outputs.metadata["point_clouds_rgb"] = point_clouds_rgb

    def get_num_train_data(self) -> int:
        return len(self.train_dataset) + len(self.train_lidar_dataset)

    def save_pose(self, step: Optional[int] = None,
                  output_path: Optional[Path] = None):

        print("Running eval_noval...")
        # print(self.datamanager.eval_dataset.cameras.camera_to_worlds.shape)
        # eval_poses = self.camera_to_worlds_to_numpy(self.datamanager.eval_dataset.cameras.camera_to_worlds)
        # eval_img = [d.copy() for d in self.datamanager.eval_dataset.cached_eval]

        # train_poses = self.camera_to_worlds_to_numpy(self.datamanager.train_dataset.cameras.camera_to_worlds)

        dataloader_output = self.slam_dataset  # 注意这里要加括号()来调用函数

        camera_items = [item for item in dataloader_output if item[1] == 'camera']

        all_camera_objects = [item[2] for item in camera_items]
        camera_to_worlds_tensors = [cam.camera_to_worlds for cam in all_camera_objects]
        sensor_idxs_tensors = [cam.metadata['sensor_idxs'] for cam in all_camera_objects]
        times_tensors = [cam.times for cam in all_camera_objects]

        # 将列表中的张量合并成一个大的张量，方便进行批处理
        combined_camera_to_worlds = torch.cat(camera_to_worlds_tensors, dim=0)
        combined_sensor_idxs = torch.cat(sensor_idxs_tensors, dim=0)
        combined_times = torch.cat(times_tensors, dim=0)

        T_rgb0_vlp16 = camera_items[0][2].metadata['lidar2lcam']

        # 6. 保存筛选并处理后的位姿和对应的时间戳
        assert output_path is not None
        # 创建一个布尔掩码，只选择 sensor_id 为 0 的数据
        mask = (combined_sensor_idxs == 0)

        # 应用掩码来筛选出符合条件的位姿和时间戳
        filtered_c2w = combined_camera_to_worlds[mask.squeeze()]
        filtered_times = combined_times[mask.squeeze()]

        # 将筛选后的位姿张量转换为 NumPy 数组以便处理
        novel_poses = self.camera_to_worlds_to_numpy(filtered_c2w)
        # 将时间戳转换为 NumPy 数组
        times_to_save = filtered_times.flatten().cpu().numpy()

        # 确保位姿数量和时间戳数量完全匹配
        if len(times_to_save) == len(novel_poses):
            # 定义带有时间戳的位姿文件路径
            pose_file_path_with_time = output_path / f"xyzijkw_{step}.txt"
            with open(pose_file_path_with_time, "w") as f:
                pose_file_path_no_time = output_path / f"kitti_poses{step}.txt"
                with open(pose_file_path_no_time, "w") as ff:
                    f.write("# format: x y z qx qy qz qw\n")
                    for i, pose_4x4 in enumerate(novel_poses):
                        OPENCV_TO_NERFSTUDIO = np.array(
                            [
                                [1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1],
                            ]
                        )
                        opencv_to_nerfstudio_4x4 = np.eye(4, dtype=np.float32)
                        opencv_to_nerfstudio_4x4[:3, :3] = OPENCV_TO_NERFSTUDIO
                        pose_4x4 = T_rgb0_vlp16 @ (
                                pose_4x4 @ opencv_to_nerfstudio_4x4 @ T_rgb0_vlp16) @ np.linalg.inv(
                            T_rgb0_vlp16)

                        translation = pose_4x4[:3, 3]
                        rotation_matrix = pose_4x4[:3, :3]

                        try:
                            r = Rotation.from_matrix(rotation_matrix)
                            quat = r.as_quat()  # (x, y, z, w)
                        except Exception as e:
                            print(f"警告: 无法为位姿 {i} 转换旋转矩阵: {e}。将使用单位四元数。")
                            quat = [0.0, 0.0, 0.0, 1.0]

                        time = times_to_save[i]
                        # 写入文件，格式为: time tx ty tz qx qy qz qw
                        f.write(
                            f"{translation[0]} {translation[1]} {translation[2]} "
                            f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}\n"
                        )

                        pose_3x4 = pose_4x4[:3, :]

                        pose_elements = pose_3x4.flatten()

                        # 3. 将所有浮点数元素转换为字符串
                        pose_str_elements = [str(elem) for elem in pose_elements]

                        # 4. 用空格连接所有字符串元素，形成单行文本
                        line = " ".join(pose_str_elements)

                        # 5. 写入文件并添加换行符
                        ff.write(line + "\n")


        else:
            print(f"错误：最终位姿数量 ({len(novel_poses)}) 与时间戳数量 ({len(times_to_save)}) 不匹配。已跳过保存。")

    def apply_z_rotation_and_translation(self, poses: np.ndarray,
                                         degrees: float,
                                         translation_vector: list or tuple) -> np.ndarray:
        """
        对一组 4x4 的位姿矩阵应用一个绕Z轴的旋转和一个平移变换。

        Args:
            poses (np.ndarray): 输入的位姿矩阵，形状为 [N, 4, 4]。
            degrees (float): 绕Z轴旋转的角度（单位：度）。正值表示逆时针旋转。
            translation_vector (list or tuple): 一个包含 [tx, ty, tz] 的平移向量。

        Returns:
            np.ndarray: 施加变换后的新位姿矩阵，形状为 [N, 4, 4]。
        """
        # 1. 将角度从度转换为弧度，因为三角函数使用弧度
        radians = np.radians(degrees)

        # 2. 创建一个 4x4 的齐次变换矩阵，用于绕Z轴旋转和平移
        #    从一个单位矩阵开始
        transform_matrix = np.identity(4)

        # 计算旋转分量
        cos_r = np.cos(radians)
        sin_r = np.sin(radians)

        # # 填充旋转部分 (左上角 3x3)
        # transform_matrix[0, 0] = cos_r
        # transform_matrix[0, 1] = -sin_r
        # transform_matrix[1, 0] = sin_r
        # transform_matrix[1, 1] = cos_r

        transform_matrix[0, 0] = cos_r
        transform_matrix[0, 2] = sin_r
        transform_matrix[2, 0] = -sin_r
        transform_matrix[2, 2] = cos_r

        # 填充平移部分 (最后一列的前三行)
        transform_matrix[:3, 3] = translation_vector

        # 3. 将这个变换矩阵右乘到所有的位姿上
        #    NumPy的 @ 运算符可以处理广播，它会自动将 [N, 4, 4] @ [4, 4]
        #    正确地计算为 [N, 4, 4]
        #    P' = P @ T
        transformed_poses = poses @ transform_matrix

        return transformed_poses

    def apply_bev(self, poses: np.ndarray,
                                         degrees: float,
                                         translation_vector: list or tuple) -> np.ndarray:
        """
        对一组 4x4 的位姿矩阵应用一个绕Z轴的旋转和一个平移变换。

        Args:
            poses (np.ndarray): 输入的位姿矩阵，形状为 [N, 4, 4]。
            degrees (float): 绕Z轴旋转的角度（单位：度）。正值表示逆时针旋转。
            translation_vector (list or tuple): 一个包含 [tx, ty, tz] 的平移向量。

        Returns:
            np.ndarray: 施加变换后的新位姿矩阵，形状为 [N, 4, 4]。
        """
        # 1. 将角度从度转换为弧度，因为三角函数使用弧度
        radians = np.radians(degrees)

        # 2. 创建一个 4x4 的齐次变换矩阵，用于绕Z轴旋转和平移
        #    从一个单位矩阵开始
        transform_matrix = np.identity(4)

        # 计算旋转分量
        cos_r = np.cos(radians)
        sin_r = np.sin(radians)

        # # 填充旋转部分 (左上角 3x3)
        # transform_matrix[0, 0] = cos_r
        # transform_matrix[0, 1] = -sin_r
        # transform_matrix[1, 0] = sin_r
        # transform_matrix[1, 1] = cos_r

        transform_matrix[1, 1] = cos_r
        transform_matrix[1, 2] = -sin_r
        transform_matrix[2, 1] = sin_r
        transform_matrix[2, 2] = cos_r

        # 填充平移部分 (最后一列的前三行)
        transform_matrix[:3, 3] = translation_vector

        # 3. 将这个变换矩阵右乘到所有的位姿上
        #    NumPy的 @ 运算符可以处理广播，它会自动将 [N, 4, 4] @ [4, 4]
        #    正确地计算为 [N, 4, 4]
        #    P' = P @ T
        transformed_poses = poses @ transform_matrix

        return transformed_poses

    def camera_to_worlds_to_numpy(self, camera_tensor: torch.Tensor) -> np.ndarray:
        """
        将 [N, 3, 4] 的相机到世界矩阵转换为 [N, 4, 4] 的齐次坐标形式

        参数:
            camera_tensor: Float[Tensor, "*num_cameras 3 4"]

        返回:
            np.ndarray: (num_images, 4, 4) 的齐次变换矩阵
        """
        num_cameras = camera_tensor.shape[0]

        # 创建齐次坐标的最后一行 [0, 0, 0, 1]
        last_row = torch.tensor([0, 0, 0, 1],
                                dtype=camera_tensor.dtype,
                                device=camera_tensor.device).expand(num_cameras, 1, 4)

        # 拼接成 [N, 4, 4]
        c = deepcopy(camera_tensor).to(camera_tensor.device)
        homogeneous = torch.cat([camera_tensor, last_row], dim=1)

        # 转换为NumPy数组
        return homogeneous.cpu().numpy()
def sync_LR(left: BTMonocularDataset, right: BTMonocularDataset) -> np.ndarray:
    # Constant - Transformation from cam 1 to cam 2 (L -> R)
    T_LR = np.linalg.inv(right.T_BS) @ left.T_BS

    # --- 解决方案：确保所有矩阵参数的数据类型一致 ---
    # 将所有输入转换为 float64
    K1 = left.K.astype(np.float64)
    D1 = left.distort_factor.astype(np.float64)
    K2 = right.K.astype(np.float64)
    D2 = right.distort_factor.astype(np.float64)
    R = T_LR[:3, :3].astype(np.float64)
    T = T_LR[:3, 3].astype(np.float64)
    image_size = (left.width, left.height)

    # Rectify stereo and undistort based on Left and Right camera.
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(K1, D1, K2, D2,
                                                                image_size,
                                                                R, T,
                                                                flags=cv2.CALIB_ZERO_DISPARITY,
                                                                alpha=-1)

    left.undistort_map = cv2.initUndistortRectifyMap(K1, D1 , R1, P1, (left.width, left.height),
                                                     cv2.CV_32FC1)
    right.undistort_map = cv2.initUndistortRectifyMap(K2, D2, R2, P2,
                                                      (left.width, left.height), cv2.CV_32FC1)
    left.K = P1[:3, :3]
    right.K = P2[:3, :3]

    return P1

class BTMonocularDataset():
    """
    Return images in the given directory ends with .png
    """

    def __init__(self, K: np.ndarray, undistort: np.ndarray, T_BS: np.ndarray, width, height) -> None:

        self.K = K
        self.T_BS = T_BS
        self.distort_factor = undistort
        self.undistort_map: None | tuple[np.ndarray, np.ndarray] = None
        self.width = width
        self.height = height

    def correct_distortion(self, image: np.ndarray) -> np.ndarray:

        h, w = image.shape[:2]
        if self.undistort_map is None:
            raise Exception("Monocular sequence is not rectified.")
        else:
            undistorted_image = cv2.remap(image, self.undistort_map[0], self.undistort_map[1], cv2.INTER_LINEAR)
        return undistorted_image

    def undistort(self, image: np.ndarray) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        result = self.correct_distortion(image)

        result = torch.tensor(result, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        result /= 255.
        return result


def pad_image_to_size(image: torch.Tensor,
                      target_h: int,
                      target_w: int
                      ) -> torch.Tensor:
    """
    将图像张量填充到指定尺寸，原始图像位于中心，四周用0填充。

    参数:
    - image (torch.Tensor): 输入的图像张量，形状为 BxCxHxW。
    - target_h (int): 目标高度。
    - target_w (int): 目标宽度。

    返回:
    - torch.Tensor: 填充后的图像张量，形状为 BxCx(target_h)x(target_w)。
    """
    # 检查输入张量的维度是否正确
    if image.dim() != 4:
        raise ValueError(f"输入张量需要是4维 (BxCxHxW)，但当前维度为 {image.dim()}")

    b,c,h, w   = image.shape

    # 确保目标尺寸不小于原始图像尺寸
    if target_h < h or target_w < w:
        raise ValueError(f"目标尺寸 ({target_h}x{target_w}) 不能小于原始图像尺寸 ({h}x{w})")

    # 1. 创建一个目标尺寸的全零画布
    padded_image = torch.full(( b,c,target_h, target_w ),
                              0.99,
                               dtype=image.dtype,
                               device=image.device)

    # 2. 计算将原始图像放置在中心的起始坐标
    # (目标尺寸 - 原始尺寸) / 2
    start_h = (target_h - h) // 2
    start_w = (target_w - w) // 2

    # 3. 将原始图像复制到画布的中心区域
    padded_image[:,:,start_h:start_h + h, start_w:start_w + w] = image
    image = padded_image

    return image
