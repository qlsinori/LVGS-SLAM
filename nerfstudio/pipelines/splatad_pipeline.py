# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Abstracts for the Pipeline class.
"""

from __future__ import annotations
from tqdm import tqdm
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Literal, Optional, Tuple, Type
from scipy.ndimage import binary_dilation
import numpy as np
import cv2
import torch
import torch.distributed as dist
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image, to_tensor

from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import FullImageLidarDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from copy import deepcopy
from nerfstudio.data.datasets.base_dataset import SimpleDataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# print("!!!!!",str(Path(__file__).parent.parent.parent))
from src.pipeline_difix import DifixPipeline

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from nerfstudio.cameras.cameras import Cameras
@dataclass
class SplatADPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SplatADPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=FullImageLidarDatamanagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig) #代码中是别被骗了 SplatADModelConfig
    """specifies the model config"""
    calc_fid_steps: Tuple[int, ...] = (99999999,)  # 30000   NOTE: must also be an eval step for this to work 30000
    diffix_time: int =199
    steps_per_fix: int = 50000
    start_fix: int = 15000
    noval_rot: float = 30.0
    noval_eval_width: int = 1200
    save_traj: bool = True
    diffix2gpu: bool = False
    test: bool = False
class SplatADPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: SplatADPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank,diffix_one=self.diffix_one if self.config.diffix2gpu else None
        )

        self.config.noval_eval_width = self.config.datamanager.noval_eval_width
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            print("seed_pts 1")
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        elif (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "point_clouds" in self.datamanager.train_dataparser_outputs.metadata
            and "lidars" in self.datamanager.train_dataparser_outputs.metadata

        ):
            print("seed_pts 2")
            points_in_world = []
            returning_masks = []
            for l2w, pc in zip(
                self.datamanager.train_dataparser_outputs.metadata["lidars"].lidar_to_worlds[:20],
                self.datamanager.train_dataparser_outputs.metadata["point_clouds"][:20],
            ):
                returning = (
                    pc[:, :3].norm(dim=-1)
                    < self.datamanager.train_dataparser_outputs.metadata["lidars"].valid_lidar_distance_threshold
                )
                returning_masks.append(returning)
                points_in_world.append(transform_points(pc[returning, :3], l2w))
            points_in_world = torch.cat([pc_[:, :3] for pc_ in points_in_world], dim=0)
            print("points_in_world:",points_in_world.shape)
            if (
                "point_clouds_times" in self.datamanager.train_dataparser_outputs.metadata
                and self.datamanager.train_dataparser_outputs.metadata["point_clouds_times"] is not None
            ):

                points_in_world_times = torch.cat(
                    [
                        t_[r_]
                        for t_, r_ in zip(
                            self.datamanager.train_dataparser_outputs.metadata["point_clouds_times"][:20], returning_masks
                        )
                    ]
                )
                print("point_clouds_times:", points_in_world_times.shape)
            else:
                #111111
                points_in_world_times = None


            if (
                "point_clouds_rgb" in self.datamanager.train_dataparser_outputs.metadata
                and self.datamanager.train_dataparser_outputs.metadata["point_clouds_rgb"] is not None
            ):

                points_in_world_rgb = torch.cat(
                    [
                        c_[r_]
                        for c_, r_ in zip(
                            self.datamanager.train_dataparser_outputs.metadata["point_clouds_rgb"][:20], returning_masks
                        )
                    ]
                )
                print("point_clouds_rgb", points_in_world_rgb.shape)
            else:
                points_in_world_rgb = torch.rand_like(points_in_world) * 255
            seed_pts = (points_in_world, points_in_world_rgb, points_in_world_times)

        if seed_pts == None:
            print("No seed points")
        else:
            print("seed points size")

        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=self.datamanager.get_num_train_data(),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
        #qls_botanic
        if self.config.diffix2gpu:
            self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)

    def forward(self):

        raise NotImplementedError

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        lidar, batch = self.datamanager.next_eval_lidar(step)
        outputs = self.model.get_lidar_outputs(lidar)
        lidar_metrics_dict, lidar_images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        images_dict.update(lidar_images_dict)
        assert not set(lidar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(lidar_metrics_dict)

        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
            self,
            step: Optional[int] = None,
            output_path: Optional[Path] = None,
            get_std: bool = False,
            dump_img_to_disk: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        num_lidar = len(self.datamanager.fixed_indices_eval_lidar_dataloader)
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))

        if self.datamanager.config.slam and not(self.config.test) :
            self.eval_noval(step,output_path,dump_img_to_disk)
        if not(self.config.test) :
            return
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            lane_shift_fids = (
                {i: FrechetInceptionDistance().to(self.device) for i in (0, 2, 3)}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            vertical_shift_fids = (
                {i: FrechetInceptionDistance().to(self.device) for i in (1,)}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            actor_edits = {
                "rot": [(0.5, 0), (-0.5, 0)],
                "trans": [(0, 2.0), (0, -2.0)],
                # "both": [(0.5, 2.0), (-0.5, 2.0), (0.5, -2.0), (-0.5, -2.0)],
            }
            actor_fids = (
                {k: FrechetInceptionDistance().to(self.device) for k in actor_edits.keys()}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            if actor_fids:
                actor_fids["true"] = FrechetInceptionDistance().to(self.device)

            if self.datamanager.config.slam:
                # task = progress.add_task("[green]Evaluating all slam eval images...", total=len(self.datamanager.slam_eval_dataset))
                # dataloader = [ [it[2],it[3]]  for it in self.datamanager.slam_eval_dataset if it[1]=='camera' ]

                eval_cameras = getattr(self.datamanager.dataparser.config, "eval_cameras", ()) or ()
                train_cam_count = len(self.datamanager.dataparser.config.cameras)
                raw_camera_items = [it for it in self.datamanager.slam_eval_dataset if it[1] == "camera"]
                if len(eval_cameras) > 0:
                    selected_items = [it for it in raw_camera_items if int(it[4]) >= train_cam_count]
                else:
                    selected_items = raw_camera_items

                if len(selected_items) == 0 and len(raw_camera_items) > 0:
                    print("[eval][warn] selected slam eval cameras is empty after sensor filtering, fallback to raw camera items.")
                    selected_items = raw_camera_items

                dataloader = [[it[2], it[3]] for it in selected_items]
                task = progress.add_task("[green]Evaluating all slam eval images...", total=len(dataloader))

                # Debug summary to confirm we are really evaluating new-camera stream.
                sensor_ids = [int(it[4]) for it in selected_items]
                uniq_sensor_ids = sorted(set(sensor_ids))
                print(
                    f"[eval] slam_eval cameras={len(raw_camera_items)}, selected={len(selected_items)}, "
                    f"eval_cameras={eval_cameras}, train_cam_count={train_cam_count}, "
                    f"selected_sensor_ids={uniq_sensor_ids}"
                )
                if len(eval_cameras) > 0:
                    train_sensor_ids = [sid for sid in sensor_ids if sid < train_cam_count]
                    if len(train_sensor_ids) > 0:
                        print(f"[eval][warn] training-camera sensor ids mixed in selected eval stream: {sorted(set(train_sensor_ids))}")

                for dbg_idx, (camera_dbg, _) in enumerate(dataloader[:10]):
                    sensor_dbg = None
                    if camera_dbg.metadata is not None and "sensor_idxs" in camera_dbg.metadata:
                        try:
                            sensor_dbg = int(camera_dbg.metadata["sensor_idxs"].item())
                        except Exception:
                            sensor_dbg = camera_dbg.metadata["sensor_idxs"]
                    cam_idx_dbg = camera_dbg.metadata.get("cam_idx", "na") if camera_dbg.metadata is not None else "na"
                    scene_idx_dbg = camera_dbg.metadata.get("scene_index", "na") if camera_dbg.metadata is not None else "na"
                    print(
                        f"[eval][sample {dbg_idx}] sensor_idx={sensor_dbg}, cam_idx={cam_idx_dbg}, scene_index={scene_idx_dbg}"
                    )
            else:
                task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
                dataloader = self.datamanager.fixed_indices_eval_dataloader

            for camera, batch in dataloader:
                torch.cuda.synchronize()
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                torch.cuda.synchronize()
                inference_time_camera = time() - inner_start
                height, width = camera.height, camera.width
                num_camera_rays = height * width
                # Compute metrics for the original image
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                pred_height, pred_width = outputs["rgb"].shape[:2]
                batch["image"] = batch["image"][:pred_height, :pred_width]
                if True or dump_img_to_disk:
                    assert output_path is not None
                    os.makedirs(output_path, exist_ok=True)
                    os.makedirs(output_path / "fid", exist_ok=True)
                    # os.makedirs(output_path / "fid" / "gt_rgb", exist_ok=True)

                    gt_img = batch["image"]
                    if gt_img.max() > 1:
                        gt_img = gt_img / 255.0
                    pred_height, pred_width = outputs["rgb"].shape[:2]
                    gt_img = gt_img[:pred_height, :pred_width]
                    os.makedirs(output_path / "fid" / "pred_rgb" / str(step), exist_ok=True)

                    scene_idx = camera.metadata["scene_index"]
                    cam_idx = int(camera.metadata["cam_idx"])
                    filename = f"{cam_idx:06d}_{scene_idx}.png"

                    if self.datamanager.config.slam:


                        # 使用 f-string 创建新的文件名格式，例如："场景索引_摄像头索引.png"
                        # :06d 的格式化语法保持不变，作用于 cam_idx


                        # 保存基准真值图像 (ground truth image)
                        # if step < 30000:
                        #     # Image.fromarray((gt_img * 255).byte().cpu().numpy()).save(
                        #     #     output_path / "fid" / "gt_rgb" / filename
                        #     # )

                        # 保存预测图像 (predicted image)
                        gt_numpy = (gt_img * 255).byte().cpu().numpy()
                        pred_numpy = (outputs["rgb"] * 255).byte().cpu().numpy()

                        # 拼接图片 (axis=1 代表横向拼接)
                        combined_image = np.concatenate((pred_numpy,gt_numpy), axis=1)

                        Image.fromarray(combined_image ).save(
                            output_path / "fid" / "pred_rgb" / str(step) / filename
                        )


                    else:
                        Image.fromarray((gt_img * 255).byte().cpu().numpy()).save(
                            output_path / "fid" / "gt_rgb" / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                        )

                        Image.fromarray((outputs["rgb"] * 255).byte().cpu().numpy()).save(
                            output_path / "fid" / "pred_rgb" / str(step) /"{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                        )

                    if "depth" in outputs:
                        # 1. 创建用于保存深度图的文件夹
                        depth_output_dir = output_path / "fid" / "noval_depth" / str(step)
                        depthmm_output_dir = output_path / "fid" / "noval_depth_mm" / str(step)
                        os.makedirs(depth_output_dir, exist_ok=True)
                        os.makedirs(depthmm_output_dir, exist_ok=True)
                        # 2. 提取并归一化深度图
                        depth_tensor = outputs["depth"].squeeze()

                        depth_in_mm = depth_tensor * 1000.0

                        depth_image_16uc1 = depth_in_mm.cpu().numpy().astype(np.uint16)

                        full_save_path = str(depthmm_output_dir / filename)
                        cv2.imwrite(full_save_path, depth_image_16uc1)


                        # for vis
                        min_depth = 0.0
                        max_depth = 60.0
                        clamped_depth = torch.clamp(depth_tensor, min_depth, max_depth)
                        if max_depth > min_depth:
                            depth_normalized = (clamped_depth - min_depth) / (max_depth - min_depth)
                        else:
                            depth_normalized = torch.zeros_like(depth_tensor)

                        # 3. 转换为图像格式并保存
                        depth_image_data = (depth_normalized * 255).byte().cpu().numpy()
                        depth_image = Image.fromarray(depth_image_data)

                        # depth_image.save(depth_output_dir / filename)

                # if output_path is not None:
                #     raise NotImplementedError("Saving images is not implemented ye    t")

                assert "num_camera_rays_per_sec" not in metrics_dict
                metrics_dict["num_camera_rays_per_sec"] = (num_camera_rays / inference_time_camera).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_camera_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)

                if lane_shift_fids:
                    if dump_img_to_disk:
                        assert output_path is not None
                        for shift in lane_shift_fids.keys():
                            if shift == 0:
                                continue
                            os.makedirs(output_path / "fid" / f"lane_shift_{shift}", exist_ok=True)
                    self._update_lane_shift_fid(
                        lane_shift_fids, camera, batch["image"], outputs["rgb"], dump_img_to_disk, output_path
                    )
                if vertical_shift_fids:
                    if dump_img_to_disk:
                        assert output_path is not None
                        for shift in vertical_shift_fids.keys():
                            os.makedirs(output_path / "fid" / f"vertical_shift_{shift}", exist_ok=True)
                    self._update_vertical_shift_fid(
                        vertical_shift_fids, camera, batch["image"], dump_img_to_disk, output_path
                    )
                if actor_fids:
                    if dump_img_to_disk:
                        assert output_path is not None
                        for edit_type, edit_amounts in actor_edits.items():
                            for edit_amount in edit_amounts:
                                edit_amount = edit_amount[0] if edit_type == "rot" else edit_amount[1]
                                mount_prefix = "neg" if edit_amount < 0 else "pos"
                                if abs(edit_amount - int(edit_amount)) < 1e-6:
                                    edit_amount = mount_prefix + str(int(edit_amount))
                                else:
                                    edit_amount = mount_prefix + str(edit_amount).replace(".", "0")
                                os.makedirs(
                                    output_path / "fid" / f"actor_shift_{edit_type}_{edit_amount}", exist_ok=True
                                )
                    self._update_actor_fids(
                        actor_fids, actor_edits, camera, batch["image"], dump_img_to_disk, output_path
                    )
                progress.advance(task)

            if self.datamanager.config.slam :
                task = progress.add_task("[green]Evaluating all slam train images...",
                                         total=len(self.datamanager.slam_train_dataset))

                dataloader = [[it[2], it[3]] for it in self.datamanager.slam_train_dataset if it[1] == 'camera']


                for camera, batch in dataloader:
                    torch.cuda.synchronize()
                    # time this the following line
                    inner_start = time()
                    outputs = self.model.get_outputs_for_camera(camera=camera)
                    torch.cuda.synchronize()
                    inference_time_camera = time() - inner_start
                    height, width = camera.height, camera.width
                    num_camera_rays = height * width
                    # Compute metrics for the original image
                    metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                    pred_height, pred_width = outputs["rgb"].shape[:2]
                    batch["image"] = batch["image"][:pred_height, :pred_width]
                    if True or dump_img_to_disk:
                        assert output_path is not None
                        os.makedirs(output_path, exist_ok=True)
                        os.makedirs(output_path / "fid", exist_ok=True)

                        pred_height, pred_width = outputs["rgb"].shape[:2]
                        os.makedirs(output_path / "fid" / "train_pred_rgb" / str(step), exist_ok=True)

                        scene_idx = camera.metadata["scene_index"]
                        cam_idx = int(camera.metadata["cam_idx"])
                        filename = f"{cam_idx:06d}_{scene_idx}.png"

                        if self.datamanager.config.slam:
                            gt_img = batch["image"]
                            if gt_img.max() > 1:
                                gt_img = gt_img / 255.0
                            pred_height, pred_width = outputs["rgb"].shape[:2]
                            gt_img = gt_img[:pred_height, :pred_width]

                            gt_numpy = (gt_img * 255).byte().cpu().numpy()
                            pred_numpy = (outputs["rgb"] * 255).byte().cpu().numpy()

                            combined_image = np.concatenate((pred_numpy, gt_numpy), axis=1)

                            Image.fromarray(combined_image).save(
                                output_path / "fid" / "train_pred_rgb" / str(step) / filename
                            )

                        if "depth" in outputs:
                            # 1. 创建用于保存深度图的文件夹
                            depth_output_dir = output_path / "fid" / "train_noval_depth" / str(step)
                            os.makedirs(depth_output_dir, exist_ok=True)

                            # 2. 提取并归一化深度图
                            depth_tensor = outputs["depth"].squeeze()
                            min_depth = 0.0
                            max_depth = 60.0
                            clamped_depth = torch.clamp(depth_tensor, min_depth, max_depth)
                            if max_depth > min_depth:
                                depth_normalized = (clamped_depth - min_depth) / (max_depth - min_depth)
                            else:
                                depth_normalized = torch.zeros_like(depth_tensor)

                            # 3. 转换为图像格式并保存
                            depth_image_data = (depth_normalized * 255).byte().cpu().numpy()
                            depth_image = Image.fromarray(depth_image_data)

                            depth_image.save(depth_output_dir / filename)


##################add noval nvs dump




                train_camera_list = [[it[2], it[3]] for it in self.datamanager.slam_train_dataset if it[1] == 'camera']
                # rot = self.datamanager.config.noval_ag
                # trans = self.datamanager.config.noval_tr

                rot = 30
                trans = 1
                rot_save_dir = output_path /  f"train_rotated_{rot}" / str(step)
                trans_save_dir = output_path /  f"train_trans_{trans}" / str(step)
                os.makedirs(rot_save_dir, exist_ok=True)
                os.makedirs(trans_save_dir, exist_ok=True)

                for _camera,batch in train_camera_list:
                    # 深度拷贝相机对象以免影响原训练流程
                    camera = deepcopy(_camera).to(self.device)

                    # 1. 获取当前相机位姿 numpy 数组 [N, 4, 4]
                    c2w_cpu = self.camera_to_worlds_to_numpy(camera.camera_to_worlds)

                    # 2. 应用 60 度旋转 (使用类中已有的辅助函数)
                    # 注意：apply_z_rotation_and_translation 内部实现是绕 Y 轴旋转 (对应 XZ 平面)，
                    # 这通常是自动驾驶场景下的全景/环视旋转方向。

                    c2w_rotated = self.apply_z_rotation_and_translation(
                        c2w_cpu,
                        rot,  # 旋转角度
                        [0.0, 0.0, 0.0]  # 平移向量
                    )

                    c2w_trans = self.apply_z_rotation_and_translation(
                        c2w_cpu,
                        0.0,  # 旋转角度
                        [-trans, 0.0, 0.0]  # 平移向量
                    )


                    # 3. 将旋转后的位姿赋值回相机对象
                    render_tasks = [
                        (c2w_rotated, rot_save_dir),
                        (c2w_trans, trans_save_dir)
                    ]

                    for c2w_aft,of in render_tasks:
                        camera.camera_to_worlds = torch.from_numpy(c2w_aft[:, :3, :]).float().to(self.device)

                        # 4. 渲染
                        with torch.no_grad():
                            outputs = self.model.get_outputs_for_camera(camera)

                        # 构建文件名
                        if camera.metadata is not None:
                            scene_idx = camera.metadata.get("scene_index", 0)
                            # 尝试获取 cam_idx，如果不存在则使用循环索引
                            if "cam_idx" in camera.metadata:
                                cam_idx = int(camera.metadata["cam_idx"])
                            elif "slam_frame_idx" in camera.metadata:
                                cam_idx = int(camera.metadata["slam_frame_idx"])
                            else:
                                cam_idx = -1
                        else:
                            scene_idx = 0
                            cam_idx = -1

                        filename = f"{cam_idx:06d}_{scene_idx}.png"


                        gt_img = batch["image"]
                        if gt_img.max() > 1:
                            gt_img = gt_img / 255.0
                        pred_height, pred_width = outputs["rgb"].shape[:2]
                        gt_img = gt_img[:pred_height, :pred_width]

                        gt_numpy = (gt_img * 255).byte().cpu().numpy()
                        pred_numpy = (outputs["rgb"] * 255).byte().cpu().numpy()

                        combined_image = np.concatenate((pred_numpy, gt_numpy), axis=1)

                        Image.fromarray(combined_image).save(
                            of / filename
                        )
                        # 保存为 PNG
            # task = progress.add_task("[green]Evaluating all eval point clouds...", total=num_lidar)
            # for lidar, batch in self.datamanager.fixed_indices_eval_lidar_dataloader:
            #     torch.cuda.synchronize()
            #     inner_start = time()
            #     outputs = self.model.get_lidar_outputs(lidar)
            #     torch.cuda.synchronize()
            #     inference_time_lidar = time() - inner_start
            #     metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
            #     num_lidar_rays = (batch["raster_pts"][..., 2] > 0).sum()
            #     assert "num_lidar_rays_per_sec" not in metrics_dict
            #     metrics_dict["num_lidar_rays_per_sec"] = (num_lidar_rays / inference_time_lidar).item()
            #     metrics_dict_list.append(metrics_dict)
            #     if dump_img_to_disk:
            #         assert output_path is not None
            #         os.makedirs(output_path / "fid" / "lidar", exist_ok=True)
            #         gt_points = batch["lidar"][batch["lidar_pts_did_return"]]  # N, 5 (xyz, intensity, time_offset)
            #         # if filter_lidar_pred_and_gt is a function of the model, call it here
            #         if hasattr(self.model, "filter_lidar_pred_and_gt"):
            #             lidar_pred, lidar_gt = self.model.filter_lidar_pred_and_gt(
            #                 outputs, batch, output_point_cloud=True
            #             )
            #             pred_points = lidar_pred["point_cloud"]  # M, 3
            #             pred_points_median = lidar_pred["median_point_cloud"]
            #             pred_points_mask = (lidar_pred["ray_drop"].sigmoid() <= 0.5) * lidar_gt["valid"]
            #             intensity = outputs["intensity"].flatten()[pred_points_mask]
            #             time_offset = batch["raster_pts"][..., 3].flatten()[pred_points_mask]
            #             pred_points = torch.cat(
            #                 [pred_points, intensity[..., None], time_offset[..., None]], dim=-1
            #             )  # M, 5 (xyz, intensity, time_offset)
            #             pred_points_median = torch.cat(
            #                 [pred_points_median, intensity[..., None], time_offset[..., None]], dim=-1
            #             )  # M, 5 (xyz, intensity, time_offset)
            #
            #             # save the pred_points and gt_points to a file
            #             np.savez(
            #                 output_path / "fid" / "lidar" / f"points_{str(lidar.metadata['cam_idx']).zfill(6)}.npz",
            #                 pred_points=pred_points.cpu().numpy(),
            #                 pred_points_median=pred_points_median.cpu().numpy(),
            #                 gt_points=gt_points.cpu().numpy(),
            #             )
            #     progress.advance(task)

        # average the metrics list
        metrics_dict = {}
        keys = {key for metrics_dict in metrics_dict_list for key in metrics_dict.keys()}
        # remove the keys related to actor metrics as they need to be averaged differently
        actor_keys = {key for key in keys if key.startswith("actor_")}
        keys = keys - actor_keys

        for key in keys:
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                    )
                )

        # average the actor metrics. Note that due to the way we compute the actor metrics,
        # we need to weight them by how big portion of the image they cover.
        actor_metrics_dict = [md for md in metrics_dict_list if "actor_coverage" in md]
        if actor_metrics_dict:
            actor_coverages = torch.tensor([md["actor_coverage"] for md in actor_metrics_dict])
            for key in actor_keys:
                # we dont want to average the actor coverage in this way.
                if key == "actor_coverage":
                    continue
                # we should weight the actor metrics by the actor coverage
                metrics_dict[key] = float(
                    torch.sum(
                        torch.tensor(
                            [md[key] for md in actor_metrics_dict],
                        )
                        * actor_coverages
                    )
                    / actor_coverages.sum()
                )

        # Add FID metrics (if applicable)
        for shift, fid in lane_shift_fids.items():
            metrics_dict[f"lane_shift_{shift}_fid"] = fid.compute().item()

        for shift, fid in vertical_shift_fids.items():
            metrics_dict[f"vertical_shift_{shift}_fid"] = fid.compute().item()

        if actor_fids:
            for edit_type in actor_edits.keys():
                metrics_dict[f"actor_shift_{edit_type}_fid"] = actor_fids[edit_type].compute().item()

        self.train()
        return metrics_dict
    @torch.no_grad()

    def eval_noval(self,step: Optional[int] = None,
        output_path: Optional[Path] = None,save_images=True):

        print("Running eval_noval...")
        print(self.datamanager.eval_dataset.cameras.camera_to_worlds.shape)
        eval_poses = self.camera_to_worlds_to_numpy(self.datamanager.eval_dataset.cameras.camera_to_worlds)
        # eval_img = [d.copy() for d in self.datamanager.eval_dataset.cached_eval]

        train_poses = self.camera_to_worlds_to_numpy(self.datamanager.train_dataset.cameras.camera_to_worlds)

        dataloader_output = self.datamanager.slam_dataset  # 注意这里要加括号()来调用函数

        camera_items = [item for item in dataloader_output if item[1] == 'camera']

        # all_cameras = [item[0] for item in dataloader_output]
        # all_img = [item[1]['image'].cpu().numpy() for item in dataloader_output]
        # camera_to_worlds_tensors = [cam.camera_to_worlds for cam in all_cameras]
        # combined_camera_to_worlds = torch.cat(camera_to_worlds_tensors, dim=0)

        all_camera_objects = [item[2] for item in camera_items]
        # 提取所有相机的 camera_to_worlds, sensor_idxs, 和 times 张量
        camera_to_worlds_tensors = [cam.camera_to_worlds for cam in all_camera_objects]
        sensor_idxs_tensors = [cam.metadata['sensor_idxs'] for cam in all_camera_objects]
        times_tensors = [cam.times for cam in all_camera_objects]

        # 将列表中的张量合并成一个大的张量，方便进行批处理
        combined_camera_to_worlds = torch.cat(camera_to_worlds_tensors, dim=0)
        combined_sensor_idxs = torch.cat(sensor_idxs_tensors, dim=0)
        combined_times = torch.cat(times_tensors, dim=0)

        T_rgb0_vlp16 = camera_items[0][2].metadata['lidar2lcam']


        # 6. 保存筛选并处理后的位姿和对应的时间戳
        if output_path is not None and self.config.save_traj:
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
                pose_file_path_with_time = output_path / f"novel_poses_sensor_0_xyzijkw_{step}.txt"
                with open(pose_file_path_with_time, "w") as f:
                    pose_file_path_no_time = output_path / f"novel_kitti_poses_sensor_0_{step}.txt"
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
                            pose_4x4 = T_rgb0_vlp16 @ (pose_4x4 @ opencv_to_nerfstudio_4x4 @ T_rgb0_vlp16) @ np.linalg.inv(T_rgb0_vlp16)

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

        if step == 0 :
            return
        for index, (_camera, _batch) in enumerate(
                tqdm(self.datamanager.fixed_indices_train_dataloader, desc="Processing noval images")):
            torch.cuda.synchronize()
            # time this the following line
            camera = deepcopy(_camera).to(self.device)
            batch = _batch.copy()


            # camera.camera_to_worlds = torch.from_numpy(novel_poses[index][:3, :]).float()
            new_width_scalar = self.config.noval_eval_width
            device = camera.device  # 或者直接用 self.device

            camera.cx = torch.tensor([new_width_scalar / camera.width * camera.cx ], dtype=torch.float, device=device)
            camera.width = torch.tensor([new_width_scalar], dtype=torch.int32, device=device)

            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera(camera=camera)
            torch.cuda.synchronize()
            height, width = camera.height, camera.width
            num_camera_rays = height * width
            pred_height, pred_width = outputs["rgb"].shape[:2]
            batch["image"] = batch["image"][:pred_height, :pred_width]
            # if save_images:
            if True :
                assert output_path is not None
                os.makedirs(output_path, exist_ok=True)
                os.makedirs(output_path / "fid", exist_ok=True)
                # os.makedirs(output_path / "fid" / "gt_rgb", exist_ok=True)
                os.makedirs(output_path / "fid" / "noval_rgb"/ str(step) , exist_ok=True)
                gt_img = batch["image"]
                if gt_img.max() > 1:
                    gt_img = gt_img / 255.0
                pred_height, pred_width = outputs["rgb"].shape[:2]
                gt_img = gt_img[:pred_height, :pred_width]

                Image.fromarray((outputs["rgb"] * 255).byte().cpu().numpy()).save(
                    output_path / "fid"  / "noval_rgb" /  str(step) /"{0:06d}.png".format(
                        int(camera.metadata["cam_idx"]))
                )

    def diffusion_fix(self,step: Optional[int] = None,
        output_path: Optional[Path] = None,save_images=True):
        self.eval()
        self.difix.set_progress_bar_config(disable=True)
        # self.difix.to("cuda:1")

        # torch.cuda.empty_cache()
        # 打印运行修复器
        print("Running fixer...")
        print(self.datamanager.eval_dataset.cameras.camera_to_worlds.shape)
        eval_poses = self.camera_to_worlds_to_numpy(self.datamanager.eval_dataset.cameras.camera_to_worlds)
        # eval_img = [d.copy() for d in self.datamanager.eval_dataset.cached_eval]

        # train_poses = self.camera_to_worlds_to_numpy(self.datamanager.train_dataset.cameras.camera_to_worlds)

        dataloader_output = self.datamanager.fixed_indices_train_dataloader  # 注意这里要加括号()来调用函数

        all_cameras = [item[0] for item in dataloader_output]
        all_img = [item[1]['image'].cpu().numpy() for item in dataloader_output]

        # 将所有训练相机合并为一个可批量操作的对象
        train_cameras = all_cameras[0].cat(all_cameras[1:]).to(self.device)
        train_poses = self.camera_to_worlds_to_numpy(train_cameras.camera_to_worlds)

        camera_to_worlds_tensors = [cam.camera_to_worlds for cam in all_cameras]


        combined_camera_to_worlds = torch.cat(camera_to_worlds_tensors, dim=0)
        iter = False

        if len(self.datamanager.noval_dataset) == 0 or not(iter) :

            novel_poses = self.camera_to_worlds_to_numpy(combined_camera_to_worlds)

        else:
            list_camera = self.datamanager.noval_dataset[-1].cameras

            c_numpy_list = [t.camera_to_worlds.cpu().numpy() for t in list_camera]

            novel_poses = np.stack(c_numpy_list, axis=0)
        #
        # novel_poses = self.shift_poses(novel_poses,
        #
        #                                eval_poses, distance=20.0)
        # novel_poses = self.apply_z_rotation_and_translation(novel_poses,30.0,[0.0,0.0,0.0])

        num_poses = len(novel_poses)
        mid_point = num_poses // 2

        first_half_poses = novel_poses[:mid_point]
        second_half_poses = novel_poses[mid_point:]

        rotated_first_half = self.apply_z_rotation_and_translation(
            first_half_poses, self.config.noval_rot, [0.0, 0.0, 0.0]
        )


        rotated_second_half = self.apply_z_rotation_and_translation(
            second_half_poses, -1.0*self.config.noval_rot, [0.0, 0.0, 0.0]
        )

        novel_poses = np.concatenate([rotated_first_half, rotated_second_half], axis=0)

        ref_image_indices = self.find_nearest_assignments(train_poses,novel_poses)
        ref_image_indices = np.array(ref_image_indices)
        all_img_np = np.array(all_img)
        ref_image = all_img_np[ref_image_indices]

        noval_camera, noval_data = [], []
        outputs_img = []
        batch_list = []
        for index,(_camera, _batch) in enumerate(tqdm(self.datamanager.fixed_indices_train_dataloader,desc="Processing noval images")):
            torch.cuda.synchronize()
            # time this the following line
            camera = deepcopy(_camera).to(self.device)
            batch = _batch.copy()
            with torch.no_grad():
                outputs_old = self.model.get_outputs_for_camera(camera=camera)
            camera.camera_to_worlds = torch.from_numpy(novel_poses[index][:3,:]).float()
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera(camera=camera)
            torch.cuda.synchronize()
            height, width = camera.height, camera.width
            num_camera_rays = height * width
            pred_height, pred_width = outputs["rgb"].shape[:2]
            batch["image"] = batch["image"][:pred_height, :pred_width]
            if save_images:
                assert output_path is not None
                os.makedirs(output_path, exist_ok=True)
                os.makedirs(output_path / "diffix", exist_ok=True)
                os.makedirs(output_path / "diffix" / "gt_rgb", exist_ok=True)
                os.makedirs(output_path / "diffix" / str(step)/ "bef_rgb", exist_ok=True)
                gt_img = batch["image"]
                if gt_img.max() > 1:
                    gt_img = gt_img / 255.0
                pred_height, pred_width = outputs["rgb"].shape[:2]
                gt_img = gt_img[:pred_height, :pred_width]
                Image.fromarray((gt_img * 255).byte().cpu().numpy()).save(
                    output_path / "diffix" / "gt_rgb" / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                )
                Image.fromarray((outputs["rgb"] * 255).byte().cpu().numpy()).save(
                    output_path / "diffix" / str(step) / "bef_rgb" / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                )

            outputs['depth_from_old_pose'] = outputs_old['depth']
            outputs_img.append(outputs)
            noval_camera.append(camera)
            batch_list.append(batch)


        for index,outputs in enumerate(tqdm(outputs_img,desc="Processing fix")):

            crop_ref_image = ref_image[index][:height, :width]

            img = Image.fromarray((outputs["rgb"]* 255).byte().cpu().numpy().astype('uint8')).convert('RGB')
            ref_img = Image.fromarray(crop_ref_image.astype('uint8')).convert('RGB')
            bef_time = time()
            output_image = \
            self.difix(prompt="remove degradation", image=img, ref_image=ref_img, num_inference_steps=1,
                       timesteps=[self.config.diffix_time], guidance_scale=0.0).images[0]

            print("difix:", time()-bef_time)
            if save_images:
                os.makedirs(output_path / "diffix" / str(step) / "ref_rgb", exist_ok=True)
                os.makedirs(output_path / "diffix" / str(step) / "aft_rgb", exist_ok=True)
                ref_img.save(
                        output_path / "diffix" / str(step) / "ref_rgb" / "{0:06d}.png".format(int(noval_camera[index].metadata["cam_idx"]))
                )
                output_image.save(
                    output_path / "diffix" / str(step) / "aft_rgb" / "{0:06d}.png".format(int(noval_camera[index].metadata["cam_idx"]))
                )

            output_image = np.array(output_image, dtype="uint8")
            assert output_image.dtype == np.uint8
            assert output_image.shape[2] in [3, 4], f"Image shape of {output_image.shape} is in correct."

            output_image = torch.from_numpy(output_image)
            new_batch = batch_list[index]
            new_batch['image']= output_image
    ########

            original_depth = outputs['depth_from_old_pose'].squeeze()

            novelty_mask = self._create_mask_by_forward_projection(
                train_cameras[index],
               original_depth,
             noval_camera[index]
            )


            new_batch['mask'] = novelty_mask[:, :, None]

            if True or save_images:
                os.makedirs(output_path / "diffix" / str(step) / "mask", exist_ok=True)
                black_image = torch.zeros_like( outputs["rgb"]* 255)
                final_image = torch.where(
                    novelty_mask.unsqueeze(-1),  # torch.where 更喜欢布尔条件
                    output_image.to(novelty_mask.device),
                    black_image
                )

                # new_batch['image'] = final_image

                Image.fromarray(final_image.byte().cpu().numpy()).save(
                    output_path / "diffix" / str(step) / "mask" / "{0:06d}.png".format(
                        int(noval_camera[index].metadata["cam_idx"]))
                )

    ########


            noval_data.append(new_batch)

            #qls test
            # break

        new = SimpleDataset()
        new.cameras.extend(noval_camera)
        new.data.extend(noval_data)
        self.datamanager.noval_dataset.append(new)
        self.datamanager.train_unseen_noval = [i for i in range(len(self.datamanager.noval_dataset[-1]))]

        print("Finish fixer...")

    @torch.no_grad()
    def diffix_one(self,camera,ref_camera_tensor,timers,scalar = False, save_dir: str = None, frame_idx: int = 0,bev: bool = False):
        # 600 960 3
        # torch.cuda.synchronize()
        self.eval()
        # time this the following line
        # camera = deepcopy(_camera).to(self.device)
        # batch = _batch.copy()

        # camera.camera_to_worlds = torch.from_numpy(novel_poses[index][:3, :]).float()
        new_width_scalar = self.config.noval_eval_width
        device = camera.device  # 或者直接用 self.device

        if scalar :
            camera.cx = torch.tensor([new_width_scalar / camera.width * camera.cx], dtype=torch.float, device=device)
            camera.width = torch.tensor([new_width_scalar], dtype=torch.int32, device=device)


        if bev :
            camera.cx = camera.cx * 2
            camera.width = camera.width * 2
            camera.cy = camera.cy * 2
            camera.height = camera.height * 2

        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera(camera=camera)


        height = camera.height.item()
        width = camera.width.item()

        img = Image.fromarray((outputs["rgb"] * 255).byte().cpu().numpy().astype('uint8')).convert('RGB')
        print("outputs[rgb]:",outputs["rgb"].shape)
        ref_img_np = ref_camera_tensor.cpu().numpy().astype('uint8')
        ref_img = Image.fromarray(ref_img_np).convert('RGB')
        # 原始尺寸
        if scalar:
            orig_h, orig_w, _ = ref_img_np.shape  # 应该是 600, 960

            # 目标裁剪尺寸
            crop_h = height  / width * orig_w  # 1200 / 600 * 960

            # 计算居中裁剪的起始和结束行
            crop_start_h =  int( (orig_h - crop_h) // 2)
            crop_end_h = int(crop_start_h + crop_h)

            # 执行居中裁剪
            cropped_img_np = ref_img_np[crop_start_h:crop_end_h, :, :]  # [60:540, 0:960, :]

            # 目标缩放尺寸
            target_w, target_h = width, height

            # 执行缩放（插值）
            # INTER_CUBIC 效果通常比 INTER_LINEAR 好，但稍慢
            resized_ref_img_np = cv2.resize(cropped_img_np, (width, height), interpolation=cv2.INTER_CUBIC)

            # 将处理后的 NumPy 数组转换为 PIL.Image 对象
            ref_img_processed = Image.fromarray(resized_ref_img_np).convert('RGB')

            ref_img = ref_img_processed


        bef_time = time()
        timers["Diffix_Gen"][0].record()
        if not(bev):
            output_image = \
                self.difix(prompt="remove degradation", image=img, ref_image=ref_img, num_inference_steps=1,
                           timesteps=[self.config.diffix_time], guidance_scale=0.0).images[0]
        else:
            output_image = img
        timers["Diffix_Gen"][1].record()

        print("difix:", time() - bef_time)
        output_image = output_image.resize(
            (width, height),
            resample=Image.BICUBIC  # 直接使用 PIL.Image 中的高质量插值常量
        )
        output_image = np.array(output_image, dtype="uint8")
        assert output_image.dtype == np.uint8
        assert output_image.shape[2] in [3, 4], f"Image shape of {output_image.shape} is in correct."

        output_image = torch.from_numpy(output_image)
        # new_batch = batch_list[index]
        # new_batch['image'] = output_image
        ########
        if scalar:
            self._visualize_plot(
                image_before=outputs["rgb"],
                ref_image=torch.from_numpy(resized_ref_img_np),
                image_after=output_image,
                save_dir = save_dir, frame_idx = frame_idx,
            title = "Diffix One - Visualization"
            )
        else:
            self._visualize_plot(
                image_before=outputs["rgb"],
                ref_image=torch.from_numpy(ref_img_np),
                image_after=output_image,
                title="Diffix One - Visualization",
                save_dir=save_dir, frame_idx=frame_idx,
            )
        return output_image

    def apply_z_rotation_and_translation(self,poses: np.ndarray,
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
    def camera_to_worlds_to_numpy(self,camera_tensor: torch.Tensor) -> np.ndarray:
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

    def shift_poses(self, training_poses, testing_poses, distance=0.1, threshold=0.1):
        """
        Shift nearest training poses toward testing poses by a specified distance.

        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            distance: float, the step size to move training pose toward testing pose

        Returns:
            novel_poses: [M, 4, 4] array of shifted poses
        """
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        novel_poses = []

        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]

            if self.compute_pose_distance(train_pose, test_pose) <= distance:
                novel_poses.append(test_pose)
                continue

            # Calculate translation step if shifting is necessary
            t1, t2 = train_pose[:3, 3], test_pose[:3, 3]
            translation_direction = t2 - t1
            translation_norm = np.linalg.norm(translation_direction)

            if translation_norm > 1e-6:
                translation_step = (translation_direction / translation_norm) * distance
                new_translation = t1 + translation_step
            else:
                # If translation direction is too small, use testing pose translation directly
                new_translation = t2

            # Check if the new translation would overshoot the testing pose translation
            if np.dot(new_translation - t1, t2 - t1) <= 0 or np.linalg.norm(new_translation - t2) <= distance:
                new_translation = t2

            # Update rotation
            R1 = train_pose[:3, :3]
            R2 = test_pose[:3, :3]
            if translation_norm > 1e-6:
                R_interp = self.interpolate_rotation(R1, R2, min(distance / translation_norm, 1.0))
            else:
                R_interp = R2  # Use testing rotation if too close

            # Construct shifted pose
            shifted_pose = np.eye(4)
            shifted_pose[:3, :3] = R_interp
            shifted_pose[:3, 3] = new_translation

            novel_poses.append(shifted_pose)

        return np.array(novel_poses)

    def interpolate_rotation(self, R1, R2, t):
        """
        Interpolate between two rotation matrices using SLERP.
        """
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()

        if np.dot(q1, q2) < 0:
            q2 = -q2

        # Clamp dot product to avoid invalid values in arccos
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        theta = np.arccos(dot_product)

        if np.abs(theta) < 1e-6:
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)

        q_interp = q_interp / np.linalg.norm(q_interp)
        return Rotation.from_quat(q_interp).as_matrix()
    def find_nearest_assignments(self, training_poses, testing_poses):
        """
        Find the nearest training camera pose for each testing camera pose.

        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses

        Returns:
            assignments: list of closest training pose indices for each testing pose
        """
        M = len(testing_poses)
        assignments = []

        for j in range(M):
            # Compute distance from each training pose to this testing pose
            distances = [self.compute_pose_distance(training_pose, testing_poses[j])
                         for training_pose in training_poses]
            # Find the index of the nearest training pose
            nearest_index = np.argmin(distances)
            assignments.append(nearest_index)

        return assignments

    def compute_pose_distance(self, pose1, pose2):
        """
        Compute weighted distance between two camera poses.

        Args:
            pose1, pose2: 4x4 transformation matrices

        Returns:
            Combined weighted distance between poses
        """
        # Translation distance (Euclidean)
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)

        # Rotation distance (angular distance between quaternions)
        R1 = Rotation.from_matrix(pose1[:3, :3])
        R2 = Rotation.from_matrix(pose2[:3, :3])
        q1 = R1.as_quat()
        q2 = R2.as_quat()

        # Ensure quaternions are in the same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2

        rotation_dist = np.arccos(2 * np.dot(q1, q2) ** 2 - 1)

        return (1.0 * translation_dist +
                1.0 * rotation_dist)

    @staticmethod
    def _downsample_img(
        img: torch.Tensor,
        out_size: Tuple[int, int] = (
            299,
            299,
        ),
    ):
        """Converts tensor to PIL, downsamples with bicubic, and converts back to tensor."""
        img = to_pil_image(img)
        img = img.resize(out_size, Image.BICUBIC)
        img = to_tensor(img)
        return img

    def _update_lane_shift_fid(
        self, fids: Dict[int, FrechetInceptionDistance], camera, orig_img, gen_img, dump_img_to_disk, output_path
    ):
        """Updates the FID metrics (for shifted views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # Compute FID for shifted views
        assert fids.keys() == {0, 2, 3}, "Shift amounts are hardcoded for now."
        imgs_generated = {0: gen_img}

        driving_direction = camera.metadata["velocities"][0].clone()
        driving_direction = driving_direction / driving_direction.norm()
        orth_right_direction = torch.cross(
            driving_direction, torch.tensor([0.0, 0.0, 1.0], device=driving_direction.device)
        )

        # TODO: Do we need to take z axis into account?
        shift_sign = self.datamanager.eval_lidar_dataset.metadata.get("lane_shift_sign", 1)
        original_camera_to_worlds = camera.camera_to_worlds.clone()
        camera.camera_to_worlds[0, :2, 3] += 2 * orth_right_direction[:2] * shift_sign
        imgs_generated[2] = self.model.get_outputs_for_camera(camera)["rgb"]
        camera.camera_to_worlds[0, :2, 3] += 1 * orth_right_direction[:2] * shift_sign
        imgs_generated[3] = self.model.get_outputs_for_camera(camera)["rgb"]
        camera.camera_to_worlds = original_camera_to_worlds
        for shift, img in imgs_generated.items():
            if dump_img_to_disk and shift != 0:
                assert output_path is not None
                fid_output_path = output_path / "fid" / f"lane_shift_{shift}"
                filepath = fid_output_path / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                Image.fromarray((img * 255).byte().cpu().numpy()).save(filepath)
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_vertical_shift_fid(
        self, fids: Dict[int, FrechetInceptionDistance], camera, orig_img, dump_img_to_disk, output_path
    ):
        """Updates the FID metrics (for shifted views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # Compute FID for shifted views
        assert fids.keys() == {1}, "Shift amounts are hardcoded for now."
        imgs_generated = {}

        original_camera_to_worlds = camera.camera_to_worlds.clone()
        camera.camera_to_worlds[0, 2, 3] += 1.0
        imgs_generated[1] = self.model.get_outputs_for_camera(camera)["rgb"]
        camera.camera_to_worlds = original_camera_to_worlds

        if dump_img_to_disk:
            assert output_path is not None
            fid_output_path = output_path / "fid" / "vertical_shift_1"
            filepath = fid_output_path / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
            Image.fromarray((imgs_generated[1] * 255).byte().cpu().numpy()).save(filepath)

        for shift, img in imgs_generated.items():
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_actor_fids(
        self,
        fids: Dict[str, FrechetInceptionDistance],
        actor_edits: Dict[str, List[Tuple]],
        camera,
        orig_img,
        dump_img_to_disk,
        output_path,
    ) -> None:
        """Updates the FID metrics (for shifted actor views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids["true"].update(img_original, real=True)
        for edit_type in actor_edits.keys():
            fids[edit_type].real_features_sum = fids["true"].real_features_sum
            fids[edit_type].real_features_cov_sum = fids["true"].real_features_cov_sum
            fids[edit_type].real_features_num_samples = fids["true"].real_features_num_samples

        # Compute FID for actor edits
        imgs_generated_per_edit = {}
        for edit_type in actor_edits.keys():
            imgs = []
            for rotation, lateral in actor_edits[edit_type]:
                self.model.dynamic_actors.actor_editing["rotation"] = rotation
                self.model.dynamic_actors.actor_editing["lateral"] = lateral
                prediction = self.model.get_outputs_for_camera(camera)["rgb"]
                imgs.append(prediction)
                if dump_img_to_disk:
                    assert output_path is not None
                    edit_amount = rotation if edit_type == "rot" else lateral
                    mount_prefix = "neg" if edit_amount < 0 else "pos"
                    if abs(edit_amount - int(edit_amount)) < 1e-6:
                        edit_amount = mount_prefix + str(int(edit_amount))
                    else:
                        edit_amount = mount_prefix + str(edit_amount).replace(".", "0")
                    fid_output_path = output_path / "fid" / f"actor_shift_{edit_type}_{edit_amount}"
                    filepath = fid_output_path / "{0:06d}.png".format(int(camera.metadata["cam_idx"]))
                    Image.fromarray((prediction * 255).byte().cpu().numpy()).save(filepath)
            imgs_generated_per_edit[edit_type] = imgs

        for edit_type, imgs in imgs_generated_per_edit.items():
            for img in imgs:
                img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
                fids[edit_type].update(img, real=False)

        self.model.dynamic_actors.actor_editing["rotation"] = 0
        self.model.dynamic_actors.actor_editing["lateral"] = 0



    def _create_mask_by_forward_projection(
            self,
            original_camera: Cameras,
            original_depth: torch.Tensor,
            novel_camera: Cameras
    ) -> torch.Tensor:
        """
        通过前向投影生成新颖性蒙版。(严格基于手动计算)

        此方法将原始视角可见的点云投影到新视角上，
        任何没有被投影点覆盖的区域都被视为“新颖”。

        Args:
            original_camera: 原始训练视角的相机对象。
            original_depth: 与原始相机匹配的深度图。
            novel_camera: 我们要为其生成蒙版的新视角相机对象。

        Returns:
            一个布尔类型的张量，形状为新相机的高度和宽度，在新颖区域为 True。
        """
        # 步骤 0: 统一设备并准备数据
        # 为避免任何错误，将所有计算都放在一个设备上（例如 CPU）
        device = torch.device('cpu')
        original_camera = original_camera.to(device)
        original_depth = original_depth.to(device)
        novel_camera = novel_camera.to(device)

        if original_depth.dim() == 3 and original_depth.shape[-1] == 1:
            original_depth = original_depth.squeeze(-1)

        H_orig, W_orig = original_depth.shape
        H_novel, W_novel = novel_camera.height.item(), novel_camera.width.item()

        # --- 步骤 1: 从原始视角反向投影，构建“可见世界”点云 ---
        yy_orig, xx_orig = torch.meshgrid(
            torch.arange(H_orig, device=device),
            torch.arange(W_orig, device=device),
            indexing="ij"
        )
        xx_orig, yy_orig = xx_orig.reshape(-1), yy_orig.reshape(-1)
        depth_flat = original_depth.reshape(-1)

        # 使用原始相机的内参
        fx_orig, fy_orig = original_camera.fx.item(), original_camera.fy.item()
        cx_orig, cy_orig = original_camera.cx.item(), original_camera.cy.item()

        x_ndc = (xx_orig - cx_orig) / fx_orig
        y_ndc = (yy_orig - cy_orig) / fy_orig

        points_cam_orig = torch.stack([x_ndc * depth_flat, y_ndc * depth_flat, depth_flat], dim=-1)
        points_cam_orig_hom = torch.cat([points_cam_orig, torch.ones_like(points_cam_orig[..., :1])], dim=-1)

        # 使用原始相机的外参，转换到世界坐标
        c2w_orig = original_camera.camera_to_worlds
        points_world = torch.matmul(points_cam_orig_hom, c2w_orig.T)

        # --- 步骤 2: 将世界点云投影到新视角 ---
        # a) 将世界点变换到新相机的坐标系
        c2w_novel = novel_camera.camera_to_worlds
        last_row = torch.tensor([0, 0, 0, 1], device=device).unsqueeze(0).unsqueeze(1)
        c2w_novel_hom = torch.cat([c2w_novel, last_row], dim=1)
        w2c_novel = torch.inverse(c2w_novel_hom)[:, :3, :]  # 获取 3x4 的世界到相机矩阵

        points_world_hom = torch.cat([points_world, torch.ones_like(points_world[..., :1])], dim=-1)
        points_in_novel_cam_space = torch.matmul(points_world_hom, w2c_novel[0].T)

        # b) 检查点是否在新相机前方
        z_in_novel = points_in_novel_cam_space[..., 2]
        is_in_front = z_in_novel > 1e-4

        # c) 将相机坐标系下的点投影到新相机的2D像素平面
        fx_novel, fy_novel = novel_camera.fx.item(), novel_camera.fy.item()
        cx_novel, cy_novel = novel_camera.cx.item(), novel_camera.cy.item()

        x_proj = (points_in_novel_cam_space[..., 0] / z_in_novel) * fx_novel + cx_novel
        y_proj = (points_in_novel_cam_space[..., 1] / z_in_novel) * fy_novel + cy_novel
        pixels_in_novel_cam = torch.stack([x_proj, y_proj], dim=-1)

        # --- 步骤 3: 筛选出有效投影点 ---
        # a) 检查点是否在新图像边界内
        is_in_bounds_x = (pixels_in_novel_cam[:, 0] >= 0) & (pixels_in_novel_cam[:, 0] < W_novel)
        is_in_bounds_y = (pixels_in_novel_cam[:, 1] >= 0) & (pixels_in_novel_cam[:, 1] < H_novel)
        is_in_bounds = is_in_bounds_x & is_in_bounds_y

        # b) 结合前方检查，得到最终有效点
        valid_mask = is_in_front & is_in_bounds
        valid_pixels = pixels_in_novel_cam[valid_mask].long()  # 转换为整数坐标用于索引

        # --- 步骤 4: 创建蒙版并在“已见”区域进行“绘制” ---
        # 初始化一个全为 True (所有像素都是新颖) 的蒙版
        novelty_mask = torch.ones(H_novel, W_novel, dtype=torch.bool, device=device)

        # 如果有有效的投影点
        if valid_pixels.shape[0] > 0:
            x_coords = valid_pixels[:, 0]
            y_coords = valid_pixels[:, 1]

            # 使用高级索引，将所有被投影点覆盖的像素设置为 False (表示已见过)
            novelty_mask[y_coords, x_coords] = False

        seen_mask = ~novelty_mask

        # 将 PyTorch 张量转为 NumPy 数组以使用 scipy
        seen_mask_np = seen_mask.cpu().numpy()

        # 定义膨胀的结构元素（可以理解为画笔的大小和形状）
        # 例如，一个 5x5 的方形画笔
        structure = np.ones((5, 5), dtype=bool)

        # 执行膨胀操作
        dilated_seen_mask_np = binary_dilation(seen_mask_np, structure=structure, iterations=1)

        # 将膨胀后的 "已见" 蒙版转换回 PyTorch 张量
        dilated_seen_mask = torch.from_numpy(dilated_seen_mask_np).to(device)

        # 再次取反，得到最终连续的 "新颖性" 蒙版
        final_novelty_mask = ~dilated_seen_mask

        # --- 步骤 6: (可选但推荐) 根据Y轴朝上的约定翻转蒙版 ---
        flipped_mask = torch.flip(torch.flip(final_novelty_mask, dims=[0]), dims=[1])
        # 将最终结果移回调用者期望的设备（假设是self.device）
        return flipped_mask.to(self.device)

    def _visualize_plot(self, image_before: torch.Tensor, ref_image: torch.Tensor, image_after: torch.Tensor, save_dir: str = None, frame_idx: int = 0,
                        title: str = "Visualization"):
        """
        使用 matplotlib 可视化三张图像：修复前、参考、修复后。

        Args:
            image_before (torch.Tensor): 原始渲染图，范围 [0, 1]。
            ref_image (torch.Tensor): 参考图，范围 [0, 255]。
            image_after (torch.Tensor): 修复后的图，范围 [0, 255]。
            title (str): 绘图窗口的标题。
        """
        # --- 准备用于绘图的 NumPy 数组 ---

        image_before_np = image_before.cpu().numpy()  # [0, 1] float
        ref_image_np = ref_image.cpu().numpy()  # [0, 255]
        image_after_np = (image_after.float() / 255.0).cpu().numpy()  # [0, 1] float

        if save_dir is not None:
            root_path = Path(save_dir)

            # 定义三个子文件夹名称
            save_dict = {
                "before": (image_before_np * 255).astype(np.uint8),
                "reference": ref_image_np.astype(np.uint8),
                "after": (image_after_np * 255).astype(np.uint8)
            }

            for sub_name, img_data in save_dict.items():
                # 创建子文件夹: save_dir/before, save_dir/reference, ...
                dir_path = root_path / sub_name
                dir_path.mkdir(parents=True, exist_ok=True)

                # 保存图片，文件名例如: 00123.png
                save_path = dir_path / f"{frame_idx:05d}.png"
                Image.fromarray(img_data).save(str(save_path))

        # --- 使用 Matplotlib 进行可视化 ---
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(title, fontsize=16)

        # 子图1: 原始渲染图像 (修复前)
        axes[0].imshow(image_before_np)
        axes[0].set_title("Before (Raw Render)")
        axes[0].axis('off')

        # 子图2: 参考图像
        axes[1].imshow(ref_image_np)
        axes[1].set_title("Reference Image")
        axes[1].axis('off')

        # 子图3: 修复后的图像
        axes[2].imshow(image_after_np)
        axes[2].set_title("After (Fixed)")
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应主标题
        plt.show()