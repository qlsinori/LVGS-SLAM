#!/usr/bin/env python
"""
ROS-based entry point for the 3DGS mapping module.

This script wraps the standard nerfstudio training pipeline as a ROS node,
enabling real-time communication with the SLAM module:
  - Subscribes to: /keyframe_pose, /keyframe_image, /keyframe_cloud  (from SLAM)
  - Publishes to:  /sensor/camera/depth/image_rect                   (for SLAM)

Usage (inside a ROS environment):
    python nerfstudio/scripts/train_ros.py splatad-wild --vis viewer+tensorboard botanic-data

Or via roslaunch (see the accompanying launch file).
"""

from __future__ import annotations

import signal
import sys
import threading

import rospy

from nerfstudio.scripts.train import main as train_main, entrypoint as _entrypoint
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig

import tyro


def _ros_spin_thread():
    """Run rospy.spin() in a background thread so ROS callbacks keep working."""
    rospy.spin()


def main():
    rospy.init_node('gs_mapping_node', anonymous=True, disable_signals=True)
    rospy.loginfo("[gs_mapping_node] ROS node initialized. Starting 3DGS mapping pipeline...")

    spin_thread = threading.Thread(target=_ros_spin_thread, daemon=True)
    spin_thread.start()

    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(
        AnnotatedBaseConfigUnion,
        description=convert_markup_to_ansi(
            "ROS-enabled 3DGS mapping trainer.\n"
            "Subscribes to keyframes from SLAM and publishes depth maps."
        ),
    )

    if hasattr(config, 'pipeline') and hasattr(config.pipeline, 'datamanager'):
        config.pipeline.datamanager.use_ros = True
        rospy.loginfo("[gs_mapping_node] Enabled use_ros=True in datamanager config.")

    train_main(config)

    rospy.loginfo("[gs_mapping_node] Training complete. Shutting down ROS node.")
    rospy.signal_shutdown("Training complete")


if __name__ == "__main__":
    main()
