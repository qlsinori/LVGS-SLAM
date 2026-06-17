#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keyframe Bridge Node
====================
Bridges SLAM module output to 3DGS mapping module.

Subscribes:
    /estimate/complete_path   (nav_msgs/Path)          — all keyframe poses from SLAM BA
    /sensor/camera/grayscale/left/image_rect (sensor_msgs/Image) — camera images from rosbag
    /sensor/velodyne/cloud_euclidean (sensor_msgs/PointCloud2)   — lidar clouds from rosbag

Publishes (for 3DGS mapping):
    /keyframe_pose   (geometry_msgs/PoseStamped)
    /keyframe_image  (sensor_msgs/Image)
    /keyframe_cloud  (sensor_msgs/PointCloud2)

Logic:
    Each time /estimate/complete_path grows (new keyframe from BA), the node
    looks up the latest keyframe timestamp, finds the nearest buffered image
    and cloud, and publishes the three messages with matching timestamps.
"""

import threading
import collections
import copy

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2


class KeyframeBridge:

    def __init__(self):
        rospy.init_node('keyframe_bridge', anonymous=False)

        # --- parameters ---
        self.image_buffer_sec = rospy.get_param('~image_buffer_sec', 5.0)
        self.cloud_buffer_sec = rospy.get_param('~cloud_buffer_sec', 5.0)
        self.time_tolerance = rospy.get_param('~time_tolerance', 0.15)

        # --- state ---
        self._lock = threading.Lock()
        self._image_buffer = collections.deque()
        self._cloud_buffer = collections.deque()
        self._published_kf_stamps = set()
        self._last_path_len = 0

        # --- publishers ---
        self.pub_kf_pose = rospy.Publisher('/keyframe_pose', PoseStamped, queue_size=30)
        self.pub_kf_image = rospy.Publisher('/keyframe_image', Image, queue_size=30)
        self.pub_kf_cloud = rospy.Publisher('/keyframe_cloud', PointCloud2, queue_size=30)

        # --- subscribers ---
        rospy.Subscriber('/estimate/complete_path', Path, self._path_cb, queue_size=30) 
        rospy.Subscriber(
            '/sensor/camera/grayscale/left/image_rect', Image,
            self._image_cb, queue_size=30
        )
        rospy.Subscriber(
            '/sensor/velodyne/cloud_euclidean', PointCloud2,
            self._cloud_cb, queue_size=30
        )

        rospy.loginfo('[keyframe_bridge] Node started.')
        rospy.loginfo('[keyframe_bridge]   Subscribing: /estimate/complete_path, '
                      '/sensor/camera/grayscale/left/image_rect, '
                      '/sensor/velodyne/cloud_euclidean')
        rospy.loginfo('[keyframe_bridge]   Publishing:  /keyframe_pose, '
                      '/keyframe_image, /keyframe_cloud')

    # ------------------------------------------------------------------ #
    #  Sensor data buffering
    # ------------------------------------------------------------------ #

    def _image_cb(self, msg):
        with self._lock:
            self._image_buffer.append(msg)
            self._trim_buffer(self._image_buffer, self.image_buffer_sec)

    def _cloud_cb(self, msg):
        with self._lock:
            self._cloud_buffer.append(msg)
            self._trim_buffer(self._cloud_buffer, self.cloud_buffer_sec)

    @staticmethod
    def _trim_buffer(buf, max_age_sec):
        """Remove messages older than max_age_sec from the front of buf."""
        if len(buf) < 2:
            return
        latest = buf[-1].header.stamp.to_sec()
        while len(buf) > 1 and (latest - buf[0].header.stamp.to_sec()) > max_age_sec:
            buf.popleft()

    # ------------------------------------------------------------------ #
    #  Path callback — detect new keyframes
    # ------------------------------------------------------------------ #

    def _path_cb(self, path_msg):
        """Called when odom publishes /estimate/complete_path."""
        n_poses = len(path_msg.poses)

        if n_poses <= self._last_path_len:
            return

        new_poses = path_msg.poses[self._last_path_len:]
        self._last_path_len = n_poses

        for pose_stamped in new_poses:
            kf_stamp = pose_stamped.header.stamp
            stamp_key = (kf_stamp.secs, kf_stamp.nsecs)

            if stamp_key in self._published_kf_stamps:
                continue

            with self._lock:
                image_msg = self._find_nearest(self._image_buffer, kf_stamp)
                cloud_msg = self._find_nearest(self._cloud_buffer, kf_stamp)

            if image_msg is None:
                rospy.logwarn_throttle(
                    2.0,
                    '[keyframe_bridge] No image near t=%.3f (buf=%d)' %
                    (kf_stamp.to_sec(), len(self._image_buffer))
                )
                continue

            if cloud_msg is None:
                rospy.logwarn_throttle(
                    2.0,
                    '[keyframe_bridge] No cloud near t=%.3f (buf=%d)' %
                    (kf_stamp.to_sec(), len(self._cloud_buffer))
                )

            unified_stamp = kf_stamp

            out_pose = copy.deepcopy(pose_stamped)
            out_pose.header.stamp = unified_stamp
            out_pose.header.frame_id = 'estimate/local_cs_vehicle'
            self.pub_kf_pose.publish(out_pose)

            out_image = copy.deepcopy(image_msg)
            out_image.header.stamp = unified_stamp
            self.pub_kf_image.publish(out_image)

            if cloud_msg is not None:
                out_cloud = copy.deepcopy(cloud_msg)
                out_cloud.header.stamp = unified_stamp
                self.pub_kf_cloud.publish(out_cloud)
            else:
                empty_cloud = PointCloud2()
                empty_cloud.header.stamp = unified_stamp
                empty_cloud.header.frame_id = 'sensor/velodyne'
                self.pub_kf_cloud.publish(empty_cloud)

            self._published_kf_stamps.add(stamp_key)

            rospy.loginfo(
                '[keyframe_bridge] Published keyframe t=%.3f  (total %d)' %
                (kf_stamp.to_sec(), len(self._published_kf_stamps))
            )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _find_nearest(self, buf, target_stamp):
        """Find the message in buf whose timestamp is closest to target_stamp."""
        if len(buf) == 0:
            return None

        target_sec = target_stamp.to_sec()
        best_msg = None
        best_dt = float('inf')

        for msg in buf:
            dt = abs(msg.header.stamp.to_sec() - target_sec)
            if dt < best_dt:
                best_dt = dt
                best_msg = msg

        if best_dt > self.time_tolerance:
            return None
        return best_msg

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = KeyframeBridge()
        node.spin()
    except rospy.ROSInterruptException:
        pass
