#!/bin/bash
# Launcher script for the 3DGS mapping ROS node.
# Usage: gs_mapping_launcher.sh <workspace> <method> <vis> <data>
#   workspace: path to LVGS-SLAM root (e.g. /slam/code/LVGS-SLAM)
#   method:    nerfstudio method name (e.g. splatad-wild)
#   vis:       visualization mode (e.g. viewer+tensorboard)
#   data:      data config name (e.g. botanic-data)

GS_WORKSPACE="$1"
GS_METHOD="$2"
GS_VIS="$3"
GS_DATA="$4"

cd "$GS_WORKSPACE" || { echo "[gs_mapping_launcher] Cannot cd to $GS_WORKSPACE"; exit 1; }

exec python nerfstudio/scripts/train_ros.py "$GS_METHOD" --vis "$GS_VIS" "$GS_DATA"
