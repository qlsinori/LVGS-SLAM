
#include "tracklet_depth.h"

#include <image_geometry/pinhole_camera_model.h>

#include "std_msgs/String.h"

#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>
#include <limits> // 用于 std::numeric_limits

#include <chrono>

namespace tracklets_depth_ros_tool {

TrackletDepth::TrackletDepth(ros::NodeHandle node_handle, ros::NodeHandle private_node_handle)
        : _image_transport{node_handle} {
    _nodeHandle = {boost::make_shared<ros::NodeHandle>(node_handle)};
    _cloud_last_frame = nullptr;
    _combined_cloud_last_frame = nullptr;
    // parameters
    std::cout << "Initalize parameters" << std::endl;

    // get paths to config files
    std::string settingsPath;

    if (!private_node_handle.getParam("config_tracklet_depth", settingsPath)) {
        throw std::runtime_error("No configuration specified for: config_tracklet_depth");
    }

    std::cout << "config_tracklets_depth: " << settingsPath << std::endl;

    if (!private_node_handle.getParam("config_depth_estimator", _path_config_depthEstimator)) {
        throw std::runtime_error("No configuration specified for: config_depth_estimator");
    }

    std::cout << "config_tracklets_depth: " << _path_config_depthEstimator << std::endl;

    _params.fromFile(settingsPath);
    //    _params.print();

    depth_estimator_parameters_.fromFile(_path_config_depthEstimator);

    // Initialize
    this->InitDepthEstimatorPre();
    this->InitSubscriber(node_handle, _params.subscriber_msg_name_semantics != "");
    this->InitPublisher(node_handle);
    this->InitStaticTransforms();

    // ransac plane will be initialializedin depth estimator
    groundPlaneLast_ = nullptr;
    combined_groundPlaneLast_ = nullptr;
    //    // clear debug write
    //    std::stringstream ss;
    //    ss << "/tmp/gp.txt";
    //    std::ofstream file(ss.str().c_str());
    //    file.close();
}

void TrackletDepth::InitSubscriber(ros::NodeHandle& nh, bool use_semantics) {

    if(_params.subscriber_msg_name_3dgs!="")
    {
         ROS_INFO_STREAM("TrackletDepth: Subscribe to camera, lidar and 3dgs depth image");
        _subscriber_cloud =
            std::make_unique<SubscriberCloud>(nh, _params.subscriber_msg_name_cloud, _params.msg_queue_size);

        _subscriber_matches =
            std::make_unique<SubscriberTracklets>(nh, _params.subscriber_msg_name_tracklets, _params.msg_queue_size);

        _subscriber_camera_info =
            std::make_unique<SubscriberCameraInfo>(nh, _params.subscriber_msg_name_camera_info, _params.msg_queue_size);

        _subscriber_3dgs =
            std::make_unique<SubscriberSemantics>(nh, _params.subscriber_msg_name_3dgs, _params.msg_queue_size);

        _sync3 = std::make_unique<Synchronizer3>(Policy3(_params.msg_queue_size),
                                                 *_subscriber_cloud,
                                                 *_subscriber_matches,
                                                 *_subscriber_camera_info,
                                                 *_subscriber_3dgs);
        _sync3->registerCallback(boost::bind(&TrackletDepth::callback3dgs, this, _1, _2, _3,_4));
        
    }
    else if (!use_semantics) {
        ROS_INFO_STREAM("TrackletDepth: Subscribe to camera and lidar"<<_params.subscriber_msg_name_3dgs<<"---");
        _subscriber_cloud =
            std::make_unique<SubscriberCloud>(nh, _params.subscriber_msg_name_cloud, _params.msg_queue_size);

        _subscriber_matches =
            std::make_unique<SubscriberTracklets>(nh, _params.subscriber_msg_name_tracklets, _params.msg_queue_size);

        _subscriber_camera_info =
            std::make_unique<SubscriberCameraInfo>(nh, _params.subscriber_msg_name_camera_info, _params.msg_queue_size);

        _sync = std::make_unique<Synchronizer>(
            Policy(_params.msg_queue_size), *_subscriber_cloud, *_subscriber_matches, *_subscriber_camera_info);
        _sync->registerCallback(boost::bind(&TrackletDepth::callbackRansac, this, _1, _2, _3));

    } else {
        ROS_INFO_STREAM("TrackletDepth: Subscribe to camera, lidar and semantic image");
        _subscriber_cloud =
            std::make_unique<SubscriberCloud>(nh, _params.subscriber_msg_name_cloud, _params.msg_queue_size);

        _subscriber_matches =
            std::make_unique<SubscriberTracklets>(nh, _params.subscriber_msg_name_tracklets, _params.msg_queue_size);

        _subscriber_camera_info =
            std::make_unique<SubscriberCameraInfo>(nh, _params.subscriber_msg_name_camera_info, _params.msg_queue_size);

        _subscriber_semantics =
            std::make_unique<SubscriberSemantics>(nh, _params.subscriber_msg_name_semantics, _params.msg_queue_size);

        _sync2 = std::make_unique<Synchronizer2>(Policy2(_params.msg_queue_size),
                                                 *_subscriber_cloud,
                                                 *_subscriber_matches,
                                                 *_subscriber_camera_info,
                                                 *_subscriber_semantics);

        _sync2->registerCallback(boost::bind(&TrackletDepth::callbackSemantic, this, _1, _2, _3, _4));
    }
}

std::pair<int, int> TrackletDepth::ExractNewTrackletFrames(const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
                                                           std::vector<std::shared_ptr<TempTrackletFrame>>& newFrames) {
    int frameCountNew = 0, frameCountOld = 0;

    for (const auto& tracklet : tracklets_in->tracks) {
        int id = tracklet.id;

        // check if tracklet already exists in cache
        bool trackExists = _trackletMap.count(id);

        std::shared_ptr<TempTrackletFrame> tempFrame;

        if (!trackExists) {
            // add the last two features if tracklet is completely new
            const auto& matchNew = tracklet.feature_points.at(0);
            const auto& matchOld = tracklet.feature_points.at(1);

            tempFrame = std::make_unique<TempTrackletFrameExp>(
                id, std::make_pair(matchNew.u, matchNew.v), std::make_pair(matchOld.u, matchOld.v));

            frameCountNew++;
            frameCountOld++;
        } else {
            // add the newest feature if trcaklet already exists
            const auto& matchNew = tracklet.feature_points.at(0);

            tempFrame = std::make_unique<TempTrackletFrame>(id, std::make_pair(matchNew.u, matchNew.v));

            frameCountNew++;
        }

        newFrames.push_back(tempFrame);
    }

    return std::make_pair(frameCountNew, frameCountOld);
}

void TrackletDepth::CalculateFeatureDepthsCurFrame(const Cloud::ConstPtr& cloud_in_cur,
                                                   const std::vector<std::shared_ptr<TempTrackletFrame>>& newFrames,
                                                   const int frameCount,
                                                   Eigen::VectorXd& depths,
                                                   Mono_Lidar::GroundPlane::Ptr& ransacPlane) {
    // Convert the feature points to the interface format for the DepthEstimator
    depths.resize(frameCount);
    Eigen::Matrix2Xd featureCoordinates(2, frameCount);

    int i = 0;
    for (const auto& featureNew : newFrames) {
        // insert features of the current frame
        featureCoordinates(0, i) = featureNew->_feature.first;
        featureCoordinates(1, i) = featureNew->_feature.second;
        i++;
    }
    Eigen::VectorXi resultType;
    ROS_INFO_STREAM("Total points in cloud_in_cur: " << cloud_in_cur->points.size());
    this->_depthEstimator.CalculateDepth(cloud_in_cur, featureCoordinates, depths,resultType ,ransacPlane);
    std::map<int, int> failure_counts;
    for (int i = 0; i < resultType.size(); ++i) {
        failure_counts[resultType[i]]++;
    }

    // ROS_INFO("---CurFrame Depth Calculation Stats ---");
    // for (auto const& [reason_code, count] : failure_counts) {
    //     ROS_INFO("Reason Code %d: %d times", reason_code, count);
    // }
    // ROS_INFO("-----------------------------");
}


void TrackletDepth::CalculateFeatureDepthsLastFrame(const Cloud::ConstPtr& cloud_in_last,
                                                    const std::vector<std::shared_ptr<TempTrackletFrame>>& newFrames,
                                                    const int frameCount,
                                                    Eigen::VectorXd& depths,
                                                    Mono_Lidar::GroundPlane::Ptr& ransacPlaneOld) {
    // Convert the feature points to the interface format for the DepthEstimator
    depths.resize(frameCount);

    if (cloud_in_last == nullptr) {
        depths.setConstant(-1);
        return;
    }

    Eigen::Matrix2Xd featureCoordinates(2, frameCount);

    int i = 0;
    for (const auto& featureNew : newFrames) {
        // if the tracklet of the frame has been added this frame, there are exist two new features in successive frames
        const auto featureLast = std::dynamic_pointer_cast<TempTrackletFrameExp>(featureNew);

        if (featureLast != nullptr) {
            // feature from a new tracklet
            // Add feature of the last frame
            featureCoordinates(0, i) = featureLast->_featureLast.first;
            featureCoordinates(1, i) = featureLast->_featureLast.second;
            i++;
        }
    }



    Eigen::VectorXi resultType;
    
    this->_depthEstimator.CalculateDepth(cloud_in_last, featureCoordinates, depths,resultType, ransacPlaneOld);
    

    std::map<int, int> failure_counts;
    for (int i = 0; i < resultType.size(); ++i) {
        failure_counts[resultType[i]]++;
    }

    // ROS_INFO("---LastFrame Depth Calculation Stats ---");
    // for (auto const& [reason_code, count] : failure_counts) {
    //     ROS_INFO("Reason Code %d: %d times", reason_code, count);
    // }
    // ROS_INFO("-----------------------------");
}

std::pair<int, int> TrackletDepth::SaveFeatureDepths(const std::vector<std::shared_ptr<TempTrackletFrame>>& newFrames,
                                                     const Eigen::VectorXd& depthsLastFrame,
                                                     const Eigen::VectorXd& depthsCurFrame,
                                                     std::vector<TypeTrackletKey>& updatedIds) {
    int i = 0, j = 0;
    int newCount = 0;
    int oldCount = 0;

    for (const auto featureNew : newFrames) {
        int id = featureNew->_keyFrameId;

        const auto featureLast = std::dynamic_pointer_cast<TempTrackletFrameExp>(featureNew);

        if (featureLast != nullptr) {
            // New tracklet --> Create
            _trackletMap.emplace(id, feature_tracking::Tracklet());
            _trackletMap[id].id_ = id;
            _trackletMap[id].age_ = 0;

            int u = featureLast->_featureLast.first;
            int v = featureLast->_featureLast.second;

            auto match = feature_tracking::Match(u, v);
            // use world point to save feature's depth
            match.x_ = std::make_shared<feature_tracking::WorldPoint>();
            match.x_->data[2] = depthsLastFrame[j];
            _trackletMap[id].push_front(match);

            j++;
            newCount++;
        } else
            oldCount++;

        int u = featureNew->_feature.first;
        int v = featureNew->_feature.second;
        auto match = feature_tracking::Match(u, v);
        // use world point to save feature's depth temporarily
        match.x_ = std::make_shared<feature_tracking::WorldPoint>();
        match.x_->data[2] = depthsCurFrame[i];
        _trackletMap[id].push_front(match);

        i++;

        // mark tracklet to send for this frame
        updatedIds.push_back(id);
    }

    return std::make_pair(oldCount, newCount);
}

void TrackletDepth::callbackRansac(const Cloud::ConstPtr& cloud_in,
                                   const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
                                   const CameraInfo::ConstPtr& camInfo) {
    // Init plane ransac
    Mono_Lidar::GroundPlane::Ptr gp = std::make_shared<Mono_Lidar::RansacPlane>(
        std::make_shared<Mono_Lidar::DepthEstimatorParameters>(depth_estimator_parameters_));

    // Do estimation.
    process(cloud_in, tracklets_in, camInfo, gp);
}

void TrackletDepth::callbackSemantic(const Cloud::ConstPtr& cloud_in,
                                     const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
                                     const CameraInfo::ConstPtr& camInfo,
                                     const sensor_msgs::Image::ConstPtr& img) {

    // Init groundplane.
    image_geometry::PinholeCameraModel model;
    model.fromCameraInfo(camInfo);
    Mono_Lidar::SemanticPlane::Camera cam;
    cam.f = model.fx();
    cam.cu = model.cx();
    cam.cv = model.cy();
    cam.transform_cam_lidar = _camLidarTransform;

    cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::MONO8);
    std::set<int> gp_labels{6, 7, 8, 9};

    double plane_inlier_threshold = depth_estimator_parameters_.ransac_plane_refinement_treshold;
    Mono_Lidar::GroundPlane::Ptr gp =
        std::make_shared<Mono_Lidar::SemanticPlane>(img_ptr->image, cam, gp_labels, plane_inlier_threshold);
    ROS_DEBUG_STREAM("TrackletDepth: use semantics for ground segmentation");
    // Do estimation.
    process(cloud_in, tracklets_in, camInfo, gp);
}


// =========================================================================================
// =========================    改进后的 callback3dgs 函数    ===============================
// =========================================================================================
// void TrackletDepth::callback3dgs(const Cloud::ConstPtr& cloud_in,
//                                      const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
      
//                                const CameraInfo::ConstPtr& camInfo,
//                                      const sensor_msgs::Image::ConstPtr& img) {

//     // ========================= 阶段 1: 准备工作与建立激光雷达深度参考图 =========================
//     ROS_INFO("[callback3dgs] Phase 1: Creating LiDAR depth reference map.");

//     // --- 步骤 1.1: 初始化相机模型和定义参数 ---
//     image_geometry::PinholeCameraModel cam_model;
//     cam_model.fromCameraInfo(camInfo);

//     // 定义过滤参数 (建议作为ROS参数)
//     const int BORDER_WIDTH = 30;   
//     const int BORDER_WIDTH_TOP = 80;   

//     const int SEARCH_RADIUS_H = 20; // 图像空间搜索邻域的半径 (像素), 3 -> 7x7的窗口
//     const int SEARCH_RADIUS_W = 10;
//     const double DEPTH_DIFFERENCE_THRESHOLD = 0.1; // 深度差异阈值 (米)

//     const double MIN_DEPTH = 3; 
    
//     const double MAX_DEPTH = 20; 

//     // --- 步骤 1.2: 创建一个稀疏的激光雷达深度图 ---
//     // 初始化一个与深度图等大的Mat，用0表示无数据
//     cv::Mat lidar_depth_map = cv::Mat::zeros(img->height, img->width, CV_32FC1);

//     // 将原始雷达点云转换到相机坐标系
//     Cloud::Ptr cloud_in_camera_cs(new Cloud);
//     pcl::transformPointCloud(*cloud_in, *cloud_in_camera_cs, _camLidarTransform);

//     // --- 步骤 1.3: 将雷达点投影到图像平面，填充稀疏深度图 ---
//     for (const auto& pt_cam : cloud_in_camera_cs->points) {
//         if (pt_cam.z > 0.1) { // 只处理相机前方的点
//             cv::Point3d pt3d_cam(pt_cam.x, pt_cam.y, pt_cam.z);
//             cv::Point2d pt2d_img = cam_model.project3dToPixel(pt3d_cam);
             
//             int u = static_cast<int>(pt2d_img.x);
//             int v = static_cast<int>(pt2d_img.y);

//             // 检查投影点是否在图像有效范围内
//             if (u >= 0 && u < lidar_depth_map.cols && v >= 0 && v < lidar_depth_map.rows) {
//                 float current_depth = lidar_depth_map.at<float>(v, u);
//                 // 如果当前像素没有值，或者新点的深度更近（处理遮挡），则更新
//                 if (current_depth == 0.0f || pt_cam.z < current_depth) {
//                     lidar_depth_map.at<float>(v, u) = pt_cam.z;
//                 }
//             }
//         }
//     }
//     ROS_DEBUG("[callback3dgs] LiDAR reference map created.");

//     // ========================= 阶段 2: 过滤深度图生成的点云 =========================
//     ROS_INFO("[callback3dgs] Phase 2: Filtering dense cloud using LiDAR reference.");
    
//     // --- 步骤 2.1: 将ROS深度图消息转换为OpenCV Mat ---
//     cv_bridge::CvImageConstPtr cv_ptr;
//     try {
//         cv_ptr = cv_bridge::toCvShare(img);
//     } catch (const cv_bridge::Exception& e) {
//         ROS_ERROR("TrackletDepth::callback3dgs - cv_bridge exception: %s", e.what());
//         process(cloud_in, tracklets_in, camInfo, nullptr); // 转换失败则退回原始流程
//         return;
//     }
//     const cv::Mat& depth_image = cv_ptr->image;

//     // --- 步骤 2.2: 遍历深度图，进行验证和筛选 ---
//     Cloud::Ptr filtered_depth_cloud_camera_cs(new Cloud); // 存放筛选后点的容器
//     const double fx = cam_model.fx();
//     const double fy = cam_model.fy();
//     const double cx = cam_model.cx();
//     const double cy = cam_model.cy();
    
//     long points_raw = 0;
//     long points_kept = 0;

//     for (int v = 0; v < depth_image.rows; ++v) {
//         for (int u = 0; u < depth_image.cols; ++u) {


//              if (u < BORDER_WIDTH || u >= (depth_image.cols - BORDER_WIDTH) ||
//                 v < BORDER_WIDTH_TOP || v >= (depth_image.rows - BORDER_WIDTH)) {
//                 continue; // 在边缘区域，直接跳过此像素
//             }

//             // 获取深度值
//             float depth_value_m = 0.0f;
//             if (img->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
//                 depth_value_m = static_cast<float>(depth_image.at<uint16_t>(v, u)) * 0.001f;
//             } else if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
//                 depth_value_m = depth_image.at<float>(v, u);
//             } else {
//                 ROS_WARN_ONCE("Unsupported depth image encoding [%s].", img->encoding.c_str());
//                 goto end_filtering_loop;
//             }

//             if (depth_value_m <= MIN_DEPTH || depth_value_m >= MAX_DEPTH  ) continue; // 忽略无效深度
//             points_raw++;

//             // --- 核心过滤逻辑 ---
//             float min_lidar_depth_in_neighborhood = std::numeric_limits<float>::max();
//             float max_lidar_depth_in_neighborhood = std::numeric_limits<float>::min();
//             int lidar_point_found = 0;

//             // 在邻域内搜索有效的激光雷达深度参考
//             for (int dv = -SEARCH_RADIUS_H; dv <= SEARCH_RADIUS_H; ++dv) {
//                 for (int du = -SEARCH_RADIUS_W; du <= SEARCH_RADIUS_W; ++du) {
//                     int nu = u + du;
//                     int nv = v + dv;
//                     if (nu >= 0 && nu < lidar_depth_map.cols && nv >= 0 && nv < lidar_depth_map.rows) {
//                         float lidar_depth = lidar_depth_map.at<float>(nv, nu);
//                         if (lidar_depth > 0.0f) {
//                             lidar_point_found += 1;
//                             if (lidar_depth < min_lidar_depth_in_neighborhood) {
//                                 min_lidar_depth_in_neighborhood = lidar_depth;
//                             }
//                             if (lidar_depth > max_lidar_depth_in_neighborhood) {
//                                 max_lidar_depth_in_neighborhood = lidar_depth;
//                             }
//                         }
//                     }
//                 }
//             }

//             bool keep_point = true;
//             // ROS_INFO("[callback3dgs] lidar_point_found: %ld .", lidar_point_found);
//             if (lidar_point_found > 5) {

//                 if (depth_value_m < min_lidar_depth_in_neighborhood || depth_value_m > max_lidar_depth_in_neighborhood ) {
//                     keep_point = false; // 深度差异过大，丢弃
//                 }

//                 if (v < depth_image.rows/2 &&  depth_value_m > min_lidar_depth_in_neighborhood + DEPTH_DIFFERENCE_THRESHOLD)
//                 {
//                     keep_point = false; 
//                 }
//                 // if (max_lidar_depth_in_neighborhood - min_lidar_depth_in_neighborhood < 0.05)
//                 //     keep_point = false; 
//             }
//             else
//             {
//                 keep_point = false; 
//             }
//             // 如果没找到雷达点，keep_point 默认为 true，保留该点

//             if (keep_point) {
//                 points_kept++;
//                 Point new_point_camera_cs;
//                 new_point_camera_cs.z = depth_value_m;
//                 new_point_camera_cs.x = (static_cast<float>(u) - cx) * new_point_camera_cs.z / fx;
//                 new_point_camera_cs.y = (static_cast<float>(v) - cy) * new_point_camera_cs.z / fy;
//                 new_point_camera_cs.intensity = 100.0f; // 标记为来自深度图
//                 filtered_depth_cloud_camera_cs->points.push_back(new_point_camera_cs);
//             }
//         }
//     }
// end_filtering_loop:;
//     ROS_INFO("[callback3dgs] Filtering result: %ld / %ld dense points kept.", points_kept, points_raw);

//     // ========================= 阶段 3: 融合与最终处理 =========================
//     ROS_INFO("[callback3dgs] Phase 3: Merging clouds and proceeding to estimation.");
    
//     // --- 步骤 3.1: 将筛选后的深度点云转换回激光雷达坐标系 ---
//     Eigen::Affine3d transform_cam_to_lidar = _camLidarTransform.inverse();
//     Cloud::Ptr filtered_depth_cloud_lidar_cs(new Cloud);
//     pcl::transformPointCloud(*filtered_depth_cloud_camera_cs, *filtered_depth_cloud_lidar_cs, transform_cam_to_lidar);
    
//     // --- 步骤 3.2: 将原始激光雷达点云与筛选后的深度点云合并 ---
//     Cloud::Ptr combined_cloud_lidar_cs(new Cloud);
//     *combined_cloud_lidar_cs = *cloud_in; // 深度拷贝原始雷达点云
//     *combined_cloud_lidar_cs += *filtered_depth_cloud_lidar_cs; // 追加过滤后的稠密点

//     // 更新合并后点云的元数据
//     combined_cloud_lidar_cs->width = combined_cloud_lidar_cs->points.size();
//     combined_cloud_lidar_cs->height = 1;
//     combined_cloud_lidar_cs->is_dense = false;
    
//     ROS_INFO("[callback3dgs] Final combined cloud has %zu points.", combined_cloud_lidar_cs->points.size());

//     // --- 步骤 3.3: 初始化地面平面估算器并调用后续处理 ---
//     Mono_Lidar::GroundPlane::Ptr gp = std::make_shared<Mono_Lidar::RansacPlane>(
//         std::make_shared<Mono_Lidar::DepthEstimatorParameters>(depth_estimator_parameters_));
    
//     // 使用合并后的高质量点云和原始的特征点进行处理
//     process(combined_cloud_lidar_cs, tracklets_in, camInfo, gp);
// }



void TrackletDepth::callback3dgs(const Cloud::ConstPtr& cloud_in,
                                     const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
                                     const CameraInfo::ConstPtr& camInfo,
                                     const sensor_msgs::Image::ConstPtr& img_3dgs) {

    // ========================= 阶段 0: 初始化与准备 =========================
    ROS_INFO("--- [callback3dgs] New callback triggered. Starting two-pass depth estimation. ---");
    auto start_time_total = std::chrono::steady_clock::now();

    // 初始化相机模型（如果需要）
    if (!_isCameraInitialized) {
        this->InitCamera(camInfo);
        this->InitDepthEstimatorPost();
    }
    
    // 提取需要计算深度的新特征点帧
    std::vector<std::shared_ptr<TempTrackletFrame>> tempFrames;
    auto frameCount = ExractNewTrackletFrames(tracklets_in, tempFrames);
    int frameCountNew = frameCount.first;
    int frameCountOld = frameCount.second;

    if (frameCountNew == 0) {
        ROS_INFO("[callback3dgs] No new frames to process. Skipping.");
        return;
    }

    // ========================= 阶段 1: 第一次处理 (仅使用激光雷达) =========================
    ROS_INFO("[callback3dgs] Pass 1: Processing with LiDAR cloud only (%zu points).", cloud_in->points.size());
    
    Eigen::VectorXd depthsCurFrame_Lidar(frameCountNew), depthsLastFrame_Lidar(frameCountOld);
    Mono_Lidar::GroundPlane::Ptr groundPlane_Lidar = std::make_shared<Mono_Lidar::RansacPlane>(
        std::make_shared<Mono_Lidar::DepthEstimatorParameters>(depth_estimator_parameters_));
    Mono_Lidar::GroundPlane::Ptr groundPlaneLast_Lidar = groundPlaneLast_; // 使用上一帧的地面

    try {
        CalculateFeatureDepthsLastFrame(_cloud_last_frame, tempFrames, frameCountOld, depthsLastFrame_Lidar, groundPlaneLast_Lidar);
        CalculateFeatureDepthsCurFrame(cloud_in, tempFrames, frameCountNew, depthsCurFrame_Lidar, groundPlane_Lidar);
    } catch (const std::exception& e) {
        ROS_ERROR("[callback3dgs] Pass 1 Exception: %s. Setting LiDAR depths to invalid.", e.what());
        depthsLastFrame_Lidar.setConstant(-1.0);
        depthsCurFrame_Lidar.setConstant(-1.0);
    }
    
    // ========================= 阶段 2: 创建融合点云 (激光雷达 + 经过筛选的3DGS) =========================

    ROS_INFO("[callback3dgs] Pass 2 Prep: Creating combined cloud from LiDAR and filtered 3DGS depth.");

    // ---- START: 这部分融合逻辑可以基于您已有的代码进行优化 ----
    // 2.1 将3DGS深度图转换为OpenCV Mat
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img_3dgs);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("[callback3dgs] CV Bridge exception: %s. Cannot create combined cloud.", e.what());
        // 如果3DGS数据有问题，则退回到只使用激光雷达的结果
       
    }
    
    // 2.2 Unproject 3DGS depth image to a point cloud in CAMERA coordinate system
    image_geometry::PinholeCameraModel cam_model;
    cam_model.fromCameraInfo(camInfo);

    // 定义过滤参数 (建议作为ROS参数)
    const int BORDER_WIDTH = 30;   
    const int BORDER_WIDTH_TOP = 80;   

    const int SEARCH_RADIUS_H = 20; //  
    const int SEARCH_RADIUS_W = 6;
    const double DEPTH_DIFFERENCE_THRESHOLD = 0.3; // 深度差异阈值 (米)

    const double MIN_DEPTH = 3; 
    
    const double MAX_DEPTH = 20; 

    // --- 步骤 1.2: 创建一个稀疏的激光雷达深度图 ---
    // 初始化一个与深度图等大的Mat，用0表示无数据
    cv::Mat lidar_depth_map = cv::Mat::zeros(img_3dgs->height, img_3dgs->width, CV_32FC1);

    // 将原始雷达点云转换到相机坐标系
    Cloud::Ptr cloud_in_camera_cs(new Cloud);
    pcl::transformPointCloud(*cloud_in, *cloud_in_camera_cs, _camLidarTransform);

    // --- 步骤 1.3: 将雷达点投影到图像平面，填充稀疏深度图 ---
    for (const auto& pt_cam : cloud_in_camera_cs->points) {
        if (pt_cam.z > 0.1) { // 只处理相机前方的点
            cv::Point3d pt3d_cam(pt_cam.x, pt_cam.y, pt_cam.z);
            cv::Point2d pt2d_img = cam_model.project3dToPixel(pt3d_cam);
             
            int u = static_cast<int>(pt2d_img.x);
            int v = static_cast<int>(pt2d_img.y);

            // 检查投影点是否在图像有效范围内
            if (u >= 0 && u < lidar_depth_map.cols && v >= 0 && v < lidar_depth_map.rows) {
                float current_depth = lidar_depth_map.at<float>(v, u);
                // 如果当前像素没有值，或者新点的深度更近（处理遮挡），则更新
                if (current_depth == 0.0f || pt_cam.z < current_depth) {
                    lidar_depth_map.at<float>(v, u) = pt_cam.z;
                }
            }
        }
    }
    ROS_DEBUG("[callback3dgs] LiDAR reference map created.");

    // ========================= 阶段 2: 过滤深度图生成的点云 =========================
    ROS_INFO("[callback3dgs] Phase 2: Filtering dense cloud using LiDAR reference.");
    
    // --- 步骤 2.1: 将ROS深度图消息转换为OpenCV Mat ---
    const cv::Mat& depth_image = cv_ptr->image;

    // --- 步骤 2.2: 遍历深度图，进行验证和筛选 ---
    Cloud::Ptr filtered_depth_cloud_camera_cs(new Cloud); // 存放筛选后点的容器
    const double fx = cam_model.fx();
    const double fy = cam_model.fy();
    const double cx = cam_model.cx();
    const double cy = cam_model.cy();
    
    long points_raw = 0;
    long points_kept = 0;

    for (int v = 0; v < depth_image.rows; ++v) {
        for (int u = 0; u < depth_image.cols; ++u) {


             if (u < BORDER_WIDTH || u >= (depth_image.cols - BORDER_WIDTH) ||
                v < BORDER_WIDTH_TOP || v >= (depth_image.rows - BORDER_WIDTH)) {
                continue; // 在边缘区域，直接跳过此像素
            }

            // 获取深度值
            float depth_value_m = 0.0f;
            if (img_3dgs->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                depth_value_m = static_cast<float>(depth_image.at<uint16_t>(v, u)) * 0.001f;
            } else if (img_3dgs->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                depth_value_m = depth_image.at<float>(v, u);
            } else {
                ROS_WARN_ONCE("Unsupported depth image encoding [%s].", img_3dgs->encoding.c_str());
                goto end_filtering_loop;
            }

            if (depth_value_m <= MIN_DEPTH || depth_value_m >= MAX_DEPTH  ) continue; // 忽略无效深度
            points_raw++;

            // --- 核心过滤逻辑 ---
            float min_lidar_depth_in_neighborhood = std::numeric_limits<float>::max();
            float max_lidar_depth_in_neighborhood = std::numeric_limits<float>::min();
            int lidar_point_found = 0;

            // 在邻域内搜索有效的激光雷达深度参考
            for (int dv = -SEARCH_RADIUS_H; dv <= SEARCH_RADIUS_H; ++dv) {
                for (int du = -SEARCH_RADIUS_W; du <= SEARCH_RADIUS_W; ++du) {
                    int nu = u + du;
                    int nv = v + dv;
                    if (nu >= 0 && nu < lidar_depth_map.cols && nv >= 0 && nv < lidar_depth_map.rows) {
                        float lidar_depth = lidar_depth_map.at<float>(nv, nu);
                        if (lidar_depth > 0.0f) {
                            lidar_point_found += 1;
                            if (lidar_depth < min_lidar_depth_in_neighborhood) {
                                min_lidar_depth_in_neighborhood = lidar_depth;
                            }
                            if (lidar_depth > max_lidar_depth_in_neighborhood) {
                                max_lidar_depth_in_neighborhood = lidar_depth;
                            }
                        }
                    }
                }
            }

            bool keep_point = true;
            // ROS_INFO("[callback3dgs] lidar_point_found: %ld .", lidar_point_found);
            if (lidar_point_found > 5) {

                if (depth_value_m < min_lidar_depth_in_neighborhood || depth_value_m > max_lidar_depth_in_neighborhood ) {
                    keep_point = false; // 深度差异过大，丢弃
                }

                if (v < depth_image.rows/2 &&  depth_value_m > min_lidar_depth_in_neighborhood + DEPTH_DIFFERENCE_THRESHOLD)
                {
                    keep_point = false; 
                }
                // if (max_lidar_depth_in_neighborhood - min_lidar_depth_in_neighborhood < 0.05)
                //     keep_point = false; 
            }
            else
            {
                keep_point = false; 
            }
            // 如果没找到雷达点，keep_point 默认为 true，保留该点

            if (keep_point) {
                points_kept++;
                Point new_point_camera_cs;
                new_point_camera_cs.z = depth_value_m;
                new_point_camera_cs.x = (static_cast<float>(u) - cx) * new_point_camera_cs.z / fx;
                new_point_camera_cs.y = (static_cast<float>(v) - cy) * new_point_camera_cs.z / fy;
                new_point_camera_cs.intensity = 100.0f; // 标记为来自深度图
                filtered_depth_cloud_camera_cs->points.push_back(new_point_camera_cs);
            }
        }
    }
end_filtering_loop:;
    ROS_INFO("[callback3dgs] Filtering result: %ld / %ld dense points kept.", points_kept, points_raw);
   
    // 2.3 Transform 3DGS cloud from CAMERA to LIDAR coordinate system
    Cloud::Ptr filtered_depth_cloud_lidar_cs(new Cloud);
    pcl::transformPointCloud(*filtered_depth_cloud_camera_cs, *filtered_depth_cloud_lidar_cs, _camLidarTransform.inverse());
 
    // 2.4 Combine LiDAR cloud with the new 3DGS cloud
    Cloud::Ptr combined_cloud(new Cloud(*cloud_in)); // Start with a copy of LiDAR cloud
    *combined_cloud += *filtered_depth_cloud_lidar_cs;
    ROS_INFO("[callback3dgs] Combined cloud created with %zu points.", combined_cloud->points.size());
    // ---- END: 融合逻辑 ----
    
    // ========================= 阶段 3: 第二次处理 (使用融合点云) =========================
    ROS_INFO("[callback3dgs] Pass 2: Processing with combined cloud.");

    Eigen::VectorXd depthsCurFrame_Combined(frameCountNew), depthsLastFrame_Combined(frameCountOld);
    Mono_Lidar::GroundPlane::Ptr groundPlane_Combined = std::make_shared<Mono_Lidar::RansacPlane>(
        std::make_shared<Mono_Lidar::DepthEstimatorParameters>(depth_estimator_parameters_));
    Mono_Lidar::GroundPlane::Ptr groundPlaneLast_Combined = combined_groundPlaneLast_; 
    
    try {
        CalculateFeatureDepthsLastFrame(_combined_cloud_last_frame, tempFrames, frameCountOld, depthsLastFrame_Combined, groundPlaneLast_Combined);
        CalculateFeatureDepthsCurFrame(combined_cloud, tempFrames, frameCountNew, depthsCurFrame_Combined, groundPlane_Combined);
    } catch (const std::exception& e) {
        ROS_ERROR("[callback3dgs] Pass 2 Exception: %s. Setting combined depths to invalid.", e.what());
        depthsLastFrame_Combined.setConstant(-1.0);
        depthsCurFrame_Combined.setConstant(-1.0);
    }
    
    // ========================= 阶段 4: 合并结果并保存 =========================

    ROS_INFO("[callback3dgs] Merging results from both passes and saving.");

    Eigen::VectorXd depthsCurFrame_Final(frameCountNew), depthsLastFrame_Final(frameCountOld);
    int supplemented_cur = 0, supplemented_last = 0;

    // 合并当前帧的深度结果
    for (int i = 0; i < frameCountNew; ++i) {
        if (depthsCurFrame_Lidar(i) >= 0) {
            depthsCurFrame_Final(i) = depthsCurFrame_Lidar(i);
        } else if (depthsCurFrame_Combined(i) >= 0) {
            // 激光雷达失败，但融合点云成功，进行补充
            depthsCurFrame_Final(i) = -std::abs(depthsCurFrame_Combined(i)); // 设为负值以区分来源
            supplemented_cur++;
        } else {
            depthsCurFrame_Final(i) = -1.0; // 两者都失败
        }
    }

    // 合并上一帧的深度结果
    for (int i = 0; i < frameCountOld; ++i) {
        if (depthsLastFrame_Lidar(i) >= 0) {
            depthsLastFrame_Final(i) = depthsLastFrame_Lidar(i);
        } else if (depthsLastFrame_Combined(i) >= 0) {
            // 激光雷达失败，但融合点云成功，进行补充
            depthsLastFrame_Final(i) = -std::abs(depthsLastFrame_Combined(i)); // 设为负值以区分来源
            supplemented_last++;
        } else {
            depthsLastFrame_Final(i) = -1.0; // 两者都失败
        }
    }
    
    ROS_INFO("[callback3dgs] Supplementation stats | Current Frame: %d points | Last Frame: %d points.", supplemented_cur, supplemented_last);

    // 使用最终合并的深度结果保存和发布
    std::vector<TypeTrackletKey> updatedIds;
    SaveFeatureDepths(tempFrames, depthsLastFrame_Final, depthsCurFrame_Final, updatedIds);

    // 更新地面和点云缓存
    groundPlaneLast_ =  groundPlane_Lidar;
    combined_groundPlaneLast_ = groundPlane_Combined;
    
    _cloud_last_frame = cloud_in;
    _combined_cloud_last_frame = combined_cloud;

    // 发布结果
    matches_msg_depth_ros::MatchesMsg msgOut;
    convert_tracklets_to_matches_msg(tracklets_in, updatedIds, msgOut);
    _publisher_matches.publish(msgOut);


    if(_params.publisher_msg_name_image_projection_cloud!=""){
        PublishImageProjectionCloud(camInfo->width, camInfo->height);
    }

    if(_params.publisher_msg_name_cloud_camera_cs!=""){
        PublishPointCloudCameraCs();
    }


    // 清理工作
    TidyUpTracklets(updatedIds);
    TidyUpTimeStamps();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_total).count();
    ROS_INFO("[callback3dgs] Total processing time: %ld ms.", duration_ms);
}



void TrackletDepth::process(const Cloud::ConstPtr& cloud_in,
                            const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
                            const CameraInfo::ConstPtr& camInfo,
                            Mono_Lidar::GroundPlane::Ptr groundPlaneCur) {
    if (cloud_in->points.size() < 100) {
        ROS_WARN_STREAM("In tracklet_depth: received less than 100 points.");
    }

    // Timer
    auto start_time = std::chrono::steady_clock::now();

    std::stringstream stats;

    _msgCount++;

    _timestamps.push_front(tracklets_in->header.stamp);

    // Initialize depthestimator it not done
    if (!_isDepthEstimatorInitialized) {
        this->InitCamera(camInfo);
        this->InitDepthEstimatorPost();
    }

    double sec = tracklets_in->header.stamp.toSec();
    stats << "Received tracklet-lidar pair at: " << sec << std::endl;
    stats << "Total messages recieved: " << _msgCount << std::endl;
    stats << "Tracklet length: " << tracklets_in->tracks.size() << std::endl;

    // Save all new feature points of the incoming tracklets which haven't been estimated in previous frames
    std::vector<std::shared_ptr<TempTrackletFrame>> tempFrames;
    auto frameCount = ExractNewTrackletFrames(tracklets_in, tempFrames);
    int frameCountNew = frameCount.first;
    int frameCountOld = frameCount.second;

    Eigen::VectorXd depthsCurFrame, depthsLastFrame;

    try {
        // Use the DespthEstiamtor to calculate the depth all newly arrived feature points
        CalculateFeatureDepthsLastFrame(
            _cloud_last_frame, tempFrames, frameCountOld, depthsLastFrame, groundPlaneLast_);
    } catch (const Mono_Lidar::GroundPlane::ExceptionPclInvalid& e) {
        ROS_ERROR_STREAM(e.what());
        ROS_WARN_STREAM("TrackletDepth: Old frame continue with invalid depths");

        depthsLastFrame.resize(frameCountOld);
        depthsLastFrame.setConstant(-1);
    }
    try {
        CalculateFeatureDepthsCurFrame(cloud_in, tempFrames, frameCountNew, depthsCurFrame, groundPlaneCur);
        if (groundPlaneCur == nullptr) {
            ROS_ERROR_STREAM("TrackletDepth: Plane not calculated");
        }

        // remember cloud
        _cloud_last_frame = cloud_in;
    } catch (const Mono_Lidar::GroundPlane::ExceptionPclInvalid& e) {
        ROS_ERROR_STREAM(e.what());
        ROS_WARN_STREAM("TrackletDepth: Cur frame continue with invalid depths");

        depthsCurFrame.resize(frameCountNew);
        depthsCurFrame.setConstant(-1);

        // mark plane invalid so it will be recalculated next time.
        groundPlaneCur = nullptr;

        // declare cloud invalid
        _cloud_last_frame = nullptr;
    }

    // remember the ransac plane
    groundPlaneLast_ = groundPlaneCur;

    // Write Results
    std::vector<TypeTrackletKey> updatetIds;
    auto trackletsCount = SaveFeatureDepths(tempFrames, depthsLastFrame, depthsCurFrame, updatetIds);

    stats << "Old tracklets Count: " << trackletsCount.first << std::endl;
    stats << "New tracklets Count: " << trackletsCount.second << std::endl;

    // Convert newly updated tracklets to the msg format and publish
    matches_msg_depth_ros::MatchesMsg msgOut;
    auto matchesSuccess = convert_tracklets_to_matches_msg(tracklets_in, updatetIds, msgOut);
    // msgOut.header.stamp = tracklets_in->header.stamp;
    _publisher_matches.publish(msgOut);

    if(_params.publisher_msg_name_image_projection_cloud!=""){
        PublishImageProjectionCloud(camInfo->width, camInfo->height);
    }

    if(_params.publisher_msg_name_cloud_camera_cs!=""){
        PublishPointCloudCameraCs();
    }

    stats << "Feature Estimation success count: " << matchesSuccess.first << std::endl;
    stats << "Feature Estimation fail count: " << matchesSuccess.second << std::endl;

    // Delete old tracklets (tracklets with no updates in the current frame)
    TidyUpTracklets(updatetIds);
    TidyUpTimeStamps();


    //    if (groundPlaneLast_ != nullptr) {
    //        std::stringstream ss;
    //        ss << "/tmp/gp.txt";
    //        std::ofstream file(ss.str().c_str(), std::ios_base::app);
    //        file.precision(12);
    //        Eigen::Vector4f plane_params = groundPlaneLast_->getModelCoeffs();
    //        file << plane_params[0] << " " << plane_params[1] << " " << plane_params[2] << " " << plane_params[3]
    //             << std::endl;
    //        file.close();
    //    }

    ROS_DEBUG_STREAM("TrackletDepthRosTool: " + stats.str());
    ROS_INFO_STREAM(
        "Duration tracklet_depth_ros_tool="
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count()
        << " ms");
}


void TrackletDepth::TidyUpTracklets(const std::vector<TypeTrackletKey>& updatedIds) {
    std::vector<TypeTrackletKey> toDelete;

    for (const auto& keyAvailable : _trackletMap){
        toDelete.push_back(keyAvailable.first);
    }
    
    for (const auto& keyStay : updatedIds) {
        for (int i = 0; i < int(toDelete.size()); i++) {
            if (toDelete.at(i) == keyStay) {
                toDelete.erase(toDelete.begin() + i);
                break;
            }
        }
    }

    // delete old tracklets which haven't been updated this frame
    for (const auto id : toDelete) {
        auto it = _trackletMap.find(id);
        _trackletMap.erase(it);
    }
}

void TrackletDepth::TidyUpTimeStamps() {
    int maxLength = 0;

    for (const auto& tracklet : _trackletMap) {
        if (int(tracklet.second.size()) > maxLength) {
            maxLength = tracklet.second.size();
        }
    }

    while (int(_timestamps.size()) > maxLength) {
        _timestamps.pop_back();
    }
}

std::pair<int, int> TrackletDepth::convert_tracklets_to_matches_msg(
    const matches_msg_ros::MatchesMsgConstPtr& tracklets_in,
    const std::vector<TypeTrackletKey>& trackletIds,
    matches_msg_depth_ros::MatchesMsg& Out) {
    using MsgFeature = matches_msg_depth_ros::FeaturePoint;
    using MsgTracklet = matches_msg_depth_ros::Tracklet;

    Out.tracks.reserve(trackletIds.size());

    int success = 0;
    int failed = 0;

    // Insert tracklets to new message
    for (const auto trackletId : trackletIds) {
        const auto& curTracklet = _trackletMap[trackletId];
        MsgTracklet t;

        t.id = curTracklet.id_;
        t.age = curTracklet.age_;
        t.feature_points.reserve(curTracklet.size());

        for (const auto& cur_match : curTracklet) {
            const auto& cur_point = cur_match.p1_;
            const float depth = cur_match.x_->data[2];
            MsgFeature msg_feature;

            depth >= 0 ? success++ : failed++;

            msg_feature.u = float(cur_point.u_);
            msg_feature.v = float(cur_point.v_);
            msg_feature.d = depth;

            t.feature_points.push_back(std::move(msg_feature));
        }

        assert(t.feature_points.size() == curTracklet.size());
        Out.tracks.push_back(std::move(t));
    }

    assert(Out.tracks.size() == trackletIds.size());

    // Insert timestamps
    for (const auto& stamp : tracklets_in->stamps) {
        Out.stamps.push_back(stamp);
    }

    // Init header of new message
    Out.header = tracklets_in->header;

    return std::make_pair(success, failed);
}


void TrackletDepth::InitPublisher(ros::NodeHandle& nh) {
    // Pusblisher
    std::cout << "Pusblisher name pointcloud camera cs: " << _params.publisher_msg_name_cloud_camera_cs << std::endl;
    if(_params.publisher_msg_name_cloud_camera_cs!=""){
    _publisher_cloud_camera_cs =
        nh.advertise<Cloud>(_params.publisher_msg_name_cloud_camera_cs, _params.msg_queue_size);
    }
    std::cout << "Publisher name pointcloud interpolated: " << _params.publisher_msg_name_cloud_interpolated
              << std::endl;
    if(_params.publisher_msg_name_cloud_interpolated!=""){
    _publisher_cloud_interpolated =
        nh.advertise<Cloud>(_params.publisher_msg_name_cloud_interpolated, _params.msg_queue_size);
    }
    std::cout << "Publisher name image projection cloud: " << _params.publisher_msg_name_image_projection_cloud
              << std::endl;
    if(_params.publisher_msg_name_image_projection_cloud!=""){
    _publisher_image_projection_cloud =
        _image_transport.advertise(_params.publisher_msg_name_image_projection_cloud, _params.msg_queue_size);
    }
    std::cout << "Publisher name calc depth stats: " << _params.publisher_msg_name_depthcalc_stats << std::endl;
    if(_params.publisher_msg_name_depthcalc_stats!=""){
    _publisher_depthcalc_stats =
        nh.advertise<std_msgs::String>(_params.publisher_msg_name_depthcalc_stats, _params.msg_queue_size);
    }
    
    std::cout << "Publisher name tracklets depth with depth: " << _params.publisher_msg_name_tracklets_depth
              << std::endl;
    _publisher_matches = nh.advertise<matches_msg_depth_ros::MatchesMsg>(_params.publisher_msg_name_tracklets_depth,
                                                                         _params.msg_queue_size);
}

bool TrackletDepth::InitDepthEstimatorPre() {
    std::cout << "Initialize DepthEstimator Pre" << std::endl;

    if (_path_config_depthEstimator == "") {
        std::cout << "No config file for DepthEstimator set. Loading from parameter server." << std::endl;

        if (!_depthEstimator.InitConfig())
            throw "Error in 'initConfig' of DepthEstimator.";
    } else {
        std::cout << "Config file DepthEstimator: " << _path_config_depthEstimator << std::endl;
    }

    if (!_depthEstimator.InitConfig(_path_config_depthEstimator))
        throw "Error in 'initConfig' of DepthEstimator.";

    return true;
}

bool TrackletDepth::InitDepthEstimatorPost() {
    std::cout << "Initialize DepthEstimator Post" << std::endl;

    if (!_depthEstimator.Initialize(_camera, _camLidarTransform))
        throw "Error in 'Initialize' of DepthEstimator.";

    std::cout << "DepthEstimator successful initialized" << std::endl;

    _isDepthEstimatorInitialized = true;

    return true;
}


bool TrackletDepth::InitStaticTransforms() {
    using namespace std;

    tf::TransformListener listener;
    tf::StampedTransform transform;
    Eigen::Affine3d transformEigen;

    bool tf_success = false;

    string targetFrame = _params.tf_frame_name_cameraLeft; // target frame to which the coordinates will be transformed
    string sourceFrame = _params.tf_frame_name_velodyne;   // frame in which the coordinates are currently in

    ROS_INFO_STREAM("Waiting for the ROSBAG to start");

    while (!tf_success) {
        try {
            // ros::spinOnce();

            if (!listener.frameExists(targetFrame)) {
                // ROS_INFO_STREAM("waiting for target frame "<< targetFrame);
                // continue;
            }

            if (!listener.frameExists(sourceFrame)) {
                // ROS_INFO_STREAM("waiting for source frame "<< sourceFrame);
                // continue;
            }
            listener.lookupTransform(targetFrame, sourceFrame, ros::Time(0), transform);
            tf_success = true;
        } catch (tf::TransformException& ex) {
            ROS_ERROR("%s", ex.what());
            ros::Duration(0.5).sleep();
        }
    }

    // Apply additional camera transformation (correction)
    tf::transformTFToEigen(transform, transformEigen);

    _camLidarTransform = transformEigen;

    ROS_INFO_STREAM("Got the tf transformations: " << endl << _camLidarTransform.matrix());

    return true;
}


void TrackletDepth::InitCamera(const CameraInfo::ConstPtr& cam_info) {
    // Check preconditions
    if (_isCameraInitialized)
        return;

    image_geometry::PinholeCameraModel model;
    model.fromCameraInfo(cam_info);
    assert(model.fx() == model.fy()); // we only support undistorted images

    // Extract camera parameters from calibration matrix
    double focalLengthX = model.fx();
    double principlePointX = model.cx();
    double principlePointY = model.cy();

    // Create camera model
    _camera = std::make_shared<CameraPinhole>(
        cam_info->width, cam_info->height, focalLengthX, principlePointX, principlePointY);

    _isCameraInitialized = true;

    // debug
    ROS_INFO_STREAM("Got the camera info:" << std::endl
                                           << "Got the camera info:"
                                           << std::endl
                                           << "img_width: "
                                           << cam_info->width
                                           << std::endl
                                           << "img_height: "
                                           << cam_info->height
                                           << std::endl
                                           << "focal_length_x: "
                                           << focalLengthX
                                           << std::endl
                                           << "principalPointX: "
                                           << principlePointX
                                           << std::endl
                                           << "principalPointY: "
                                           << principlePointY
                                           << std::endl);
}

void TrackletDepth::PublishPointCloudCameraCs() {
    std::cout << "Publich pointcloud camera cs" << std::endl;
    Cloud::Ptr cloud_camera_cs = {boost::make_shared<Cloud>()};

    _depthEstimator.getCloudCameraCs(cloud_camera_cs);
    cloud_camera_cs->header.frame_id = _params.tf_frame_name_cameraLeft;
    cloud_camera_cs->header.stamp = ros::Time::now().toNSec()/ 1000;
    // cloud_camera_cs->header.stamp = ros::Time::now().toNSec(); //qlsqls
    _publisher_cloud_camera_cs.publish(cloud_camera_cs);
}

void TrackletDepth::PublishPointCloudInterpolated() {
    std::cout << "Publish pointcloud interpolated" << std::endl;
    Cloud::Ptr cloud_interpolated = {boost::make_shared<Cloud>()};

    _depthEstimator.getCloudInterpolated(cloud_interpolated);
    cloud_interpolated->header.frame_id = _params.tf_frame_name_cameraLeft;
    cloud_interpolated->header.stamp = ros::Time::now().toNSec();
    _publisher_cloud_interpolated.publish(cloud_interpolated);
}


void TrackletDepth::PublishImageProjectionCloud(int imgWidth, int imgHeight) {
    std::cout << "Publish image projection cloud" << std::endl;

    Eigen::Matrix2Xd points;
    _depthEstimator.getPointsCloudImageCs(points);

    cv::Mat img = cv::Mat(imgHeight, imgWidth, CV_8UC1);
    // Init image with white color
    img.setTo(255);

    for (int i = 0; i < points.cols(); i++) {
        uchar grayValue = (uchar)(0);
        int x = (int)points(0, i);
        int y = (int)points(1, i);
        img.at<uchar>(y, x) = grayValue;
    }

    cv::cvtColor(img, img, CV_GRAY2RGB);

    // publish image
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    _publisher_image_projection_cloud.publish(msg);
}


void TrackletDepth::PublishDepthCalcStats() {
    // get statistics
    auto stats = this->_depthEstimator.getDepthCalcStats();
    int pointCount = stats.getPointCount();
    int pointsSuccess = stats.getSuccess();
    int radiusSearchInsufficientPoints = stats.getRadiusSearchInsufficientPoints();
    int histogramNoLocalMax = stats.getHistogramNoLocalMax();
    int tresholdDepthGlobalGreaterMax = stats.getTresholdDepthGlobalGreaterMax();
    int tresholdDepthGlobalSmallerMin = stats.getTresholdDepthGlobalSmallerMin();
    int tresholdDepthLocalGreaterMax = stats.getTresholdDepthLocalGreaterMax();
    int tresholdDepthLocalSmallerMin = stats.getTresholdDepthLocalSmallerMin();
    //    int insufficientRoadPoints = stats.getInsufficientRoadPoints();
    int pcaIsCubic = stats.getPCAIsCubic();
    int pcaIsLine = stats.getPCAIsLine();
    int pcaIsPoint = stats.getPCAIsPoint();
    int triangleNotPlanar = stats.getTriangleNotPlanar();
    int pointsBehindCamera = stats.getCornerBehindCamera();
    int notPlanarInsufficientPoints = stats.getTriangleNotPlanarInsufficientPoints();
    int viewrayPlaneNotOrthogonal = stats.getPlaneViewrayNotOrthogonal();

    // publish statistics as a string
    std_msgs::String msg;
    std::stringstream ss;

    using namespace std;
    ss << "Depth calculation point statistics:" << endl;
    ss << "Points Count: " << pointCount << endl;
    ss << "Depth Calc Success count: " << pointsSuccess << endl;
    ss << "Radius Search Insufficient points: " << radiusSearchInsufficientPoints << endl;
    ss << "Histogram no Local max: " << histogramNoLocalMax << endl;
    ss << "Global Treshold Depth Greater Max: " << tresholdDepthGlobalGreaterMax << endl;
    ss << "Global Treshold Depth Smaller min: " << tresholdDepthGlobalSmallerMin << endl;
    ss << "Local" << tresholdDepthLocalGreaterMax << endl;
    ss << "Local Treshold Depth Smaller min: " << tresholdDepthLocalSmallerMin << endl;
    ss << "Triangle not planar: " << triangleNotPlanar << endl;
    ss << "Triangle not planar insuficien points: " << notPlanarInsufficientPoints << endl;
    ss << "PCA is cubic: " << pcaIsCubic << endl;
    ss << "PCA is line: " << pcaIsLine << endl;
    ss << "PCA is point: " << pcaIsPoint << endl;
    ss << "ViewRay plane not orthogonal: " << viewrayPlaneNotOrthogonal << endl;
    ss << "Points interpolated behind camera: " << pointsBehindCamera << endl;

    msg.data = ss.str();
    _publisher_depthcalc_stats.publish(msg);
}
}
