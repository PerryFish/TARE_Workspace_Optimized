#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/polygon_stamped.h>
#include <geometry_msgs/msg/point_stamped.h>

#include "tf2/transform_datatypes.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>

#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "rmw/types.h"
#include "rmw/qos_profiles.h"

using namespace std;

const double PI = 3.1415926;

string metricFile;
string trajFile;
string mapFile;
double overallMapVoxelSize = 0.5;
double exploredAreaVoxelSize = 0.3;
double exploredVolumeVoxelSize = 0.5;
double transInterval = 0.2;
double yawInterval = 10.0;
int overallMapDisplayInterval = 2;
int overallMapDisplayCount = 0;
int exploredAreaDisplayInterval = 1;
int exploredAreaDisplayCount = 0;

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZ>::Ptr overallMapCloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr overallMapCloudDwz(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZI>::Ptr exploredAreaCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr exploredAreaCloud2(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr exploredVolumeCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr exploredVolumeCloud2(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr trajectory(new pcl::PointCloud<pcl::PointXYZI>());

const int systemDelay = 5;
int systemDelayCount = 0;
bool systemDelayInited = false;
double systemTime = 0;
double systemInitTime = 0;
bool systemInited = false;

float vehicleYaw = 0;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;
float exploredVolume = 0, travelingDis = 0, runtime = 0, timeDuration = 0;

pcl::VoxelGrid<pcl::PointXYZ> overallMapDwzFilter;
pcl::VoxelGrid<pcl::PointXYZI> exploredAreaDwzFilter;
pcl::VoxelGrid<pcl::PointXYZI> exploredVolumeDwzFilter;

sensor_msgs::msg::PointCloud2 overallMap2;

shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pubExploredAreaPtr;

shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pubTrajectoryPtr;

shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> pubExploredVolumePtr;

shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> pubTravelingDisPtr;

shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> pubTimeDurationPtr;

FILE *metricFilePtr = NULL;
FILE *trajFilePtr = NULL;

// ============================================================
// 全局地图保存相关变量
// ============================================================
pcl::PointCloud<pcl::PointXYZI>::Ptr globalMapAccumulator(
    new pcl::PointCloud<pcl::PointXYZI>());

bool exploration_finished_flag = false;
bool map_saved_flag = false;

// 保存路径：修改这个为你想要的输出目录
const std::string kDefaultSaveDir = "/home/nuaa/ZHY/TARE/src/tare_planner/src/tare_planner/map_saved/";

// ============================================================
// 辅助函数：生成带时间戳的文件名（带安全检查）
// ============================================================
std::string generateTimestampedFilename(const std::string& dir,
                                        const std::string& prefix,
                                        const std::string& extension) {
  std::time_t now = std::time(nullptr);
  std::tm* ltm = std::localtime(&now);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", ltm);
  std::string path = dir.empty() ? prefix + "_" + buf + extension
                                  : dir + "/" + prefix + "_" + buf + extension;
  return path;
}

// ============================================================
// 核心保存函数：将累计地图保存为 PCD 和 PLY 两种格式
// （含完整的异常保护和防重复保存）
// ============================================================
bool saveGlobalMap(const std::string& save_dir,
                   const std::string& tag,
                   rclcpp::Node::SharedPtr node) {
  if (globalMapAccumulator->empty()) {
    RCLCPP_WARN(node->get_logger(),
                "[MapSaver] globalMapAccumulator is empty, nothing to save.");
    return false;
  }

  // 安全：确保目录非空
  std::string safe_dir = save_dir.empty()
                            ? "/tmp/tare_saved_maps"
                            : save_dir;

  // 安全：确保输出目录存在（带异常保护）
  try {
    if (!std::filesystem::exists(safe_dir)) {
      std::filesystem::create_directories(safe_dir);
      RCLCPP_INFO(node->get_logger(),
                  "[MapSaver] Created save directory: %s", safe_dir.c_str());
    }
  } catch (const std::filesystem::filesystem_error& e) {
    RCLCPP_ERROR(node->get_logger(),
                 "[MapSaver] Failed to create directory '%s': %s",
                 safe_dir.c_str(), e.what());
    return false;
  }

  // 生成文件名
  std::string pcd_file = generateTimestampedFilename(safe_dir,
                                                      "map_" + tag, ".pcd");
  std::string ply_file = generateTimestampedFilename(safe_dir,
                                                      "map_" + tag, ".ply");

  // 保存 PCD（ASCII 格式）
  try {
    int ret = pcl::io::savePCDFileASCII(pcd_file, *globalMapAccumulator);
    if (ret == 0) {
      RCLCPP_INFO(node->get_logger(),
                  "[MapSaver] PCD saved to: %s  (points: %zu)",
                  pcd_file.c_str(), globalMapAccumulator->size());
    } else {
      RCLCPP_ERROR(node->get_logger(),
                   "[MapSaver] Failed to save PCD (error code: %d)", ret);
      return false;
    }
  } catch (const pcl::IOException& e) {
    RCLCPP_ERROR(node->get_logger(),
                 "[MapSaver] PCD IOException: %s", e.what());
    return false;
  }

  // 保存 PLY（方便用 MeshLab 等工具查看）
  try {
    pcl::io::savePLYFileASCII(ply_file, *globalMapAccumulator);
    RCLCPP_INFO(node->get_logger(),
                "[MapSaver] PLY saved to: %s", ply_file.c_str());
  } catch (const pcl::IOException& e) {
    // PLY 保存失败不影响主流程，打个警告即可
    RCLCPP_WARN(node->get_logger(),
                "[MapSaver] PLY save failed (non-critical): %s", e.what());
  }

  return true;
}

// ============================================================
// Handler 1: 订阅 /registered_scan（每帧激光点云）
// 累计存入 globalMapAccumulator（含降采样防内存爆涨）
// ============================================================
const float GLOBAL_MAP_VOXEL_SIZE = 0.15f;  // 米，可调整：0.1 精密 / 0.2 快速

void laserCloudForSaveHandler(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloudIn) {

  if (!systemDelayInited) {
    systemDelayCount++;
    if (systemDelayCount > systemDelay) {
      systemInited = true;
    }
    return;
  }

  if (!systemInited) return;

  pcl::PointCloud<pcl::PointXYZI> tempCloud;
  pcl::fromROSMsg(*laserCloudIn, tempCloud);

  // 降采样：只保留 voxel 内的代表点，大幅降低存储压力
  static pcl::VoxelGrid<pcl::PointXYZI> voxelFilter;
  static bool voxelFilterInit = false;
  if (!voxelFilterInit) {
    voxelFilter.setLeafSize(GLOBAL_MAP_VOXEL_SIZE,
                             GLOBAL_MAP_VOXEL_SIZE,
                             GLOBAL_MAP_VOXEL_SIZE);
    voxelFilterInit = true;
  }

  pcl::PointCloud<pcl::PointXYZI> downsampledCloud;
  voxelFilter.setInputCloud(tempCloud.makeShared());
  voxelFilter.filter(downsampledCloud);

  // 定期打印点数量（每超过 500 万点时）
  static size_t last_reported_size = 0;
  *globalMapAccumulator += downsampledCloud;

  size_t current_size = globalMapAccumulator->size();
  if (current_size > 5'000'000 && current_size - last_reported_size > 500'000) {
    RCLCPP_INFO(rclcpp::get_logger("visualizationTools"),
                "[MapSaver] Point count: %zu (downsampled, voxel=%.2fm)",
                current_size, GLOBAL_MAP_VOXEL_SIZE);
    last_reported_size = current_size;
  }
}

// ============================================================
// Handler 2: 订阅 /exploration_finish
// 当收到 true 时自动触发保存
// ============================================================
void explorationFinishHandler(const std_msgs::msg::Bool::ConstSharedPtr msg) {
  if (msg->data && !map_saved_flag) {
    exploration_finished_flag = true;
    RCLCPP_INFO(rclcpp::get_logger("MapSaver"),
                "[MapSaver] /exploration_finish = true, "
                "triggering global map save...");

    bool ok = saveGlobalMap(kDefaultSaveDir, "exploration_finish",
                             rclcpp::Node::make_shared("MapSaver"));
    map_saved_flag = true;
    (void)ok; // 忽略未使用警告
  }
}

// ============================================================
// Handler 3: 手动强制保存 Service
// ============================================================
void triggerSaveService(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
  (void)request;
  RCLCPP_INFO(rclcpp::get_logger("MapSaver"),
              "[MapSaver] Manual save triggered via service call!");

  bool ok = saveGlobalMap(kDefaultSaveDir, "manual_trigger",
                           rclcpp::Node::make_shared("MapSaver"));

  if (ok) {
    response->success = true;
    response->message = "Map saved successfully.";
  } else {
    response->success = false;
    response->message = "Map save failed (see console for details).";
  }
}

// ============================================================
// Handler 4: 手动强制保存 Topic
// 收到任意消息都会触发保存（发布一个空消息即可）
// ============================================================
void triggerSaveTopicHandler(
    const std_msgs::msg::Empty::ConstSharedPtr /*msg*/) {
  if (map_saved_flag) {
    RCLCPP_WARN(rclcpp::get_logger("MapSaver"),
                "[MapSaver] Map already saved this session. "
                "Save again anyway.");
  }

  RCLCPP_INFO(rclcpp::get_logger("MapSaver"),
              "[MapSaver] Manual save triggered via Topic /trigger_map_save!");

  saveGlobalMap(kDefaultSaveDir, "manual_topic_trigger",
                 rclcpp::Node::make_shared("MapSaver"));
  map_saved_flag = true;
}

// ============================================================
// 辅助函数：定时检查 exploration_finished_flag
// 在主循环中被调用
// ============================================================
void checkExplorationFinished(rclcpp::Node::SharedPtr node) {
  if (exploration_finished_flag && !map_saved_flag) {
    saveGlobalMap(kDefaultSaveDir, "exploration_finish", node);
    map_saved_flag = true;
  }
}

void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  systemTime = rclcpp::Time(odom->header.stamp).seconds();
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odom->pose.pose.orientation;
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  float dYaw = fabs(yaw - vehicleYaw);
  if (dYaw > PI) dYaw = 2 * PI  - dYaw;

  float dx = odom->pose.pose.position.x - vehicleX;
  float dy = odom->pose.pose.position.y - vehicleY;
  float dz = odom->pose.pose.position.z - vehicleZ;
  float dis = sqrt(dx * dx + dy * dy + dz * dz);

  if (!systemDelayInited) {
    vehicleYaw = yaw;
    vehicleX = odom->pose.pose.position.x;
    vehicleY = odom->pose.pose.position.y;
    vehicleZ = odom->pose.pose.position.z;
    return;
  }

  if (systemInited) {
    timeDuration = systemTime - systemInitTime;
    
    std_msgs::msg::Float32 timeDurationMsg;
    timeDurationMsg.data = timeDuration;
    pubTimeDurationPtr->publish(timeDurationMsg);
  }

  if (dis < transInterval && dYaw < yawInterval) {
    return;
  }

  if (!systemInited) {
    dis = 0;
    systemInitTime = systemTime;
    systemInited = true;
  }

  travelingDis += dis;

  vehicleYaw = yaw;
  vehicleX = odom->pose.pose.position.x;
  vehicleY = odom->pose.pose.position.y;
  vehicleZ = odom->pose.pose.position.z;

  fprintf(trajFilePtr, "%f %f %f %f %f %f %f\n", vehicleX, vehicleY, vehicleZ, roll, pitch, yaw, timeDuration);

  pcl::PointXYZI point;
  point.x = vehicleX;
  point.y = vehicleY;
  point.z = vehicleZ;
  point.intensity = travelingDis;
  trajectory->push_back(point);

  sensor_msgs::msg::PointCloud2 trajectory2;
  pcl::toROSMsg(*trajectory, trajectory2);
  trajectory2.header.stamp = odom->header.stamp;
  trajectory2.header.frame_id = "map";
  pubTrajectoryPtr->publish(trajectory2);
}

void laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloudIn)
{
  if (!systemDelayInited) {
    systemDelayCount++;
    if (systemDelayCount > systemDelay) {
      systemDelayInited = true;
    }
  }

  if (!systemInited) {
    return;
  }

  laserCloud->clear();
  pcl::fromROSMsg(*laserCloudIn, *laserCloud);

  *exploredVolumeCloud += *laserCloud;

  exploredVolumeCloud2->clear();
  exploredVolumeDwzFilter.setInputCloud(exploredVolumeCloud);
  exploredVolumeDwzFilter.filter(*exploredVolumeCloud2);

  pcl::PointCloud<pcl::PointXYZI>::Ptr tempCloud = exploredVolumeCloud;
  exploredVolumeCloud = exploredVolumeCloud2;
  exploredVolumeCloud2 = tempCloud;

  exploredVolume = exploredVolumeVoxelSize * exploredVolumeVoxelSize * 
                   exploredVolumeVoxelSize * exploredVolumeCloud->points.size();

  *exploredAreaCloud += *laserCloud;

  exploredAreaDisplayCount++;
  if (exploredAreaDisplayCount >= 5 * exploredAreaDisplayInterval) {
    exploredAreaCloud2->clear();
    exploredAreaDwzFilter.setInputCloud(exploredAreaCloud);
    exploredAreaDwzFilter.filter(*exploredAreaCloud2);

    tempCloud = exploredAreaCloud;
    exploredAreaCloud = exploredAreaCloud2;
    exploredAreaCloud2 = tempCloud;

    sensor_msgs::msg::PointCloud2 exploredArea2;
    pcl::toROSMsg(*exploredAreaCloud, exploredArea2);
    exploredArea2.header.stamp = laserCloudIn->header.stamp;
    exploredArea2.header.frame_id = "map";
    pubExploredAreaPtr->publish(exploredArea2);

    exploredAreaDisplayCount = 0;
  }

  fprintf(metricFilePtr, "%f %f %f %f\n", exploredVolume, travelingDis, runtime, timeDuration);

  std_msgs::msg::Float32 exploredVolumeMsg;
  exploredVolumeMsg.data = exploredVolume;
  pubExploredVolumePtr->publish(exploredVolumeMsg);
  
  std_msgs::msg::Float32 travelingDisMsg;
  travelingDisMsg.data = travelingDis;
  pubTravelingDisPtr->publish(travelingDisMsg);
}

void runtimeHandler(const std_msgs::msg::Float32::ConstSharedPtr runtimeIn)
{
  runtime = runtimeIn->data;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto nh = rclcpp::Node::make_shared("visualizationTools");

  nh->declare_parameter<std::string>("metricFile", metricFile);
  nh->declare_parameter<std::string>("trajFile", trajFile);
  nh->declare_parameter<std::string>("mapFile", mapFile);
  nh->declare_parameter<double>("overallMapVoxelSize", overallMapVoxelSize);
  nh->declare_parameter<double>("exploredAreaVoxelSize", exploredAreaVoxelSize);
  nh->declare_parameter<double>("exploredVolumeVoxelSize", exploredVolumeVoxelSize);
  nh->declare_parameter<double>("transInterval", transInterval);
  nh->declare_parameter<double>("yawInterval", yawInterval);
  nh->declare_parameter<int>("overallMapDisplayInterval", overallMapDisplayInterval);
  nh->declare_parameter<int>("exploredAreaDisplayInterval", exploredAreaDisplayInterval);

  nh->get_parameter("metricFile", metricFile);
  nh->get_parameter("trajFile", trajFile);
  nh->get_parameter("mapFile", mapFile);
  nh->get_parameter("overallMapVoxelSize", overallMapVoxelSize);
  nh->get_parameter("exploredAreaVoxelSize", exploredAreaVoxelSize);
  nh->get_parameter("exploredVolumeVoxelSize", exploredVolumeVoxelSize);
  nh->get_parameter("transInterval", transInterval);
  nh->get_parameter("yawInterval", yawInterval);
  nh->get_parameter("overallMapDisplayInterval", overallMapDisplayInterval);
  nh->get_parameter("exploredAreaDisplayInterval", exploredAreaDisplayInterval);

  // No direct replacement present for $(find pkg) in ROS2. Edit file path.
  {
    size_t pos;
    pos = metricFile.find("/install/");
    if (pos != std::string::npos) metricFile.replace(pos, 8, "/src");
    pos = trajFile.find("/install/");
    if (pos != std::string::npos) trajFile.replace(pos, 8, "/src");
  }

  auto subOdometry = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odometryHandler);

  auto subLaserCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan", 5, laserCloudHandler);

  auto subRuntime = nh->create_subscription<std_msgs::msg::Float32>("/runtime", 5, runtimeHandler);

  auto pubOverallMap = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/overall_map", 5);

  pubExploredAreaPtr = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/explored_areas", 5);

  pubTrajectoryPtr = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/trajectory", 5);
  
  pubExploredVolumePtr = nh->create_publisher<std_msgs::msg::Float32>("/explored_volume", 5);

  pubTravelingDisPtr = nh->create_publisher<std_msgs::msg::Float32>("/traveling_distance", 5);

  pubTimeDurationPtr = nh->create_publisher<std_msgs::msg::Float32>("/time_duration", 5);

  // ============================================================
  // 新增：地图保存相关 subscriber 和 service
  // ============================================================

  // 订阅 /registered_scan，累计存入 globalMapAccumulator
  auto subLaserCloudForSave =
      nh->create_subscription<sensor_msgs::msg::PointCloud2>(
          "/registered_scan", 5, laserCloudForSaveHandler);

  // 订阅 /exploration_finish（消息类型为 std_msgs/Bool）
  auto subExplorationFinish =
      nh->create_subscription<std_msgs::msg::Bool>(
          "/exploration_finish", 5, explorationFinishHandler);

  // 手动强制保存 Topic（收到任意消息即保存）
  auto subTriggerSave =
      nh->create_subscription<std_msgs::msg::Empty>(
          "/trigger_map_save", 5, triggerSaveTopicHandler);

  // 手动强制保存 Service
  auto srvTriggerSave =
      nh->create_service<std_srvs::srv::Trigger>(
          "/save_map_trigger", &triggerSaveService);

  RCLCPP_INFO(nh->get_logger(),
              "[MapSaver] Map save service  : /save_map_trigger");
  RCLCPP_INFO(nh->get_logger(),
              "[MapSaver] Map save topic    : /trigger_map_save");
  RCLCPP_INFO(nh->get_logger(),
              "[MapSaver] Map save auto     : /exploration_finish");
  RCLCPP_INFO(nh->get_logger(),
              "[MapSaver] Output directory  : %s", kDefaultSaveDir.c_str());

  overallMapDwzFilter.setLeafSize(overallMapVoxelSize, overallMapVoxelSize, overallMapVoxelSize);
  exploredAreaDwzFilter.setLeafSize(exploredAreaVoxelSize, exploredAreaVoxelSize, exploredAreaVoxelSize);
  exploredVolumeDwzFilter.setLeafSize(exploredVolumeVoxelSize, exploredVolumeVoxelSize, exploredVolumeVoxelSize);

  pcl::PLYReader ply_reader;
  if (ply_reader.read(mapFile, *overallMapCloud) == -1) {
    RCLCPP_INFO(nh->get_logger(), "Couldn't read pointcloud.ply file.");
  }

  overallMapCloudDwz->clear();
  overallMapDwzFilter.setInputCloud(overallMapCloud);
  overallMapDwzFilter.filter(*overallMapCloudDwz);
  overallMapCloud->clear();

  pcl::toROSMsg(*overallMapCloudDwz, overallMap2);

  time_t logTime = time(0);
  tm *ltm = localtime(&logTime);
  string timeString = to_string(1900 + ltm->tm_year) + "-" + to_string(1 + ltm->tm_mon) + "-" + to_string(ltm->tm_mday) + "-" +
                      to_string(ltm->tm_hour) + "-" + to_string(ltm->tm_min) + "-" + to_string(ltm->tm_sec);

  metricFile += "_" + timeString + ".txt";
  trajFile += "_" + timeString + ".txt";
  metricFilePtr = fopen(metricFile.c_str(), "w");
  trajFilePtr = fopen(trajFile.c_str(), "w");

  rclcpp::Rate rate(100);
  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);
    overallMapDisplayCount++;

    // 每隔一定次数检查是否需要触发保存
    checkExplorationFinished(nh);

    if (overallMapDisplayCount >= 100 * overallMapDisplayInterval) {
      overallMap2.header.stamp = rclcpp::Time(static_cast<uint64_t>(systemTime * 1e9));
      overallMap2.header.frame_id = "map";
      pubOverallMap->publish(overallMap2);

      overallMapDisplayCount = 0;
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  fclose(metricFilePtr);
  fclose(trajFilePtr);

  RCLCPP_INFO(nh->get_logger(), "Exploration metrics and vehicle trajectory are saved in 'src/vehicle_simulator/log'.");

  return 0;
}
