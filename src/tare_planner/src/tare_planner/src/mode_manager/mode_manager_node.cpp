/**
 * @file mode_manager_node.cpp
 * @brief Hierarchical mode manager for dynamic obstacle avoidance
 *
 * Architecture:
 *   TARE (global exploration) → Mode Manager → Local Planner → Flight Controller
 *
 * State machine:
 *   FOLLOW_GLOBAL  – Normal operation, forwarding TARE waypoints as local goals
 *   LOCAL_AVOIDANCE – Dynamic obstacle on path; local planner takes over
 *   REJOIN_GLOBAL   – Obstacle cleared; guiding back to TARE global path
 *
 * Dependencies: rclcpp, geometry_msgs, nav_msgs, sensor_msgs, std_msgs
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <Eigen/Dense>
#include <deque>
#include <cmath>
#include <string>
#include <mutex>

class ModeManagerNode : public rclcpp::Node
{
public:
  // ================================================================
  // State enum
  // ================================================================
  enum class State : uint8_t
  {
    FOLLOW_GLOBAL = 0,
    LOCAL_AVOIDANCE = 1,
    REJOIN_GLOBAL = 2
  };

  ModeManagerNode()
  : Node("mode_manager_node"),
    current_state_(State::FOLLOW_GLOBAL),
    obstacle_detected_(false),
    obstacle_clear_timer_(0.0),
    rejoin_progress_(0.0)
  {
    // --- Parameters ---
    this->declare_parameter<std::string>("global_waypoint_topic", "/way_point");
    this->declare_parameter<std::string>("global_path_topic", "/smoothed_trajectory");
    this->declare_parameter<std::string>("obstacle_cloud_topic", "/dynamic_obstacle_cloud");
    this->declare_parameter<std::string>("odometry_topic", "/state_estimation_at_scan");
    this->declare_parameter<std::string>("local_goal_topic", "/local_goal");
    this->declare_parameter<std::string>("mode_status_topic", "/mode_status");
    this->declare_parameter<std::string>("world_frame", "map");
    this->declare_parameter<double>("safety_radius", 1.5);          // meters around path
    this->declare_parameter<double>("lookahead_distance", 5.0);     // meters ahead for local goal
    this->declare_parameter<double>("obstacle_clear_timeout", 3.0); // seconds before rejoin
    this->declare_parameter<double>("rejoin_speed", 0.3);           // fraction per cycle (0-1)
    this->declare_parameter<int>("obstacle_point_threshold", 5);    // min points to trigger avoidance
    this->declare_parameter<double>("rejoin_goal_tolerance", 0.5);  // meters to consider "rejoined"
    this->declare_parameter<bool>("publish_markers", true);

    this->get_parameter("global_waypoint_topic", wp_topic_);
    this->get_parameter("global_path_topic", path_topic_);
    this->get_parameter("obstacle_cloud_topic", cloud_topic_);
    this->get_parameter("odometry_topic", odom_topic_);
    this->get_parameter("local_goal_topic", goal_topic_);
    this->get_parameter("mode_status_topic", status_topic_);
    this->get_parameter("world_frame", world_frame_);
    this->get_parameter("safety_radius", safety_radius_);
    this->get_parameter("lookahead_distance", lookahead_dist_);
    this->get_parameter("obstacle_clear_timeout", clear_timeout_);
    this->get_parameter("rejoin_speed", rejoin_speed_);
    this->get_parameter("obstacle_point_threshold", obs_point_thr_);
    this->get_parameter("rejoin_goal_tolerance", rejoin_tol_);
    this->get_parameter("publish_markers", publish_markers_);

    // --- Subscribers ---
    waypoint_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        wp_topic_, rclcpp::QoS(1).reliable(),
        std::bind(&ModeManagerNode::WaypointCallback, this, std::placeholders::_1));

    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        path_topic_, rclcpp::QoS(1).reliable(),
        std::bind(&ModeManagerNode::GlobalPathCallback, this, std::placeholders::_1));

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ModeManagerNode::ObstacleCloudCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(1).reliable(),
        std::bind(&ModeManagerNode::OdometryCallback, this, std::placeholders::_1));

    // --- Publishers ---
    local_goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic_, 10);
    mode_status_pub_ = this->create_publisher<std_msgs::msg::String>(status_topic_, 10);

    if (publish_markers_)
    {
      obstacle_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
          "/mode_manager/obstacle_marker", 10);
      local_goal_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
          "/mode_manager/local_goal_marker", 10);
    }

    // --- Main control loop timer (20 Hz) ---
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&ModeManagerNode::ControlLoop, this));

    RCLCPP_INFO(this->get_logger(),
                "Mode Manager started. Safety radius=%.2f m, Lookahead=%.2f m",
                safety_radius_, lookahead_dist_);
  }

private:
  // ================================================================
  // Callbacks
  // ================================================================
  void WaypointCallback(const geometry_msgs::msg::PointStamped::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_waypoint_ = Eigen::Vector3d(msg->point.x, msg->point.y, msg->point.z);
  }

  void GlobalPathCallback(const nav_msgs::msg::Path::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    global_path_.clear();
    for (const auto &pose : msg->poses)
    {
      global_path_.emplace_back(
          pose.pose.position.x,
          pose.pose.position.y,
          pose.pose.position.z);
    }
  }

  void ObstacleCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    if (!msg) { return; }
    std::vector<Eigen::Vector3d> obstacle_points;
    obstacle_points.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height));

    try {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
      {
        if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z))
        {
          obstacle_points.emplace_back(*iter_x, *iter_y, *iter_z);
        }
      }
    } catch (const std::runtime_error &e) {
      RCLCPP_ERROR(this->get_logger(), "Invalid PointCloud2 fields: %s", e.what());
      return;
    }

    bool detected = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      detected = CheckObstaclesOnPath(obstacle_points);
      obstacle_detected_ = detected;
    }
  }

  void OdometryCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_position_ = Eigen::Vector3d(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z);
    has_odom_ = true;
  }

  // ================================================================
  // Core: Obstacle Detection — check point cloud against path corridor
  // ================================================================
  bool CheckObstaclesOnPath(const std::vector<Eigen::Vector3d> &obstacle_points)
  {
    if (!has_odom_ || global_path_.empty() || obstacle_points.empty())
    {
      return false;
    }

    // Build a collision corridor along the global path from robot to lookahead
    // For efficiency, only check the segment from robot to lookahead point
    Eigen::Vector3d lookahead_pt = ComputeLookaheadPoint();
    std::vector<Eigen::Vector3d> corridor_segment = ExtractPathSegment(
        current_position_, lookahead_pt);

    // Count obstacle points within safety radius of the corridor
    int collision_count = 0;
    for (const auto &obs_pt : obstacle_points)
    {
      // Quick AABB reject
      double min_x = std::min(current_position_.x(), lookahead_pt.x()) - safety_radius_;
      double max_x = std::max(current_position_.x(), lookahead_pt.x()) + safety_radius_;
      double min_y = std::min(current_position_.y(), lookahead_pt.y()) - safety_radius_;
      double max_y = std::max(current_position_.y(), lookahead_pt.y()) + safety_radius_;
      if (obs_pt.x() < min_x || obs_pt.x() > max_x ||
          obs_pt.y() < min_y || obs_pt.y() > max_y)
      {
        continue;
      }

      // Check distance to each segment of the corridor
      for (size_t i = 0; i + 1 < corridor_segment.size(); ++i)
      {
        double dist = PointToSegmentDistance(
            obs_pt, corridor_segment[i], corridor_segment[i + 1]);
        if (dist < safety_radius_)
        {
          collision_count++;
          break;
        }
      }

      if (collision_count >= obs_point_thr_)
      {
        break;  // Enough evidence — early exit
      }
    }

    return collision_count >= obs_point_thr_;
  }

  // ================================================================
  // Core: 20 Hz Control Loop — state machine execution
  // ================================================================
  void ControlLoop()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!has_odom_) return;

    double now = this->now().seconds();

    // --- State transitions ---
    switch (current_state_)
    {
      case State::FOLLOW_GLOBAL:
      {
        if (obstacle_detected_)
        {
          TransitionTo(State::LOCAL_AVOIDANCE, now);
        }
        else
        {
          // Normal operation: publish next waypoint as local goal
          PublishFollowGlobalGoal();
        }
        break;
      }

      case State::LOCAL_AVOIDANCE:
      {
        if (!obstacle_detected_)
        {
          obstacle_clear_timer_ += 0.05;  // 20 Hz → 50 ms per tick
          if (obstacle_clear_timer_ >= clear_timeout_)
          {
            TransitionTo(State::REJOIN_GLOBAL, now);
          }
        }
        else
        {
          obstacle_clear_timer_ = 0.0;  // Reset timer on re-detection
          // Local planner is active — publish an intermediate lookahead goal
          PublishAvoidanceGoal();
        }
        break;
      }

      case State::REJOIN_GLOBAL:
      {
        if (obstacle_detected_)
        {
          // Obstacle reappeared during rejoin — back to avoidance
          TransitionTo(State::LOCAL_AVOIDANCE, now);
        }
        else
        {
          // Smoothly guide back to global path
          PublishRejoinGoal();

          // Check if rejoined
          if (rejoin_progress_ >= 1.0)
          {
            RCLCPP_INFO(this->get_logger(), "Rejoin complete — resuming global following");
            TransitionTo(State::FOLLOW_GLOBAL, now);
          }
        }
        break;
      }
    }

    // Publish markers for visualization
    if (publish_markers_)
    {
      PublishMarkers();
    }
  }

  // ================================================================
  // State transition helper
  // ================================================================
  void TransitionTo(State new_state, double now)
  {
    std::string old_name = StateName(current_state_);
    std::string new_name = StateName(new_state);

    current_state_ = new_state;
    obstacle_clear_timer_ = 0.0;

    if (new_state == State::REJOIN_GLOBAL)
    {
      rejoin_progress_ = 0.0;
      rejoin_start_pos_ = current_position_;
      rejoin_target_pos_ = ComputeLookaheadPoint();
    }

    RCLCPP_WARN(this->get_logger(), "STATE TRANSITION: %s → %s", old_name.c_str(), new_name.c_str());
    PublishModeStatus(new_name);
  }

  // ================================================================
  // Goal publishers for each state
  // ================================================================
  void PublishFollowGlobalGoal()
  {
    geometry_msgs::msg::PoseStamped goal;
    goal.header.stamp = this->now();
    goal.header.frame_id = world_frame_;

    Eigen::Vector3d target = ComputeLookaheadPoint();
    goal.pose.position.x = target.x();
    goal.pose.position.y = target.y();
    goal.pose.position.z = target.z();
    goal.pose.orientation.w = 1.0;

    local_goal_pub_->publish(goal);
  }

  void PublishAvoidanceGoal()
  {
    // During avoidance, publish a short lookahead goal close to the robot
    // The local planner (EGO-Planner) will generate a safe trajectory to this
    geometry_msgs::msg::PoseStamped goal;
    goal.header.stamp = this->now();
    goal.header.frame_id = world_frame_;

    // Look ahead along global path but with reduced distance during avoidance
    Eigen::Vector3d target = ComputeLookaheadPoint(lookahead_dist_ * 0.5);
    goal.pose.position.x = target.x();
    goal.pose.position.y = target.y();
    goal.pose.position.z = target.z();
    goal.pose.orientation.w = 1.0;

    local_goal_pub_->publish(goal);
  }

  void PublishRejoinGoal()
  {
    // Gradually interpolate from current position back to global path
    rejoin_progress_ = std::min(1.0, rejoin_progress_ + rejoin_speed_);

    geometry_msgs::msg::PoseStamped goal;
    goal.header.stamp = this->now();
    goal.header.frame_id = world_frame_;

    // Linear interpolation between rejoin start point and global path target
    Eigen::Vector3d interpolated = rejoin_start_pos_ +
        rejoin_progress_ * (rejoin_target_pos_ - rejoin_start_pos_);

    goal.pose.position.x = interpolated.x();
    goal.pose.position.y = interpolated.y();
    goal.pose.position.z = interpolated.z();
    goal.pose.orientation.w = 1.0;

    local_goal_pub_->publish(goal);
  }

  void PublishModeStatus(const std::string &status)
  {
    std_msgs::msg::String msg;
    msg.data = status;
    mode_status_pub_->publish(msg);
  }

  // ================================================================
  // Interface Adapter: TARE waypoint → Local Goal
  // Extracts a lookahead point along the global path from robot position
  // ================================================================
  Eigen::Vector3d ComputeLookaheadPoint(double lookahead_dist_override = -1.0)
  {
    double lh_dist = (lookahead_dist_override > 0.0) ? lookahead_dist_override : lookahead_dist_;

    if (global_path_.empty())
    {
      // Fallback: use latest TARE waypoint
      return IsFinite(latest_waypoint_) ? latest_waypoint_ : current_position_;
    }

    // Walk along the global path accumulating distance until lookahead reached
    double accumulated = 0.0;
    Eigen::Vector3d prev = current_position_;

    for (size_t i = 0; i < global_path_.size(); ++i)
    {
      double seg_len = (global_path_[i] - prev).norm();
      if (!std::isfinite(seg_len)) { continue; }
      if (accumulated + seg_len >= lh_dist)
      {
        // Interpolate within this segment
        double remaining = lh_dist - accumulated;
        double t = (seg_len > 1e-9) ? (remaining / seg_len) : 0.0;
        t = std::max(0.0, std::min(1.0, t));
        Eigen::Vector3d target = prev + t * (global_path_[i] - prev);
        return IsFinite(target) ? target : current_position_;
      }
      accumulated += seg_len;
      prev = global_path_[i];
    }

    // Path exhausted — return last point
    return IsFinite(global_path_.back()) ? global_path_.back() : current_position_;
  }

  // ================================================================
  // Extract path segment from start to end position (subset of global path)
  // ================================================================
  std::vector<Eigen::Vector3d> ExtractPathSegment(
      const Eigen::Vector3d &start, const Eigen::Vector3d &end) const
  {
    std::vector<Eigen::Vector3d> segment;
    segment.push_back(start);

    bool past_start = false;
    for (const auto &pt : global_path_)
    {
      if (!past_start)
      {
        if ((pt - start).norm() < 0.1) past_start = true;
        continue;
      }
      segment.push_back(pt);
      if ((pt - end).norm() < 0.1) break;
    }

    segment.push_back(end);
    return segment;
  }

  // ================================================================
  // Geometry: Point-to-segment distance (3D)
  // ================================================================
  static double PointToSegmentDistance(
      const Eigen::Vector3d &p,
      const Eigen::Vector3d &a,
      const Eigen::Vector3d &b)
  {
    Eigen::Vector3d ab = b - a;
    Eigen::Vector3d ap = p - a;
    double ab_len2 = ab.squaredNorm();
    if (ab_len2 < 1e-12) { return (p - a).norm(); }
    double t = ab.dot(ap) / ab_len2;
    t = std::max(0.0, std::min(1.0, t));
    Eigen::Vector3d closest = a + t * ab;
    return (p - closest).norm();
  }

  static bool IsFinite(const Eigen::Vector3d &value)
  {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
  }

  // ================================================================
  // Visualization markers
  // ================================================================
  void PublishMarkers()
  {
    if (!obstacle_marker_pub_ || !local_goal_marker_pub_) return;

    // Obstacle collision corridor marker (red if obstacle, green if clear)
    visualization_msgs::msg::Marker corridor_marker;
    corridor_marker.header.stamp = this->now();
    corridor_marker.header.frame_id = world_frame_;
    corridor_marker.ns = "safety_corridor";
    corridor_marker.id = 0;
    corridor_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    corridor_marker.scale.x = 0.1;
    corridor_marker.color.r = obstacle_detected_ ? 1.0 : 0.0;
    corridor_marker.color.g = obstacle_detected_ ? 0.0 : 1.0;
    corridor_marker.color.b = 0.0;
    corridor_marker.color.a = 0.8;

    Eigen::Vector3d lookahead = ComputeLookaheadPoint();
    auto seg = ExtractPathSegment(current_position_, lookahead);
    for (const auto &pt : seg)
    {
      geometry_msgs::msg::Point p;
      p.x = pt.x(); p.y = pt.y(); p.z = pt.z();
      corridor_marker.points.push_back(p);
    }
    obstacle_marker_pub_->publish(corridor_marker);

    // Active local goal marker
    visualization_msgs::msg::Marker goal_marker;
    goal_marker.header.stamp = this->now();
    goal_marker.header.frame_id = world_frame_;
    goal_marker.ns = "local_goal";
    goal_marker.id = 0;
    goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
    goal_marker.scale.x = 0.5;
    goal_marker.scale.y = 0.5;
    goal_marker.scale.z = 0.5;
    goal_marker.color.r = 1.0;
    goal_marker.color.g = 0.5;
    goal_marker.color.b = 0.0;
    goal_marker.color.a = 1.0;

    switch (current_state_)
    {
      case State::FOLLOW_GLOBAL:
        goal_marker.color.g = 1.0; break;
      case State::LOCAL_AVOIDANCE:
        goal_marker.color.r = 1.0; goal_marker.color.g = 0.2; break;
      case State::REJOIN_GLOBAL:
        goal_marker.color.r = 0.0; goal_marker.color.g = 0.5;
        goal_marker.color.b = 1.0; break;
    }

    // Reuse cached lookahead from corridor marker computation above
    goal_marker.pose.position.x = lookahead.x();
    goal_marker.pose.position.y = lookahead.y();
    goal_marker.pose.position.z = lookahead.z();
    goal_marker.pose.orientation.w = 1.0;
    local_goal_marker_pub_->publish(goal_marker);
  }

  // ================================================================
  // Utility
  // ================================================================
  static std::string StateName(State s)
  {
    switch (s)
    {
      case State::FOLLOW_GLOBAL:  return "FOLLOW_GLOBAL";
      case State::LOCAL_AVOIDANCE: return "LOCAL_AVOIDANCE";
      case State::REJOIN_GLOBAL:  return "REJOIN_GLOBAL";
      default: return "UNKNOWN";
    }
  }

  // ================================================================
  // Members — Subscriptions
  // ================================================================
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr waypoint_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Members — Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr local_goal_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr obstacle_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr local_goal_marker_pub_;

  // Members — Timer
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Members — State
  State current_state_;
  bool obstacle_detected_;
  double obstacle_clear_timer_;
  double rejoin_progress_;
  Eigen::Vector3d rejoin_start_pos_;
  Eigen::Vector3d rejoin_target_pos_;

  // Members — Data buffers
  Eigen::Vector3d current_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d latest_waypoint_{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> global_path_;
  bool has_odom_{false};

  // Members — Parameters
  std::string wp_topic_, path_topic_, cloud_topic_, odom_topic_;
  std::string goal_topic_, status_topic_, world_frame_;
  double safety_radius_, lookahead_dist_, clear_timeout_;
  double rejoin_speed_, rejoin_tol_;
  int obs_point_thr_;
  bool publish_markers_;

  // Thread safety: serializes all callback/timer access to shared state
  mutable std::mutex data_mutex_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ModeManagerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
