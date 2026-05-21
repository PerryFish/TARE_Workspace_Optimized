/**
 * @file trajectory_smoother_node.cpp
 * @brief ROS2 3D trajectory smoother using uniform cubic B-Spline
 *
 * Intercepts TARE's raw waypoint output (/way_point) and produces a
 * smooth, flyable 3D trajectory suitable for multirotor flight controllers.
 *
 * Algorithm:
 *  - Accumulates waypoints into a sliding window (default 20 points)
 *  - Computes cumulative chord-length parameterization for uniform speed
 *  - Fits independent uniform cubic B-splines to X(t), Y(t), Z(t)
 *  - Resamples at configurable resolution (default 0.2 m)
 *  - Publishes smoothed path as nav_msgs::Path on /smoothed_trajectory
 *
 * Designed for low-latency onboard execution (Jetson / embedded ARM).
 * Computational cost: O(N) per update for N waypoints in window.
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/header.hpp>

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <cmath>
#include <algorithm>

class TrajectorySmoother : public rclcpp::Node
{
public:
  TrajectorySmoother()
  : Node("trajectory_smoother_node")
  {
    // --- Parameters ---
    this->declare_parameter<int>("smooth_window_size", 20);
    this->declare_parameter<double>("resample_resolution", 0.2);   // meters between output points
    this->declare_parameter<double>("max_segment_length", 5.0);    // reset window if gap > this
    this->declare_parameter<int>("min_points_for_smooth", 4);      // minimum points to perform B-spline
    this->declare_parameter<std::string>("input_waypoint_topic", "/way_point");
    this->declare_parameter<std::string>("output_path_topic", "/smoothed_trajectory");
    this->declare_parameter<std::string>("world_frame_id", "map");

    this->get_parameter("smooth_window_size", window_size_);
    this->get_parameter("resample_resolution", resample_resolution_);
    this->get_parameter("max_segment_length", max_segment_length_);
    this->get_parameter("min_points_for_smooth", min_points_);
    this->get_parameter("input_waypoint_topic", input_topic_);
    this->get_parameter("output_path_topic", output_topic_);
    this->get_parameter("world_frame_id", world_frame_);

    // --- Publishers & Subscribers ---
    waypoint_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        input_topic_, 10,
        std::bind(&TrajectorySmoother::WaypointCallback, this, std::placeholders::_1));

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(output_topic_, 10);

    RCLCPP_INFO(this->get_logger(),
                "TrajectorySmoother started. Window=%d, Resample=%.2f m, MinPoints=%d",
                window_size_, resample_resolution_, min_points_);
  }

private:
  // ================================================================
  // Callback: accumulate waypoints and trigger smoothing
  // ================================================================
  void WaypointCallback(const geometry_msgs::msg::PointStamped::ConstSharedPtr msg)
  {
    Eigen::Vector3d new_point(msg->point.x, msg->point.y, msg->point.z);

    // Detect large gaps → reset window to avoid interpolating across disconnected regions
    if (!window_.empty())
    {
      double gap = (new_point - window_.back()).norm();
      if (gap > max_segment_length_)
      {
        RCLCPP_WARN(this->get_logger(),
                    "Waypoint gap %.2f m > threshold %.2f m. Resetting window.",
                    gap, max_segment_length_);
        window_.clear();
      }
    }

    // Append to sliding window
    window_.push_back(new_point);
    if (static_cast<int>(window_.size()) > window_size_)
    {
      window_.pop_front();
    }

    // Publish smoothed trajectory
    PublishSmoothedPath(msg->header);
  }

  // ================================================================
  // Core: Uniform cubic B-Spline smoothing + resampling
  // ================================================================
  void PublishSmoothedPath(const std_msgs::msg::Header &header)
  {
    nav_msgs::msg::Path path;
    path.header = header;
    path.header.frame_id = world_frame_;

    const int n = static_cast<int>(window_.size());

    // Not enough points: pass through raw waypoints
    if (n < min_points_)
    {
      for (const auto &pt : window_)
      {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = header;
        pose.header.frame_id = world_frame_;
        pose.pose.position.x = pt.x();
        pose.pose.position.y = pt.y();
        pose.pose.position.z = pt.z();
        pose.pose.orientation.w = 1.0;
        path.poses.push_back(pose);
      }
      path_pub_->publish(path);
      return;
    }

    // Step 1: Compute cumulative chord-length parameterization
    std::vector<double> t(n, 0.0);
    for (int i = 1; i < n; ++i)
    {
      t[i] = t[i - 1] + (window_[i] - window_[i - 1]).norm();
    }
    double total_length = t.back();

    // Step 2: Build uniform cubic B-Spline basis
    // We use clamped uniform cubic B-spline with knot vector [0,0,0,0, t1, t2, ..., tm, T,T,T,T]
    // For simplicity and low latency, we use the Cox-deBoor recursion directly
    // on a uniform knot spacing for evaluation.

    // Build knot vector: open-uniform (clamped) with degree p=3
    const int p = 3;  // cubic
    const int m = n + p;  // number of knots - 1
    std::vector<double> knots(m + 1);

    // Clamped: first p+1 knots = 0, last p+1 knots = total_length
    for (int i = 0; i <= p; ++i)     knots[i] = 0.0;
    for (int i = n; i <= m; ++i)     knots[i] = total_length;

    // Interior knots: evenly spaced in parameter domain (chord-length normalized)
    // Map chord-length t_j to evenly spaced interior knot positions
    // Using averaging: interior knot k_{p+j} = (t_j + t_{j+1} + t_{j+2})/3 for j=1..n-p-1
    for (int j = 1; j <= n - p - 1; ++j)
    {
      double sum = 0.0;
      for (int k = j; k <= j + p - 1; ++k)
      {
        sum += t[std::min(k, n - 1)];
      }
      knots[p + j] = sum / p;
    }

    // Step 3: Sample control points = original waypoints (interpolation-like behavior)
    // For a true interpolation, we'd solve a linear system. For low-latency,
    // we use the waypoints as control points and evaluate the B-spline curve.
    // This produces a smoothing (approximating) spline rather than interpolating.
    const Eigen::MatrixXd &ctrl_x = ExtractCoordinate(0);  // X control points
    const Eigen::MatrixXd &ctrl_y = ExtractCoordinate(1);  // Y
    const Eigen::MatrixXd &ctrl_z = ExtractCoordinate(2);  // Z

    // Step 4: Resample at uniform arc-length spacing
    double step = resample_resolution_;
    int num_samples = std::max(2, static_cast<int>(total_length / step) + 1);

    for (int i = 0; i < num_samples; ++i)
    {
      double u = (num_samples > 1)
                   ? (static_cast<double>(i) / (num_samples - 1)) * total_length
                   : 0.0;

      // Clamp to valid range
      u = std::max(0.0, std::min(u, total_length));

      // Evaluate B-spline at parameter u
      Eigen::Vector3d pt = EvalBSpline(u, knots, ctrl_x, ctrl_y, ctrl_z, p);

      geometry_msgs::msg::PoseStamped pose;
      pose.header = header;
      pose.header.frame_id = world_frame_;
      pose.pose.position.x = pt.x();
      pose.pose.position.y = pt.y();
      pose.pose.position.z = pt.z();
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }

    path_pub_->publish(path);
  }

  // ================================================================
  // Extract one coordinate from the waypoint window into Eigen vector
  // ================================================================
  Eigen::VectorXd ExtractCoordinate(int dim) const
  {
    Eigen::VectorXd vec(window_.size());
    for (size_t i = 0; i < window_.size(); ++i)
    {
      vec(i) = (dim == 0) ? window_[i].x() :
               (dim == 1) ? window_[i].y() :
                            window_[i].z();
    }
    return vec;
  }

  // ================================================================
  // Cox-deBoor recursion: evaluate B-spline basis function N_{i,p}(u)
  // ================================================================
  double CoxDeBoor(int i, int p, double u, const std::vector<double> &knots) const
  {
    if (p == 0)
    {
      return (knots[i] <= u && u < knots[i + 1]) ? 1.0 : 0.0;
    }

    double left = 0.0, right = 0.0;

    double denom1 = knots[i + p] - knots[i];
    if (denom1 > 1e-9)
    {
      left = ((u - knots[i]) / denom1) * CoxDeBoor(i, p - 1, u, knots);
    }

    double denom2 = knots[i + p + 1] - knots[i + 1];
    if (denom2 > 1e-9)
    {
      right = ((knots[i + p + 1] - u) / denom2) * CoxDeBoor(i + 1, p - 1, u, knots);
    }

    return left + right;
  }

  // ================================================================
  // Evaluate B-spline curve at parameter u
  // C(u) = sum_{i=0}^{n-1} N_{i,p}(u) * P_i
  // ================================================================
  Eigen::Vector3d EvalBSpline(double u,
                              const std::vector<double> &knots,
                              const Eigen::VectorXd &cx,
                              const Eigen::VectorXd &cy,
                              const Eigen::VectorXd &cz,
                              int p) const
  {
    Eigen::Vector3d result(0.0, 0.0, 0.0);
    const int n = static_cast<int>(window_.size());

    for (int i = 0; i < n; ++i)
    {
      double N = CoxDeBoor(i, p, u, knots);
      if (N > 0.0)
      {
        result.x() += N * cx(i);
        result.y() += N * cy(i);
        result.z() += N * cz(i);
      }
    }

    return result;
  }

  // ================================================================
  // Members
  // ================================================================
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr waypoint_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  std::deque<Eigen::Vector3d> window_;

  int window_size_;
  double resample_resolution_;
  double max_segment_length_;
  int min_points_;
  std::string input_topic_;
  std::string output_topic_;
  std::string world_frame_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectorySmoother>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
