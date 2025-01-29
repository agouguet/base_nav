#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/color_rgba.hpp"

using namespace std::chrono_literals;

class LocalPlannerDWA : public rclcpp::Node {
public:
    LocalPlannerDWA() : Node("local_planner_dwa"), min_score_(std::numeric_limits<double>::infinity()), max_score_(-std::numeric_limits<double>::infinity()) {
        // Declare and get parameters
        declare_parameter<std::string>("global_path_topic", "/global_path");
        declare_parameter<std::string>("odom_topic", "/robot_odom");
        declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
        declare_parameter<std::string>("laser_scan_topic", "/scan");
        declare_parameter<std::string>("robot_marker_topic", "/robot_marker");
        declare_parameter<std::string>("trajectories_topic", "/dwa_trajectories");
        declare_parameter<std::string>("local_goal_marker_topic", "/local_goal_marker");

        global_path_topic_ = get_parameter("global_path_topic").as_string();
        odom_topic_ = get_parameter("odom_topic").as_string();
        cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
        laser_scan_topic_ = get_parameter("laser_scan_topic").as_string();
        robot_marker_topic_ = get_parameter("robot_marker_topic").as_string();
        trajectories_topic_ = get_parameter("trajectories_topic").as_string();
        local_goal_marker_topic_ = get_parameter("local_goal_marker_topic").as_string();

        // Subscribers and publishers
        odom_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic_, 10, std::bind(&LocalPlannerDWA::odom_callback, this, std::placeholders::_1));
        global_path_subscriber_ = create_subscription<nav_msgs::msg::Path>(global_path_topic_, 10, std::bind(&LocalPlannerDWA::path_callback, this, std::placeholders::_1));
        laser_scan_subscriber_ = create_subscription<sensor_msgs::msg::LaserScan>(laser_scan_topic_, 10, std::bind(&LocalPlannerDWA::scan_callback, this, std::placeholders::_1));
        cmd_vel_publisher_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
        trajectory_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(trajectories_topic_, 10);
        robot_marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(robot_marker_topic_, 10);
        local_goal_marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(local_goal_marker_topic_, 10);

        // Initialize DWA parameters
        max_speed_ = 0.5;
        max_omega_ = 1.0;
        acc_lim_ = 0.2;
        omega_lim_ = 0.5;
        simulation_time_ = 3.0;
        dt_ = 1.0;
        robot_radius_ = 0.3;

        // Timer for periodic planning
        timer_ = create_wall_timer(100ms, std::bind(&LocalPlannerDWA::plan_and_execute, this));
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_pose_.x = msg->pose.pose.position.x;
        current_pose_.y = msg->pose.pose.position.y;
        current_pose_.theta = yaw_from_quaternion(msg->pose.pose.orientation);
    }

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        global_path_.clear();
        for (const auto &pose : msg->poses) {
            global_path_.emplace_back(pose.pose.position.x, pose.pose.position.y);
        }
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        laser_scan_ = msg;
    }

    void plan_and_execute() {
        if (global_path_.empty() || !laser_scan_ || !current_pose_.is_initialized()) {
            RCLCPP_INFO(get_logger(), "Waiting for global path, laser scan, and odom...");
            return;
        }

        auto local_goal = select_local_goal();
        if (!local_goal) {
            RCLCPP_INFO(get_logger(), "No valid local goal found.");
            return;
        }

        publish_local_goal_marker(*local_goal);
        auto cmd_vel = dwa_control(*local_goal);
        cmd_vel_publisher_->publish(cmd_vel);
    }

    std::optional<std::pair<double, double>> select_local_goal() {
        if (global_path_.empty()) return std::nullopt;

        double current_x = current_pose_.x;
        double current_y = current_pose_.y;
        double lookahead_distance = 1.0;

        for (const auto &point : global_path_) {
            double distance = std::hypot(point.first - current_x, point.second - current_y);
            if (distance >= lookahead_distance) {
                return point;
            }
        }
        return global_path_.front();
    }

    geometry_msgs::msg::Twist dwa_control(const std::pair<double, double> &local_goal) {
        auto dynamic_window = compute_dynamic_window();

        min_score_ = std::numeric_limits<double>::infinity();
        max_score_ = -std::numeric_limits<double>::infinity();
        auto trajectories = simulate_trajectories(dynamic_window, local_goal);
        publish_trajectories(trajectories);

        auto best_trajectory = evaluate_trajectories(trajectories);

        geometry_msgs::msg::Twist cmd_vel;
        if (best_trajectory) {
            cmd_vel.linear.x = best_trajectory->v;
            cmd_vel.angular.z = best_trajectory->omega;
        } else {
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
        }
        return cmd_vel;
    }

    std::vector<std::pair<double, double>> compute_dynamic_window() {
        std::vector<std::pair<double, double>> dynamic_window;
        double v_step = max_speed_ / 5;
        double omega_step = max_omega_ / 7;

        for (double v = -max_speed_; v <= max_speed_; v += v_step) {
            for (double omega = -max_omega_; omega <= max_omega_; omega += omega_step) {
                dynamic_window.emplace_back(v, omega);
            }
        }
        return dynamic_window;
    }

    struct Trajectory {
        double v;
        double omega;
        std::vector<std::pair<double, double>> path;
    };

    std::vector<Trajectory> simulate_trajectories(const std::vector<std::pair<double, double>> &dynamic_window, const std::pair<double, double> &local_goal) {
        std::vector<Trajectory> trajectories;

        for (const auto &[v, omega] : dynamic_window) {
            Trajectory trajectory{v, omega};
            double x = current_pose_.x;
            double y = current_pose_.y;
            double theta = current_pose_.theta;

            for (double t = 0; t < simulation_time_; t += dt_) {
                x += v * std::cos(theta) * dt_;
                y += v * std::sin(theta) * dt_;
                theta += omega * dt_;
                trajectory.path.emplace_back(x, y);
            }
            trajectories.push_back(trajectory);
        }
        return trajectories;
    }

    std::optional<Trajectory> evaluate_trajectories(const std::vector<Trajectory> &trajectories) {
        std::optional<Trajectory> best_trajectory;
        double best_score = -std::numeric_limits<double>::infinity();

        for (const auto &trajectory : trajectories) {
            double goal_cost = -std::hypot(
                trajectory.path.back().first - global_path_.front().first,
                trajectory.path.back().second - global_path_.front().second
            );

            double score = goal_cost;
            if (score > best_score) {
                best_score = score;
                best_trajectory = trajectory;
            }
        }
        return best_trajectory;
    }

    void publish_local_goal_marker(const std::pair<double, double> &local_goal) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = now();
        marker.ns = "local_goal";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = local_goal.first;
        marker.pose.position.y = local_goal.second;
        marker.pose.position.z = 0.0;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        local_goal_marker_publisher_->publish(marker);
    }

    void publish_trajectories(const std::vector<Trajectory> &trajectories) {
        visualization_msgs::msg::MarkerArray marker_array;
        int id = 0;

        for (const auto &trajectory : trajectories) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = now();
            marker.ns = "trajectories";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.02;
            marker.color.a = 1.0;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;

            for (const auto &point : trajectory.path) {
                geometry_msgs::msg::Point p;
                p.x = point.first;
                p.y = point.second;
                p.z = 0.0;
                marker.points.push_back(p);
            }
            marker_array.markers.push_back(marker);
        }

        trajectory_publisher_->publish(marker_array);
    }

    struct Pose {
        double x{0.0}, y{0.0}, theta{0.0};
        bool is_initialized() const { return !(x == 0.0 && y == 0.0 && theta == 0.0); }
    };

    Pose current_pose_;
    std::vector<std::pair<double, double>> global_path_;
    sensor_msgs::msg::LaserScan::SharedPtr laser_scan_;

    double max_speed_, max_omega_, acc_lim_, omega_lim_, simulation_time_, dt_, robot_radius_;
    double min_score_, max_score_;

    std::string global_path_topic_, odom_topic_, cmd_vel_topic_, laser_scan_topic_, robot_marker_topic_, trajectories_topic_, local_goal_marker_topic_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_subscriber_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trajectory_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robot_marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr local_goal_marker_publisher_;

    rclcpp::TimerBase::SharedPtr timer_;

    double yaw_from_quaternion(const geometry_msgs::msg::Quaternion &q) {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalPlannerDWA>());
    rclcpp::shutdown();
    return 0;
}
