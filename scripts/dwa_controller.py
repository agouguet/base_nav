#!/usr/bin/env python3
import time

from scipy.spatial import KDTree
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, PoseStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor


class LocalPlannerDWA(Node):
    def __init__(self):
        super().__init__('local_planner_dwa')

        # Declare and get parameters
        self.declare_parameter('path_topic', '/global_path')
        self.declare_parameter('odom_topic', '/robot_odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('laser_scan_topic', '/scan')

        self.declare_parameter('robot_marker_topic', '/robot_marker')
        self.declare_parameter('trajectories_topic', '/dwa_trajectories')
        self.declare_parameter('local_goal_marker_topic', '/local_goal_maker')

        self.path_topic = self.get_parameter('path_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.laser_scan_topic = self.get_parameter('laser_scan_topic').value
        self.robot_marker_topic = self.get_parameter('robot_marker_topic').value
        self.trajectories_topic = self.get_parameter('trajectories_topic').value
        self.local_goal_marker_topic = self.get_parameter('local_goal_marker_topic').value

        # Subscribers and publishers
        self.odom_subscriber = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.path_subscriber = self.create_subscription(Path, self.path_topic, self.path_callback, 1)
        self.laser_scan_subscriber = self.create_subscription(LaserScan, self.laser_scan_topic, self.scan_callback, 1)
        self.cmd_vel_publisher = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.trajectory_publisher = self.create_publisher(MarkerArray, self.trajectories_topic, 1)
        self.robot_marker_publisher = self.create_publisher(Marker, self.robot_marker_topic, 1)
        self.local_goal_marker_publisher = self.create_publisher(Marker, self.local_goal_marker_topic, 1)

        # Internal state
        self.path = []
        self.laser_scan = None
        self.current_pose = None

        # DWA parameters
        self.goal_tolerance = 0.25
        self.max_speed = 0.5
        self.max_omega = 1.0
        self.acc_lim = 0.2
        self.omega_lim = 0.5
        self.simulation_time = 3.0
        self.dt = 1.0
        self.robot_radius = 0.3

        self.min_score = math.inf
        self.max_score = -math.inf

        # Timer for periodic planning
        self.create_timer(0.01, self.plan_and_execute)

    def odom_callback(self, msg):
        # Extract robot position and orientation from Odometry
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self.yaw_from_quaternion(msg.pose.pose.orientation)
        )
        self.current_velocity = msg.twist.twist.linear.x
        self.current_omega = msg.twist.twist.angular.z

    def yaw_from_quaternion(self, quaternion):
        # Convert quaternion to yaw angle
        _, _, yaw = self.euler_from_quaternion(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        return yaw

    def get_current_velocity(self):
        if self.current_pose is None:
            return 0.0, 0.0
        return self.current_velocity, self.current_omega

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def path_callback(self, msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        # self.plan_and_execute()

    def scan_callback(self, msg):
        self.laser_scan = msg

    def plan_and_execute(self):
        if not self.path or self.laser_scan is None or self.current_pose is None:
            self.get_logger().info("Waiting for global path, laser scan and odom...")
            return
        
        # Vérifier si le robot est suffisamment proche du but
        if len(self.path) <= 2:
            current_x, current_y, _ = self.current_pose
            goal_x, goal_y = self.path[-1]
            distance_to_goal = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)

            if distance_to_goal < self.goal_tolerance:
                # Arrêter le robot
                self.get_logger().info("Goal reached, stopping robot.")
                stop_cmd = Twist()
                self.cmd_vel_publisher.publish(stop_cmd)
                return

        # Use DWA to compute velocities
        cmd_vel = self.dwa_control()
        self.get_logger().info("DWA Control...  " + str(cmd_vel))

        # Publish the computed velocities
        self.cmd_vel_publisher.publish(cmd_vel)

    def dwa_control(self):
        """Dynamic Window Approach for motion planning."""

        # Dynamic window computation
        dynamic_window = self.compute_dynamic_window()
        

        self.min_score = math.inf
        self.max_score = -math.inf
        # Simulate trajectories
        trajectories = self.simulate_trajectories(dynamic_window)
        
        # Evaluate and select the best trajectory
        best_trajectory = self.evaluate_trajectories(trajectories)
        # self.publish_trajectories(trajectories)
        self.publish_trajectories([best_trajectory])
        # self.publish_robot_marker()

        # Extract velocities from the best trajectory
        cmd_vel = Twist()
        if best_trajectory:
            cmd_vel.linear.x = best_trajectory['v']
            cmd_vel.angular.z = best_trajectory['omega']
        else:
            # Stop if no valid trajectory is found
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        return cmd_vel

    def compute_dynamic_window(self):
        """Compute the dynamic window based on robot's limits."""
        return {
            'v_min': 0.0,
            'v_max': self.max_speed,
            'omega_min': -self.max_omega,
            'omega_max': self.max_omega
        }

    def simulate_trajectories(self, dynamic_window):
        """Simulate trajectories within the dynamic window."""
        t = time.time()

        trajectories = []
        for v in np.linspace(dynamic_window['v_min'], dynamic_window['v_max'], 7):
            for omega in np.linspace(dynamic_window['omega_min'], dynamic_window['omega_max'], 31):
                trajectory = self.simulate_trajectory(v, omega)
                # self.get_logger().info(str(time.time() - t))
                trajectories.append(trajectory)
        # self.get_logger().info("\n\n")
        return trajectories

    def simulate_trajectory(self, v, omega):
        """Simulate a single trajectory."""
        x, y, theta = self.current_pose
        trajectory = {'v': v, 'omega': omega, 'trajectory': [], 'cost': 0.0}

        for _ in range(int(self.simulation_time / self.dt)):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += omega * self.dt
            trajectory['trajectory'].append((x, y, theta))

        trajectory['cost'] = self.compute_trajectory_cost(trajectory)
        # self.get_logger().info(str(v) + "  " + str(omega) + "  "  + str(trajectory['cost']) + "     " + str(trajectory['trajectory']))
        trajectory['score'] = -trajectory['cost']
        self.min_score = min(self.min_score, trajectory['score'])
        self.max_score = max(self.max_score, trajectory['score'])
        return trajectory

    def compute_trajectory_cost(self, trajectory):
        """Compute the cost of a trajectory."""
        cost = 0.0

        if not self.path:
            return float('inf')

        if len(self.path) == 1:
            # Goal proximity cost
            final_x, final_y, final_theta = trajectory['trajectory'][-1]
            goal_dist = math.sqrt((self.path[0][0] - final_x)**2 + (self.path[0][1] - final_y)**2)
            cost += goal_dist
        else:
            # Distance cumulée au chemin
            path_cost = 0.0
            for x, y, _ in trajectory['trajectory']:
                min_dist = min(math.sqrt((px - x)**2 + (py - y)**2) for px, py in self.path[1:])
                path_cost += min_dist

            cost += path_cost  # Ajout de la distance cumulée au chemin

        # final_v = trajectory['v']
        # velocity_cost = 1.0 / (final_v + 0.1)  # On évite la division par zéro
        # cost += velocity_cost


        # # 3. Coût d'orientation (évite les virages brusques)
        # current_x, current_y, current_theta = self.current_pose
        # final_x, final_y, final_theta = trajectory['trajectory'][-1]
        
        # orientation_cost = abs(final_theta - current_theta)
        # cost += orientation_cost  # On ajoute ce coût


        obstacles = self.get_obstacle_distance()

        # Obstacle proximity cost
        for x, y, _ in trajectory['trajectory']:
            obstacle_dist = self.minimum_distance_to_obstacles(obstacles, x, y)
            if obstacle_dist < self.robot_radius:
                cost += 10000.0  # High penalty for collisions

        return cost

    def get_obstacle_distance(self):
        """Compute the distance from a point to the nearest obstacle."""
        if self.laser_scan is None or len(self.laser_scan.ranges) == 0:
            return float('inf')
        
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment

        obstacles = []
        for i, distance in enumerate(self.laser_scan.ranges):
            if distance == float('inf') or distance <= 0.0:
                continue
            angle = angle_min + i * angle_increment
            obstacle_x = distance * math.cos(angle)
            obstacle_y = distance * math.sin(angle)
            obstacles.append((obstacle_x, obstacle_y))
        return obstacles

    # def minimum_distance_to_obstacles(self, obstacles, x, y):
    #     if not obstacles:
    #         return float('inf')

    #     tree = KDTree(obstacles)
    #     dist, _ = tree.query([x, y])
    #     return dist

    def minimum_distance_to_obstacles(self, obstacles, x, y):
        robot_x, robot_y, robot_theta = self.current_pose

        dx = x - robot_x
        dy = y - robot_y
        local_x = dx * math.cos(-robot_theta) - dy * math.sin(-robot_theta)
        local_y = dx * math.sin(-robot_theta) + dy * math.cos(-robot_theta)

        min_distance = float('inf')
        for point in obstacles:
            dist = math.sqrt((local_x - point[0])**2 + (local_y - point[1])**2)
            if dist < min_distance:
                min_distance = dist
        return min_distance

    def evaluate_trajectories(self, trajectories):
        """Evaluate trajectories and select the one with the lowest cost."""
        if not trajectories:
            return None
        return min(trajectories, key=lambda traj: traj['cost'])

    def publish_trajectories(self, trajectories):
        marker_array = MarkerArray()

        for i, trajectory in enumerate(trajectories):
            marker = Marker()
            marker.header.frame_id = "map"  # Assurez-vous d'utiliser le même cadre que votre robot
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "trajectories"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Épaisseur de la ligne

            # Couleur basée sur le score
            score = trajectory['score']
            marker.color = self.get_color_from_score(score)

            # Points de la trajectoire
            for x, y, _ in trajectory['trajectory']:
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0  # Optionnel, si les trajectoires sont au sol
                marker.points.append(point)

            marker_array.markers.append(marker)

        # Publier les marqueurs
        self.trajectory_publisher.publish(marker_array)

    def get_color_from_score(self, score):
        # Normalisez les scores pour qu'ils soient entre 0 et 1
        normalized_score = min(max((score - self.min_score) / (self.max_score - self.min_score), 0), 1)
        color = ColorRGBA()
        color.r = 1.0 - normalized_score
        color.g = normalized_score        
        color.b = 0.0
        color.a = normalized_score # Opacité totale
        return color
    
    def publish_robot_marker(self):
        """
        Publie un marqueur circulaire représentant le robot dans RViz.
        """
        marker = Marker()
        marker.header.frame_id = "base_link"  # Cadre local du robot
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_marker"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position du robot (au sol)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0  # Z = 0, car le robot est sur le sol
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Taille du robot
        marker.scale.x = self.robot_radius  # Diamètre en X
        marker.scale.y = self.robot_radius  # Diamètre en Y
        marker.scale.z = 0.01  # Épaisseur minimale pour représenter un disque

        # Couleur du robot
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Légèrement transparent

        # Publier le marqueur
        self.robot_marker_publisher.publish(marker)

    def publish_local_goal_marker(self):
        """
        Publie un marqueur circulaire représentant le robot dans RViz.
        """
        if self.local_goal is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"  # Cadre local du robot
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_goal_marker"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position du robot (au sol)
        marker.pose.position.x = self.local_goal[0]
        marker.pose.position.y = self.local_goal[1]
        marker.pose.position.z = 0.0  # Z = 0, car le robot est sur le sol
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Taille du robot
        marker.scale.x = 0.3  # Diamètre en X
        marker.scale.y = 0.3  # Diamètre en Y
        marker.scale.z = 0.01  # Épaisseur minimale pour représenter un disque

        # Couleur du robot
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Légèrement transparent

        # Publier le marqueur
        self.local_goal_marker_publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerDWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
