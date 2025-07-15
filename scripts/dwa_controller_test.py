#!/usr/bin/env python3
import time, sys
from typing import Tuple, List, Dict, Any

from scipy.spatial import KDTree
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, PoseStamped, Pose
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from agents_msgs.msg import AgentArray # type: ignore
from std_msgs.msg import ColorRGBA
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

from rclpy.duration import Duration
import tf2_geometry_msgs.tf2_geometry_msgs
from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

THRESHOLD_DIST_STOP_ROBOT = 0.6

class LocalPlannerDWA(Node):
    def __init__(self):
        super().__init__('controller')

        # Declare and get parameters
        self.declare_parameter('path_topic', '/local_path')
        self.declare_parameter('goal_topic', '/global_goal')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('robot_pose', '/robot_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('agents_scan_topic', '/agents')

        self.declare_parameter('robot_marker_topic', '/robot_marker')
        self.declare_parameter('trajectories_topic', '/dwa_trajectories')
        self.declare_parameter('best_trajectory_topic', '/dwa_best_trajectory')
        self.declare_parameter('local_goal_marker_topic', '/local_goal_maker')

        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('hz', 10) 

        self.path_topic = self.get_parameter('path_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.agents_scan_topic = self.get_parameter('agents_scan_topic').value
        self.robot_marker_topic = self.get_parameter('robot_marker_topic').value
        self.trajectories_topic = self.get_parameter('trajectories_topic').value
        self.best_trajectory_topic = self.get_parameter('best_trajectory_topic').value
        self.local_goal_marker_topic = self.get_parameter('local_goal_marker_topic').value

        # Subscribers and publishers
        self.odom_subscriber = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.robot_pose_subscription_ = self.create_subscription(Pose, self.get_parameter('robot_pose').value, self.robot_pose_callback, 1)
        self.path_subscriber = self.create_subscription(Path, self.path_topic, self.path_callback, 1)
        self.laser_scan_subscriber = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
        self.agents_subscriber = self.create_subscription(AgentArray, self.agents_scan_topic, self.agentscallback, 1)

        self.cmd_vel_publisher = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.trajectory_publisher = self.create_publisher(MarkerArray, self.trajectories_topic, 1)
        self.best_trajectory_publisher = self.create_publisher(Marker, self.best_trajectory_topic, 1)
        self.robot_marker_publisher = self.create_publisher(Marker, self.robot_marker_topic, 1)
        self.local_goal_marker_publisher = self.create_publisher(Marker, self.local_goal_marker_topic, 1)

        # self.path_publisher = self.create_publisher(Path, "Test", 10)
        self.timer = self.create_timer(1.0/self.get_parameter('hz').value, self.plan_and_execute)

        # Internal state
        self.current_velocity = 0.0
        self.current_omega = 0.0
        self.reach_goal = False
        self.path = []
        self.laser_scan = None
        self.agents = None
        self.current_pose = None
        self.odom_time = None
        self.local_goal = None
        self.last_cmd_vel = Twist()
        self.cmd_alpha = 0.3 

        # DWA parameters
        self.goal_tolerance = 0.4
        self.min_speed = 0.0
        self.max_speed = 0.5
        self.max_omega = 0.5
        self.acc_lim = 0.4
        self.omega_lim = 2.5
        self.simulation_time = 2.0
        self.dt = 0.25
        self.robot_radius = self.get_parameter('robot_radius').value

        self.last_movement_time = time.perf_counter()
        self.block_duration_threshold = 0.75  # secondes sans bouger avant d’agir
        self.blocked_mode = False
        self.recovery_start_time = None
        self.recovery_duration = 1.0  # secondes minimum de récupération
        self.last_recovery_time = 0.0
        self.recovery_cooldown = 3.0  # secondes entre deux activations recovery

        self.min_score = math.inf
        self.max_score = -math.inf

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def odom_callback(self, msg):
        self.current_velocity = msg.twist.twist.linear.x
        self.current_omega = msg.twist.twist.angular.z

        if abs(self.current_velocity) > 0.01 or abs(self.current_omega) > 0.01:
            if self.blocked_mode:
                if time.perf_counter() - self.recovery_start_time < self.recovery_duration:
                    return
            self.last_movement_time = time.perf_counter()
            self.blocked_mode = False

    def robot_pose_callback(self, msg):
        self.current_pose = (
            msg.position.x,
            msg.position.y,
            self.yaw_from_quaternion(msg.orientation)
        )
        # self.plan_and_execute()
        

    def yaw_from_quaternion(self, quaternion):
        _, _, yaw = self.euler_from_quaternion(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        return yaw

    def get_current_velocity(self):
        if self.current_pose is None:
            return 0.0, 0.0
        return self.current_velocity, self.current_omega

    def path_callback(self, msg):
        new_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        if len(self.path) == 0 or len(new_path) == 0:
            self.reach_goal = False
        else:
            if self.path[-1] != new_path[-1]:
                self.reach_goal = False  
        self.path = new_path

    def scan_callback(self, msg):
        self.laser_scan = msg

    def agentscallback(self, msg):
        self.agents = msg

    def plan_and_execute(self):
        start_time = time.perf_counter()
        self.publish_robot_marker()

        if self.reach_goal:
            return

        if not self.path:
            return

        if self.laser_scan is None:
            return

        if self.current_pose is None:
            return
        
        if self.agents is not None:
            for agent in self.agents.agents:
                dist_robot_agent = math.sqrt((self.current_pose[0] - agent.pose.position.x)**2 + (self.current_pose[1] - agent.pose.position.y)**2)
                if dist_robot_agent <= THRESHOLD_DIST_STOP_ROBOT :
                    stop_cmd = Twist()
                    self.cmd_vel_publisher.publish(stop_cmd)


        current_time = time.perf_counter()
        # Si le robot est bloqué trop longtemps
        if ((current_time - self.last_movement_time) > self.block_duration_threshold and
            (current_time - self.last_recovery_time) > self.recovery_cooldown):
            if not self.blocked_mode:
                self.get_logger().warn("Robot bloqué : comportement de récupération activé")
                self.recovery_start_time = time.perf_counter()
                self.last_recovery_time = time.perf_counter()
                self.blocked_mode = True
                
        # Si on est en mode recovery
        if self.blocked_mode:
            recovery_cmd = self.recovery_rotation()
            self.publish_smoothed_cmd_vel(recovery_cmd)
            return

        if len(self.path) <= 2 and not self.reach_goal:
            current_x, current_y, _ = self.current_pose
            goal_x, goal_y = self.path[-1]
            distance_to_goal = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)

            if distance_to_goal < self.goal_tolerance:
                self.reach_goal = True
                stop_cmd = Twist()
                self.cmd_vel_publisher.publish(stop_cmd)
                return

        target_x, target_y = self.find_target_point_at_distance(0.8)  # distance en mètres
        self.local_goal = (target_x, target_y)
        self.publish_local_goal_marker()

        if self.local_goal is not None:
            cmd_vel = self.dwa_control()
            # self.cmd_vel_publisher.publish(cmd_vel)
            self.publish_smoothed_cmd_vel(cmd_vel)
            self.reach_goal = False

    def dwa_control(self):
        """Dynamic Window Approach for motion planning."""

        start_time = time.perf_counter()
        # Dynamic window computation
        dynamic_window = self.compute_dynamic_window()
        # self.get_logger().warn(str(dynamic_window))
        
        self.min_score = math.inf
        self.max_score = -math.inf

        # Simulate trajectories
        sim_start = time.perf_counter()
        trajectories = self.simulate_trajectories(dynamic_window)
        trajectories = self.normalize_costs(trajectories)
        if len(trajectories) == 0:
            cmd_vel = Twist()
            return cmd_vel
        
        # Evaluate and select the best trajectory
        best_trajectory = self.evaluate_trajectories(trajectories)

        self.publish_trajectories(trajectories, self.trajectory_publisher)
        self.publish_best_trajectory(best_trajectory, self.best_trajectory_publisher)

        # Extract velocities from the best trajectory
        cmd_vel = Twist()
        if best_trajectory:
            cmd_vel.linear.x = best_trajectory['v']
            cmd_vel.angular.z = best_trajectory['omega']
            print(f"\rBest: {best_trajectory['v']:.2f} {best_trajectory['omega']:.2f}", end="", flush=True)
            # self.get_logger().info(f"Best: {best_trajectory['v']:.2f} {best_trajectory['omega']:.2f}")
        return cmd_vel

    def publish_smoothed_cmd_vel(self, new_cmd):
        """Applique un lissage exponentiel aux commandes"""
        smoothed_cmd = Twist()
        smoothed_cmd.linear.x = (self.cmd_alpha * new_cmd.linear.x + 
                                (1 - self.cmd_alpha) * self.last_cmd_vel.linear.x)
        smoothed_cmd.angular.z = (self.cmd_alpha * new_cmd.angular.z + 
                                (1 - self.cmd_alpha) * self.last_cmd_vel.angular.z)
        
        current_time = time.perf_counter()
        if self.odom_time is not None:
            delta_time = current_time - self.odom_time
            self.get_logger().info(f"Temps entre récupération position et publication cmd_vel: {delta_time*1000:.2f} ms")


        self.cmd_vel_publisher.publish(smoothed_cmd)
        self.last_cmd_vel = smoothed_cmd

    def compute_dynamic_window(self):
        """Compute the dynamic window based on robot's limits."""
        cur_v, cur_omega = self.get_current_velocity()
        
        if abs(cur_omega) > self.max_omega * 2:
            cur_omega = max(-self.max_omega, min(self.max_omega, cur_omega))
        if abs(cur_v) > self.max_speed * 2:
            cur_v = max(-self.max_speed, min(self.max_speed, cur_v))

        return {
            'v_min': max(self.min_speed, cur_v - self.acc_lim * self.dt),
            'v_max': min(self.max_speed, cur_v + self.acc_lim * self.dt),
            'omega_min': max(-self.max_omega, cur_omega - self.omega_lim * self.dt),
            'omega_max': min(self.max_omega, cur_omega + self.omega_lim * self.dt)
        }


    def simulate_trajectories(self, dynamic_window):
        """Simulate trajectories within the dynamic window."""
        trajectories = []
        for v in np.linspace(dynamic_window['v_min'], dynamic_window['v_max'], 3):
            for omega in np.linspace(dynamic_window['omega_min'], dynamic_window['omega_max'], 11):
                trajectory = self.simulate_trajectory(v, omega)
                trajectories.append(trajectory)
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

        self.compute_trajectory_cost(trajectory)
        return trajectory

    def compute_trajectory_cost(self, trajectory):
        if not self.path:
            return float('inf')

        final_x, final_y, final_theta = trajectory['trajectory'][-1]
        robot_x, robot_y, robot_theta = self.current_pose

        # 1. Distance au chemin
        target_x, target_y = self.local_goal
        target_angle = math.atan2(target_y - final_y, target_x - final_x)

        current_path_cost = math.hypot(robot_x - target_x, robot_y - target_y) + abs(self.normalize_angle(robot_theta - target_angle)) # dist + angle
        final_path_cost = math.hypot(final_x - target_x, final_y - target_y) + abs(self.normalize_angle(final_theta - target_angle)) # dist + angle
        path_cost = (final_path_cost - current_path_cost) / final_path_cost
        # self.get_logger().info(f"{trajectory['v']:.2f} {trajectory['omega']:.2f} : {current_path_cost:.2f} {final_path_cost:.2f} {path_cost:.2f}" )

        # 2. Distance au but
        goal_x, goal_y = self.path[-1]
        goal_cost = math.sqrt((goal_x - final_x)**2 + (goal_y - final_y)**2)

        # 3. Orientation finale vers le but
        goal_angle = math.atan2(goal_y - final_y, goal_x - final_x)
        angle_diff = abs(self.normalize_angle(final_theta - goal_angle))

        # 4. Proximité obstacle
        obstacle_cost = 0.0
        obstacles = self.get_obstacle_distance()
        safe_distance = 0.3  # zone tampon autour des obstacles
        current_d = self.minimum_distance_to_obstacles(obstacles, robot_x, robot_y)
        # if trajectory['v'] > 1e-3:
        for i, (x, y, _) in enumerate(trajectory['trajectory']):
            d = self.minimum_distance_to_obstacles(obstacles, x, y)
            delta_d = current_d - d  # positif si on s'approche d'un obstacle
            time_factor = 1.0 + i  # pour atténuer les risques plus éloignés dans le temps

            if d < safe_distance:
                obstacle_cost += (delta_d / d) / time_factor

            if d < self.robot_radius:
                obstacle_cost += 1e6  # collision
                break
            # else:
            #     if d < safe_distance:
            #         obstacle_cost += math.exp(-d / (1.0 * self.robot_radius)) / time_factor

        # 5. Fluidité (variation de vitesse angulaire)
        smoothness_cost = abs(trajectory['omega'])

        trajectory["path_cost"] = path_cost
        trajectory["goal_cost"] = goal_cost
        trajectory["angle_cost"] = angle_diff
        trajectory["obstacle_cost"] = obstacle_cost
        trajectory["smoothness_cost"] = smoothness_cost
        t = trajectory
        # self.get_logger().info(f"     {t['v']:.2f} {t['omega']:.2f} : {t['cost']:.2f} | {t['path_cost']:.2f} {t['goal_cost']:.2f} {t['angle_cost']:.2f} {t['obstacle_cost']:.2f} {t['smoothness_cost']:.2f}  {current_d:.2f}" )

    def normalize_costs(self, trajectories):
        all_costs = []

        keys = ["path_cost", "goal_cost", "angle_cost", "obstacle_cost", "smoothness_cost"]

        min_costs = {k: min(t[k] for t in trajectories) for k in keys}
        max_costs = {k: max(t[k] for t in trajectories) for k in keys}

        for t in trajectories:
            for k in keys:
                denom = max_costs[k] - min_costs[k] + 1e-6
                t[k] = (t[k] - min_costs[k]) / denom

        weights = {
            'path_cost': 4.0,
            'goal_cost': 1.0,
            'angle_cost': 0.5,
            'obstacle_cost': 5.0,
            'smoothness_cost': 0.2
        }

        for t in trajectories:
            for k in keys:
                t["weight_"+k] = weights[k] * t[k]
            t["cost"] = sum(weights[k] * t[k] for k in keys)
            # self.get_logger().info(f"{t['v']:.2f} {t['omega']:.2f} : {t['cost']:.2f} | {t['weight_path_cost']:.2f} {t['weight_goal_cost']:.2f} {t['weight_angle_cost']:.2f} {t['weight_obstacle_cost']:.2f} {t['weight_smoothness_cost']:.2f}" )
            t['score'] = -t['cost']
            self.min_score = min(self.min_score, t['score'])
            self.max_score = max(self.max_score, t['score'])

        return trajectories

    def find_target_point_at_distance(self, distance_threshold: float) -> Tuple[float, float]:
        robot_x, robot_y, _ = self.current_pose
        obstacles = self.get_obstacle_distance()
        dense_path = self.densify_local_path(self.path, max_distance=distance_threshold+0.2, resolution=0.05)

        acc_distance = 0.0
        visible_goal = dense_path[0]
        

        for i in range(1, len(dense_path)):
            px, py = dense_path[i]
            
            step = math.hypot(px - dense_path[i-1][0], py - dense_path[i-1][1])
            acc_distance += step

            # self.get_logger().info(str(dense_path[i]) + "  " + str(self.is_line_of_sight_clear((robot_x, robot_y), (px, py), obstacles)))
            # if self.is_line_of_sight_clear((robot_x, robot_y), (px, py), obstacles):
            visible_goal = (px, py)

            if acc_distance >= distance_threshold:
                break

        return visible_goal

    def is_line_of_sight_clear(self, start, end, obstacles, resolution=0.05):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = int(distance / resolution)
        if steps == 0:
            return True

        for i in range(steps + 1):
            x = start[0] + dx * i / (steps)
            y = start[1] + dy * i / (steps)
            # if self.minimum_distance_to_obstacles(obstacles, x, y) < self.robot_radius + 0.01:
            if self.minimum_distance_to_obstacles(obstacles, x, y) < 0.1:
                return False
        return True

    def densify_local_path(self, path, max_distance=3.0, resolution=0.05):
        densified = []
        acc_distance = 0.0

        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i+1]
            dist = math.hypot(x1 - x0, y1 - y0)
            if acc_distance > max_distance:
                break
            steps = max(int(dist / resolution), 1)
            for j in range(steps):
                t = j / steps
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                densified.append((x, y))
            acc_distance += dist

        densified.append(path[min(len(path)-1, i+1)])
        return densified

    def get_obstacle_distance(self):
        """Compute the distance from a point to the nearest obstacle."""
        if self.laser_scan is None or len(self.laser_scan.ranges) == 0:
            return float('inf')
        
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment

        obstacles = []
        for i, distance in enumerate(self.laser_scan.ranges):
            if math.isnan(distance) or math.isinf(distance) or distance <= 0.1:
                continue
            angle = angle_min + i * angle_increment
            obstacle_x = distance * math.cos(angle)
            obstacle_y = distance * math.sin(angle)
            obstacles.append((obstacle_x, obstacle_y))
        return obstacles

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

    def recovery_rotation(self):
        """Retourne une commande de rotation sur place vers le chemin global"""
        cmd = Twist()
        cmd.linear.x = 0.0

        if not self.path or self.current_pose is None:
            cmd.angular.z = 0.3  # tourner par défaut
            return cmd

        robot_x, robot_y, robot_theta = self.current_pose
        goal_x, goal_y = self.path[-1]

        angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)
        angle_diff = self.normalize_angle(angle_to_goal - robot_theta)

        # Si le chemin est à gauche -> tourner à gauche, sinon à droite
        cmd.angular.z = -0.5 if angle_diff > 0 else 0.5

        return cmd

    def publish_best_trajectory(self, best_trajectory, publisher):
        now = self.get_clock().now().to_msg()        
        marker = Marker()
        marker.header.frame_id = "map" 
        marker.header.stamp = now
        marker.ns = "best_trajectory"
        marker.id = 1000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02 
        marker.color.r = 0.5
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0 
        for x, y, _ in best_trajectory['trajectory']:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0 
            marker.points.append(point)
        publisher.publish(marker)

    def publish_trajectories(self, trajectories, publisher):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, trajectory in enumerate(trajectories):
            marker = Marker()
            marker.header.frame_id = "map" 
            marker.header.stamp = now
            marker.ns = "trajectories"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Épaisseur de la ligne

            # Couleur basée sur le score
            score = trajectory['score']
            marker.color = self.get_color_from_score(score)
            marker.color.a - 0.3

            # Points de la trajectoire
            for x, y, _ in trajectory['trajectory']:
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0  # Optionnel, si les trajectoires sont au sol
                marker.points.append(point)

            marker_array.markers.append(marker)

        # Publier les marqueurs
        publisher.publish(marker_array)

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

    @staticmethod
    def normalize_angle(angle):
        """Normalise un angle entre -pi et pi."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def build_path_message(self, path):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            wx, wy = (x, y)
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerDWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
