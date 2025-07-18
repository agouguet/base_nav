#!/usr/bin/env python3
import time
import rclpy
from abc import ABC, abstractmethod
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq
import math
import cv2
from rdp import rdp
from rclpy.duration import Duration
import tf2_geometry_msgs.tf2_geometry_msgs
from tf2_ros import TransformException, LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.interpolate import splprep, splev
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion


def quaternion_to_yaw(q):
    """Convertit un quaternion ROS en angle yaw (radians)"""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def calculate_reduction_factor(width, height, threshold=500, min_limit=1):
    """
    Calculate an integer reduction factor based on the image size.

    Args:
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        threshold (int): Size threshold (default is 1000x1000).
        min_limit (int): Minimum reduction factor (default is 1).

    Returns:
        int: Integer reduction factor.
    """
    # Determine the largest dimension of the image
    max_dimension = max(width, height)
    
    # If the image is below the threshold, the factor is 1
    if max_dimension <= threshold:
        return min_limit
    
    # Calculate a proportional reduction factor, then round up
    factor = math.ceil(max_dimension / threshold)
    
    # Ensure the factor is at least the minimum limit
    return max(min_limit, factor)

class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('robot_pose', 'robot_pose')
        self.declare_parameter('goal_topic', 'global_goal')
        self.declare_parameter('global_path_topic', 'global_path')
        self.declare_parameter('robot_radius', 0.3) 
        self.declare_parameter('hz', 10) 
        self.declare_parameter('prefix', "")

        self.prefix = self.get_parameter('prefix').value
        self.prefix = self.prefix+"/" if self.prefix != "" else self.prefix

        self.scan_subscriber = self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)

        self.map_subscriber = self.create_subscription(
            OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, 10)

        # self.start_subscriber = self.create_subscription(
        #     Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        self.robot_pose_subscription_ = self.create_subscription(PoseStamped, self.get_parameter('robot_pose').value, self.robot_pose_callback, 1)

        self.goal_subscriber = self.create_subscription(
            PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)

        self.path_publisher = self.create_publisher(
            Path, self.get_parameter('global_path_topic').value, 10)

        self.costmap_publisher = self.create_publisher(
            OccupancyGrid, 'costmap', 10)

        self.laser_obstacles = None
        self.obstacle_map = None
        self.map_data = None
        self.costmap = None
        self.robot_pose = None
        self.goal_pose = None
        self.robot_radius = self.get_parameter('robot_radius').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0/self.get_parameter('hz').value, self.compute_path)
        self.timer_costmap = self.create_timer(5, self.clear_costmap)

    def clear_costmap(self):
        self.get_logger().warn("Clear Costmap !")
        # self.costmap = None
        self.costmap = self.map_data

    def scan_callback(self, msg):
        self.get_logger().info(str(self.map_data) + "  \n" + str(self.robot_pose))
        if self.map_data is None or self.robot_pose is None:
            return
        # assert self.map_data.shape == free_space_mask.shape == laser_obstacles.shape, \
        #     f"Shape mismatch: map={self.map_data.shape}, free={free_space_mask.shape}, laser={laser_obstacles.shape}"

        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        q = self.robot_pose.orientation
        _, _, robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        # if self.obstacle_map is None:
        self.obstacle_map = np.zeros_like(self.map_data, dtype=np.uint8)
        # Créer une map vide pour obstacles détectés au scan actuel
        laser_obstacles = np.zeros_like(self.map_data, dtype=np.uint8)
        # Et une map pour la zone explicitement vue libre par le laser
        free_space_mask = np.zeros_like(self.map_data, dtype=np.uint8)

        angle = msg.angle_min

        # Coordonnées robot en map
        robot_mx, robot_my = self.world_to_map(robot_x, robot_y)

        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x_laser = r * math.cos(angle)
                y_laser = r * math.sin(angle)

                # x_global = robot_x + cos_yaw * x_laser - sin_yaw * y_laser
                # y_global = robot_y + sin_yaw * x_laser + cos_yaw * y_laser
                x_global = robot_x + cos_yaw * x_laser + sin_yaw * y_laser
                y_global = robot_y + sin_yaw * x_laser - cos_yaw * y_laser

                mx, my = self.world_to_map(x_global, y_global)

                if 0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]:
                    laser_obstacles[my, mx] = 255
                    self._mark_free_cells(free_space_mask, robot_mx, robot_my, mx, my)

            elif r >= msg.range_max:
                x_laser = msg.range_max * math.cos(angle)
                y_laser = msg.range_max * math.sin(angle)

                # x_global = robot_x + cos_yaw * x_laser - sin_yaw * y_laser
                # y_global = robot_y + sin_yaw * x_laser + cos_yaw * y_laser
                x_global = robot_x + cos_yaw * x_laser + sin_yaw * y_laser
                y_global = robot_y + sin_yaw * x_laser - cos_yaw * y_laser

                mx, my = self.world_to_map(x_global, y_global)

                if 0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]:
                    self._mark_free_cells(free_space_mask, robot_mx, robot_my, mx, my)

            angle += msg.angle_increment

        # Mettre à jour obstacle_map :
        # Supprimer obstacles uniquement là où free_space_mask = 1 (zones vues libres)
        self.obstacle_map[free_space_mask == 1] = 0

        # Ajouter obstacles détectés par laser
        self.obstacle_map[laser_obstacles == 255] = 255

        # Dilater obstacles pour sécurité (comme tu fais déjà)
        self.laser_obstacles = self.inflate_map(self.obstacle_map)

        # Mettre à jour costmap
        self.costmap = np.maximum(self.map_data, self.laser_obstacles)

    # def scan_callback(self, msg):
    #     self.get_logger().info(str(self.map_data) + "  \n" + str(self.robot_pose))
    #     if self.map_data is None or self.robot_pose is None:
    #         return
    #     # assert self.map_data.shape == free_space_mask.shape == laser_obstacles.shape, \
    #     #     f"Shape mismatch: map={self.map_data.shape}, free={free_space_mask.shape}, laser={laser_obstacles.shape}"

    #     try:
    #         transform = self.tf_buffer.lookup_transform(
    #             self.prefix+'map', msg.header.frame_id, msg.header.stamp)
    #     except (TransformException, LookupException, ConnectivityException, ExtrapolationException) as e:
    #         self.get_logger().warn(f"Transform not available yet: {e}")
    #         return

    #     # if self.obstacle_map is None:
    #     self.obstacle_map = np.zeros_like(self.map_data, dtype=np.uint8)
    #     # Créer une map vide pour obstacles détectés au scan actuel
    #     laser_obstacles = np.zeros_like(self.map_data, dtype=np.uint8)
    #     # Et une map pour la zone explicitement vue libre par le laser
    #     free_space_mask = np.zeros_like(self.map_data, dtype=np.uint8)

    #     angle = msg.angle_min

    #     # Coordonnées robot en map
    #     robot_mx, robot_my = self.world_to_map(self.robot_pose.position.x, self.robot_pose.position.y)

    #     for r in msg.ranges:
    #         if msg.range_min < r < msg.range_max:
    #             # Point obstacle en frame laser
    #             x_laser = r * math.cos(angle)
    #             y_laser = r * math.sin(angle)

    #             point = Pose()
    #             point.position.x = x_laser
    #             point.position.y = y_laser
    #             point.position.z = 0.0
    #             point.orientation.w = 1.0

    #             # Transformer vers map
    #             try:
    #                 point_map = tf2_geometry_msgs.do_transform_pose(point, transform)
    #             except Exception as e:
    #                 self.get_logger().warn(f"Transform point failed: {e}")
    #                 angle += msg.angle_increment
    #                 continue

    #             x_global = point_map.position.x
    #             y_global = point_map.position.y
    #             mx, my = self.world_to_map(x_global, y_global)

    #             if 0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]:
    #                 # Marquer obstacle
    #                 laser_obstacles[my, mx] = 255

    #                 # Tracer ligne robot->obstacle pour marquer zone libre
    #                 self._mark_free_cells(free_space_mask, robot_mx, robot_my, mx, my)

    #         elif r >= msg.range_max:
    #             # Pas d'obstacle jusqu'à la distance max
    #             # Marquer libre la ligne jusqu'à max range
    #             x_laser = msg.range_max * math.cos(angle)
    #             y_laser = msg.range_max * math.sin(angle)

    #             point = Pose()
    #             point.position.x = x_laser
    #             point.position.y = y_laser
    #             point.position.z = 0.0
    #             point.orientation.w = 1.0

    #             try:
    #                 point_map = tf2_geometry_msgs.do_transform_pose(point, transform)
    #             except Exception as e:
    #                 self.get_logger().warn(f"Transform point failed: {e}")
    #                 angle += msg.angle_increment
    #                 continue

    #             x_global = point_map.position.x
    #             y_global = point_map.position.y
    #             mx, my = self.world_to_map(x_global, y_global)

    #             if 0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]:
    #                 self._mark_free_cells(free_space_mask, robot_mx, robot_my, mx, my)

    #         angle += msg.angle_increment

    #     # Mettre à jour obstacle_map :
    #     # Supprimer obstacles uniquement là où free_space_mask = 1 (zones vues libres)
    #     self.obstacle_map[free_space_mask == 1] = 0

    #     # Ajouter obstacles détectés par laser
    #     self.obstacle_map[laser_obstacles == 255] = 255

    #     # Dilater obstacles pour sécurité (comme tu fais déjà)
    #     self.laser_obstacles = self.inflate_map(self.obstacle_map)

    #     # Mettre à jour costmap
    #     self.costmap = np.maximum(self.map_data, self.laser_obstacles)

    def _mark_free_cells(self, mask, x0, y0, x1, y1):
        """
        Trace une ligne sur la grille entre (x0, y0) et (x1, y1)
        et marque les cellules traversées dans mask à 1.
        Utilise l’algorithme Bresenham.

        mask: np.array 2D (uint8)
        x0, y0, x1, y1: entiers (indices dans la grille)
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    mask[y, x] = 1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    mask[y, x] = 1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        # Dernier point
        if 0 <= x1 < mask.shape[1] and 0 <= y1 < mask.shape[0]:
            mask[y1, x1] = 1

    def map_callback(self, msg):
        self.get_logger().info('Map received')
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        self.map_data = data
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_rotation = msg.info.origin.orientation

        self.expand_obstacles()

        reduce_factor = calculate_reduction_factor(self.map_data.shape[0], self.map_data.shape[1])
        reduce_factor = 1
        self.get_logger().info("Size reduction factor: " + str(reduce_factor))
        self.downsample_map(reduce_factor)


    def expand_obstacles(self):
        if self.map_data is None:
            return
        self.map_data = self.inflate_map(self.map_data)

    def inflate_map(self, map_to_inflate):
        binary_map = (map_to_inflate > 0).astype(np.uint8) * 255
        inverted_map = 255 - binary_map
        dist_map = cv2.distanceTransform(inverted_map, cv2.DIST_L2, 0)
        dist_map_m = dist_map * self.map_resolution

        r = self.robot_radius + 0.1  # rayon du robot en mètres
        k = 5.0  # facteur de décroissance exponentielle (ajuster)

        # Décalage par rapport au rayon : la zone sous r a coût max = 100
        dist_from_inflation = np.maximum(0, dist_map_m - r)

        cost_map = 100 * np.exp(-k * dist_from_inflation)
        # cost_map[dist_map_m <= r] = 100  # coût max dans la zone <= rayon robot
        cost_map[binary_map == 255] = 100  # obstacles

        cost_map = np.clip(cost_map, 0, 100).astype(np.int8)
        return cost_map

    def downsample_map(self, factor):
        """
        Réduction de la carte sans perte d'obstacles : une cellule est occupée
        si au moins une cellule d'origine est occupée.
        """
        h, w = self.map_data.shape
        h_new = h // factor
        w_new = w // factor

        # Tronquer pour que ce soit divisible
        self.map_data = self.map_data[:h_new * factor, :w_new * factor]
        
        # Reshape et prendre le maximum sur les blocs
        map_reshaped = self.map_data.reshape(h_new, factor, w_new, factor)
        downsampled = map_reshaped.max(axis=(1, 3))  # max: garde les obstacles

        self.map_data = downsampled
        self.map_resolution *= factor


    def odom_callback(self, msg):
        try:
            # On récupère la transform de 'map' à 'base_link'
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.prefix+'map',
                self.prefix+'base_link',
                rclpy.time.Time()
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # self.robot_pose = Pose()
            # self.robot_pose.position.x = translation.x
            # self.robot_pose.position.y = translation.y
            # self.robot_pose.position.z = translation.z
            # self.robot_pose.orientation = rotation

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Transform not available yet: {e}")

    def robot_pose_callback(self, msg):
        # self.get_logger().info('T: ' + str(self.goal) + "      " + str(self.global_path))
        self.robot_pose = msg.pose

    def goal_callback(self, msg):
        self.goal_pose = msg.pose

    @abstractmethod
    def compute_path(self):
        pass

    def world_to_map(self, x, y):
        # Translation par rapport à l'origine
        dx = x - self.map_origin[0]
        dy = y - self.map_origin[1]

        # Rotation inverse (monde → carte)
        yaw = quaternion_to_yaw(self.map_rotation)
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        rx = dx * cos_yaw - dy * sin_yaw
        ry = dx * sin_yaw + dy * cos_yaw

        map_x = int(rx / self.map_resolution)
        map_y = int(ry / self.map_resolution)
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        rx = map_x * self.map_resolution
        ry = map_y * self.map_resolution

        yaw = quaternion_to_yaw(self.map_rotation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        dx = rx * cos_yaw - ry * sin_yaw
        dy = rx * sin_yaw + ry * cos_yaw

        x = dx + self.map_origin[0]
        y = dy + self.map_origin[1]
        return x, y

    def publish_costmap(self):
        if self.costmap is None:
            return

        costmap_msg = OccupancyGrid()
        costmap_msg.header = Header()
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = self.prefix+'map'

        height, width = self.costmap.shape
        costmap_msg.info.resolution = self.map_resolution
        costmap_msg.info.width = width
        costmap_msg.info.height = height

        # Origine de la carte (pose)
        costmap_msg.info.origin.position.x = self.map_origin[0]
        costmap_msg.info.origin.position.y = self.map_origin[1]
        costmap_msg.info.origin.position.z = 0.0
        # costmap_msg.info.origin.orientation.w = 1.0  # Pas de rotation
        costmap_msg.info.origin.orientation = self.map_rotation

        # Aplatir la map et la convertir en int8 (ROS attend [-1,100])
        flat_data = self.costmap.flatten().astype(np.int8)
        costmap_msg.data = flat_data.tolist()

        self.costmap_publisher.publish(costmap_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
