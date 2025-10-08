#!/usr/bin/env python3
import time

from matplotlib import pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import heapq
import math
import cv2
from rdp import rdp


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
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/robot_odom')
        self.declare_parameter('goal_topic', '/global_goal')
        self.declare_parameter('global_path_topic', '/global_path')
        self.declare_parameter('robot_radius', 0.3) 

        self.map_subscriber = self.create_subscription(
            OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, 10)

        self.start_subscriber = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        self.goal_subscriber = self.create_subscription(
            PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)

        self.path_publisher = self.create_publisher(
            Path, self.get_parameter('global_path_topic').value, 10)

        self.map_data = None
        self.robot_pose = None
        self.goal_pose = None
        self.robot_radius = self.get_parameter('robot_radius').value

        # Timer for periodic planning
        self.create_timer(0.5, self.try_plan)

    def map_callback(self, msg):
        self.get_logger().info('Map received')
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        self.map_data = data
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.expand_obstacles()

    def expand_obstacles(self):
        """
        Élargit les obstacles dans la carte pour tenir compte du rayon du robot.
        """
        if self.map_data is None:
            return
        
        radius_in_cells = int(math.ceil(self.robot_radius / self.map_resolution))
        kernel_size = 2 * radius_in_cells + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        binary_map = (self.map_data > 0).astype(np.uint8)
        inflated_map = cv2.dilate(binary_map, kernel, iterations=1)
        self.map_data = (inflated_map * 100).astype(np.int8)
        reduce_factor = calculate_reduction_factor(self.map_data.shape[0], self.map_data.shape[1])
        self.get_logger().info("Size reduction factor: " + str(reduce_factor))
        self.downsample_map(reduce_factor)
        self.map_data[self.map_data < 50] = 0  # Considérer les valeurs basses comme libres
        self.map_data[self.map_data >= 50] = 100  # Marquer clairement les obstacles

    def downsample_map(self, factor):
        """
        Réduit la résolution de la carte en moyennant les cellules sur une grille.
        :param factor: Facteur de réduction de résolution.
        """
        if self.map_data is None:
            return

        new_height = self.map_data.shape[0] // factor
        new_width = self.map_data.shape[1] // factor
        self.map_data = self.map_data.astype(np.uint8)
        self.map_data = cv2.resize(
            self.map_data, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        self.map_resolution *= factor


    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose
        # self.try_plan()

    def goal_callback(self, msg):
        self.goal_pose = msg.pose
        # self.try_plan()

    def try_plan(self):
        if self.map_data is not None and self.robot_pose is not None and self.goal_pose is not None:
            self.bidirectional_a_star()

    def world_to_map(self, x, y):
        map_x = int((x - self.map_origin[0]) / self.map_resolution)
        map_y = int((y - self.map_origin[1]) / self.map_resolution)
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        x = map_x * self.map_resolution + self.map_origin[0]
        y = map_y * self.map_resolution + self.map_origin[1]
        return x, y

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def neighbors(self, node):
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1)]
        result = []
        for dx, dy in offsets:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.map_data.shape[1] and 0 <= ny < self.map_data.shape[0]:
                if self.map_data[ny, nx] < 10:  # Vérifie si la cellule est libre
                    result.append((nx, ny))
        return result

    def bidirectional_a_star(self):
        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        start_map = self.world_to_map(robot_x, robot_y)
        goal_map = self.world_to_map(self.goal_pose.position.x, self.goal_pose.position.y)

        if 0 <= robot_x < self.map_data.shape[1] and 0 <= robot_y < self.map_data.shape[0]:
            self.map_data[int(robot_y), int(robot_x)] = 0

        open_set_start = [(0, start_map)]
        open_set_goal = [(0, goal_map)]
        came_from_start = {}
        came_from_goal = {}
        g_score_start = {start_map: 0}
        g_score_goal = {goal_map: 0}

        while open_set_start and open_set_goal:
            _, current_start = heapq.heappop(open_set_start)
            _, current_goal = heapq.heappop(open_set_goal)

            if current_start in came_from_goal:
                self.publish_combined_path(came_from_start, came_from_goal, current_start)
                return
            if current_goal in came_from_start:
                self.publish_combined_path(came_from_start, came_from_goal, current_goal)
                return

            for neighbor in self.neighbors(current_start):
                tentative_g_score = g_score_start[current_start] + self.heuristic(current_start, neighbor)
                if neighbor not in g_score_start or tentative_g_score < g_score_start[neighbor]:
                    came_from_start[neighbor] = current_start
                    g_score_start[neighbor] = tentative_g_score
                    heapq.heappush(open_set_start, (tentative_g_score + self.heuristic(neighbor, goal_map), neighbor))

            for neighbor in self.neighbors(current_goal):
                tentative_g_score = g_score_goal[current_goal] + self.heuristic(current_goal, neighbor)
                if neighbor not in g_score_goal or tentative_g_score < g_score_goal[neighbor]:
                    came_from_goal[neighbor] = current_goal
                    g_score_goal[neighbor] = tentative_g_score
                    heapq.heappush(open_set_goal, (tentative_g_score + self.heuristic(neighbor, start_map), neighbor))

        self.get_logger().warn('No path found!')

        # plt.imshow(self.map_data)
        # plt.savefig("mygraph.png")

    def publish_combined_path(self, came_from_start, came_from_goal, meeting_point):
        path_start = []
        current = meeting_point
        while current in came_from_start:
            path_start.append(current)
            current = came_from_start[current]

        path_goal = []
        current = meeting_point
        while current in came_from_goal:
            path_goal.append(current)
            current = came_from_goal[current]

        path = path_start[::-1] + path_goal[1:]  # Combine les deux chemins
        # path = rdp(path) # -> TODO

        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for node in path:
            world_x, world_y = self.map_to_world(node[0], node[1])
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_publisher.publish(msg)
        # self.get_logger().info('Global Path Updated. N = ' + str(len(path)))
        # self.get_logger().info(str(path))


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
