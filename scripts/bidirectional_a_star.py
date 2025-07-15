#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
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

from base_nav.global_planner import GlobalPlanner

class BidirectionalAStarPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()

    def compute_path(self):
        if self.map_data is not None and self.robot_pose is not None and self.goal_pose is not None:
            self.bidirectional_a_star()

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def neighbors(self, node):
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1)]
        # offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dx, dy in offsets:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.map_data.shape[1] and 0 <= ny < self.map_data.shape[0]:
                if self.map_data[ny, nx] < 50:  # Vérifie si la cellule est libre
                    result.append((nx, ny))
        return result

    def bidirectional_a_star(self):
        start_map = self.world_to_map(self.robot_pose.position.x, self.robot_pose.position.y)
        goal_map = self.world_to_map(self.goal_pose.position.x, self.goal_pose.position.y)

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

        path = path_start[::-1] + path_goal[1:]

        # Conversion en coordonnées monde
        world_coords = np.array([self.map_to_world(x, y) for x, y in path])
        if len(world_coords) < 4:
            self.get_logger().warn("Chemin trop court pour interpolation spline.")
            return

        # Séparation des X et Y
        x = world_coords[:, 0]
        y = world_coords[:, 1]

        # Création de la spline (paramétrique)
        try:
            tck, u = splprep([x, y], s=0.0)  # s = facteur de lissage
            # unew = np.linspace(0, 1.0, num=5 * len(x))  # plus de points pour un chemin fluide
            unew = np.linspace(0, 1.0, num=len(x))
            x_smooth, y_smooth = splev(unew, tck)

            # Publication
            msg = Path()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()

            for x_i, y_i in zip(x_smooth, y_smooth):
            # for x_i, y_i in zip(x, y):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.pose.position.x = x_i
                pose.pose.position.y = y_i
                pose.pose.orientation.w = 1.0
                msg.poses.append(pose)

            self.path_publisher.publish(msg)
            self.get_logger().info(str(len(msg.poses)))
            # self.get_logger().info("Global path (smoothed) published.")

        except Exception as e:
            self.get_logger().error(f"Erreur lors du lissage du chemin : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BidirectionalAStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
