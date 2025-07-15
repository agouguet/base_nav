#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import heapq
import math
from base_nav.global_planner import GlobalPlanner
from functools import lru_cache

class LazyThetaStarPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.los_cache = {}

    def compute_path(self):
        if self.map_data is not None and self.robot_pose is not None and self.goal_pose is not None:
            self.lazy_theta_star()

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @lru_cache(maxsize=512)
    def neighbors(self, node):
        x, y = node
        # offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
        #            (-1, -1), (1, -1), (-1, 1), (1, 1)]
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # plus simple, pas de diagonale

        result = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_data.shape[1] and 0 <= ny < self.map_data.shape[0]:
                if self.map_data[ny, nx] < 50:
                    result.append((nx, ny))
        return result

    def lazy_theta_star(self):
        start = self.world_to_map(self.robot_pose.position.x, self.robot_pose.position.y)
        goal = self.world_to_map(self.goal_pose.position.x, self.goal_pose.position.y)

        open_heap = [(self.heuristic(start, goal), start)]
        open_set = {start}
        g_score = {start: 0}
        parent = {start: start}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            open_set.discard(current)

            # Ligne de visée vers le parent
            if parent[current] != current and not self.line_of_sight(parent[current], current):
                # Pas de ligne de visée : chercher nouveau parent
                best_parent = current
                best_g = g_score[current]

                for neighbor in self.neighbors(current):
                    if neighbor in g_score:
                        tentative_g = g_score[neighbor] + self.heuristic(neighbor, current)
                        if tentative_g < best_g and self.line_of_sight(neighbor, current):
                            best_g = tentative_g
                            best_parent = neighbor

                parent[current] = best_parent
                g_score[current] = best_g

            if current == goal:
                self.publish_path(parent, goal)
                return

            for neighbor in self.neighbors(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    f = tentative_g + self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        heapq.heappush(open_heap, (f, neighbor))
                        open_set.add(neighbor)

        self.get_logger().warn("Lazy Theta* n'a pas trouvé de chemin.")

    @lru_cache(maxsize=512)
    def line_of_sight(self, p1, p2):
        # key = tuple(sorted((p1, p2)))  # Symmetric caching
        # if key in self.los_cache:
        #     return self.los_cache[key]

        for x, y in self.bresenham(p1, p2):
            if not (0 <= x < self.map_data.shape[1] and 0 <= y < self.map_data.shape[0]) \
                    or self.map_data[y, x] >= 50:
                # self.los_cache[key] = False
                return False

        # self.los_cache[key] = True
        return True

    def bresenham(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                yield x, y
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                yield x, y
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        yield x, y

    def publish_path(self, parent, goal):
        path = []
        current = goal
        while parent[current] != current:
            path.append(current)
            current = parent[current]
        path.append(current)
        path.reverse()

        world_coords = np.array([self.map_to_world(x, y) for x, y in path])
        x, y = world_coords[:, 0], world_coords[:, 1]

        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for x_i, y_i in zip(x, y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x_i
            pose.pose.position.y = y_i
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LazyThetaStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
