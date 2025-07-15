#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import heapq
import math
from base_nav.global_planner import GlobalPlanner

BLOCKED_THRESHOLD = 90  # au lieu de 50

class JPSPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()

        self.keep_path = False
        self.cached_cost_path = math.inf
        self.cached_path_points = None  # Liste de points (map coords) du chemin calculé
        self.cached_path_msg = None     # Message ROS Path correspondant
        self.proximity_threshold = 0.8  # Distance max (en m) pour considérer que le robot est sur le chemin

    def compute_path(self):
        # self.get_logger().info(str(self.costmap) + "  \n" + str(self.robot_pose) + "  " + str(self.goal_pose) + "  ")
        if self.costmap is None or self.robot_pose is None or self.goal_pose is None:
            return
        
        self.publish_costmap()

        start = self.world_to_map(self.robot_pose.position.x, self.robot_pose.position.y)
        goal = self.world_to_map(self.goal_pose.position.x, self.goal_pose.position.y)

        # if self.cached_path_points is not None:
        #     robot_pos_world = (self.robot_pose.position.x, self.robot_pose.position.y)
        #     goal_pos_world = (self.goal_pose.position.x, self.goal_pose.position.y)
        #     if self.is_pos_near_path(robot_pos_world, self.cached_path_points) and self.is_pos_near_path(goal_pos_world, self.cached_path_points):
        #         truncated_path = self.truncate_path_from_robot(robot_pos_world, self.cached_path_points)
        #         msg = self.build_path_message(truncated_path)
        #         self.cached_path_points = truncated_path  # 🔁 Mise à jour ici
        #         # Robot proche du chemin actuel, republier sans recalcul
        #         self.path_publisher.publish(msg)
        #         return

        # Sinon recalculer le chemin avec JPS
        cost_path, path = self.jps(start, goal)
        if path:
            # if self.is_path_traversable(self.cached_path_points):
            #     if self.cached_cost_path > cost_path * (0.98):
            #         self.path_publisher.publish(self.cached_path_msg)
            #         return
            # similarity_with_previous_path = self.hausdorff_similarity(self.cached_path_points, path)
            # self.get_logger().info(str(similarity_with_previous_path))
            # self.get_logger().info(str(self.cached_path_points))
            # self.get_logger().info(str(path))
            # self.get_logger().info(" ----- \n")
            # if similarity_with_previous_path < 0.9:
            #     if self.is_path_traversable(self.cached_path_points):
            #         if self.cached_cost_path > cost_path * (0.90):
            #             self.path_publisher.publish(self.cached_path_msg)
            #             return
            self.cached_path_points = path
            self.cached_cost_path = cost_path
            self.cached_path_msg = self.build_path_message(path)
            self.path_publisher.publish(self.cached_path_msg)
        else:
            self.get_logger().warn("JPS n'a pas trouvé de chemin.")
            self.cached_path_points = None
            self.cached_path_msg = None

    def is_pos_near_path(self, pos_world, path_map_points):
        path_world_points = np.array([self.map_to_world(x, y) for x, y in path_map_points])
        dists = np.linalg.norm(path_world_points - np.array(pos_world), axis=1)
        min_dist = np.min(dists)
        return min_dist < self.proximity_threshold

    def truncate_path_from_robot(self, robot_world_pos, path_points):
        """
        Supprime les points du chemin déjà parcourus par le robot.
        """
        world_path = np.array([self.map_to_world(x, y) for x, y in path_points])
        robot_pos = np.array(robot_world_pos)

        dists = np.linalg.norm(world_path - robot_pos, axis=1)
        min_index = int(np.argmin(dists))

        # On retourne le chemin tronqué à partir du point le plus proche
        return path_points[min_index:]

    def jps(self, start, goal):
        # Implémentation classique JPS :
        # - open set avec heap
        # - fonction jump pour sauter vers le prochain noeud intéressant
        # - heuristic euclidienne
        # - prune neighbours selon JPS
        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(start, goal), start))
        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_heap:
            value, current = heapq.heappop(open_heap)
            if current == goal:
                return value, self.reconstruct_path(came_from, current)

            closed_set.add(current)
            for neighbor in self.pruned_neighbors(current, came_from.get(current, None)):
                jump_point = self.jump(current, neighbor, goal)
                if jump_point is not None and jump_point not in closed_set:
                    tentative_g = g_score[current] + self.distance(current, jump_point)
                    if tentative_g < g_score.get(jump_point, float('inf')):
                        came_from[jump_point] = current
                        g_score[jump_point] = tentative_g
                        f = tentative_g + self.heuristic(jump_point, goal)
                        heapq.heappush(open_heap, (f, jump_point))
        return None, None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def distance(self, a, b):
        x0, y0 = a
        x1, y1 = b
        num_steps = max(abs(x1 - x0), abs(y1 - y0))
        total_cost = 0.0

        for i in range(num_steps + 1):
            t = i / max(num_steps, 1)
            xi = int(round(x0 + t * (x1 - x0)))
            yi = int(round(y0 + t * (y1 - y0)))
            if 0 <= xi < self.costmap.shape[1] and 0 <= yi < self.costmap.shape[0]:
                cost = self.costmap[yi, xi] / 255.0
                total_cost += cost ** 2  # Pénalisation quadratique

        avg_cost = total_cost #/ (num_steps + 1)
        base_dist = math.hypot(x1 - x0, y1 - y0)
        return base_dist + avg_cost * 10

    def pruned_neighbors(self, node, parent):
        # JPS pruning rules
        if parent is None:
            return self.neighbors(node)

        x, y = node
        px, py = parent
        dx = (x - px) // max(abs(x - px), 1)
        dy = (y - py) // max(abs(y - py), 1)

        neighbors = []
        # Check natural neighbors depending on movement direction
        if dx != 0 and dy != 0:  # Diagonal move
            # natural neighbors for diagonal
            candidates = [(x + dx, y), (x, y + dy), (x + dx, y + dy)]
            # forced neighbors for diagonal
            if self.is_blocked(x - dx, y + dy) and not self.is_blocked(x - dx, y):
                candidates.append((x - dx, y + dy))
            if self.is_blocked(x + dx, y - dy) and not self.is_blocked(x, y - dy):
                candidates.append((x + dx, y - dy))
        else:  # Horizontal or vertical move
            candidates = []
            if dx == 0:
                candidates = [(x, y + dy)]
                if self.is_blocked(x + 1, y):
                    candidates.append((x + 1, y + dy))
                if self.is_blocked(x - 1, y):
                    candidates.append((x - 1, y + dy))
            else:
                candidates = [(x + dx, y)]
                if self.is_blocked(x, y + 1):
                    candidates.append((x + dx, y + 1))
                if self.is_blocked(x, y - 1):
                    candidates.append((x + dx, y - 1))

        return [n for n in candidates if self.is_valid(n)]

    def neighbors(self, node):
        x, y = node
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1)]
        result = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if self.is_valid((nx, ny)):
                result.append((nx, ny))
        return result

    def is_valid(self, node):
        x, y = node
        if 0 <= x < self.costmap.shape[1] and 0 <= y < self.costmap.shape[0]:
            return self.costmap[y, x] < BLOCKED_THRESHOLD
        return False

    def is_blocked(self, x, y):
        if 0 <= x < self.costmap.shape[1] and 0 <= y < self.costmap.shape[0]:
            return self.costmap[y, x] >= BLOCKED_THRESHOLD
        return True

    def is_path_traversable(self, path):
        """
        Vérifie si un chemin est empruntable en testant chaque point.
        `path` est une liste de tuples (x, y) en coordonnées de la carte (grille).
        Retourne True si tous les points sont dans des cellules libres.
        """
        if self.costmap is None or path is None:
            return False

        for x, y in path:
            mx, my = int(round(x)), int(round(y))
            if not (0 <= mx < self.costmap.shape[1] and 0 <= my < self.costmap.shape[0]):
                return False  # en dehors de la carte
            if self.costmap[my, mx] >= BLOCKED_THRESHOLD:
                return False  # cellule occupée ou risquée
        return True

    def hausdorff_similarity(self, path1, path2):
        """
        Calcule un taux de ressemblance robuste entre deux chemins (listes de (x, y)),
        basé sur une distance de Hausdorff symétrique et normalisée.

        Retourne un float ∈ [0.0, 1.0] : 1.0 = identique, 0.0 = très éloigné.
        """
        if not path1 or not path2:
            return 0.0

        def avg_min_distance(a, b):
            return np.mean([
                min(math.hypot(p1[0] - p2[0], p1[1] - p2[1]) for p2 in b)
                for p1 in a
            ])  

        d1 = avg_min_distance(path1, path2)
        d2 = avg_min_distance(path2, path1)
        d = max(d1, d2)  # Hausdorff symétrique

        # Convertit en similarité (0 = éloigné, 1 = identique)
        # On applique un facteur d'atténuation : au-delà de 2 mètres, on considère 0
        # similarity = max(0.0, 1.0 - hd / 2.0)
        similarity = math.exp(-d / 5.0)
        return similarity


    def jump(self, current, direction, goal):
        x, y = current
        dx = direction[0] - x
        dy = direction[1] - y

        nx, ny = x + dx, y + dy

        if not self.is_valid((nx, ny)):
            return None
        if (nx, ny) == goal:
            return (nx, ny)

        # Check forced neighbors (depends if moving diagonally or straight)
        if dx != 0 and dy != 0:  # diagonal
            if (self.is_valid((nx - dx, ny + dy)) and self.is_blocked(nx - dx, ny)) or \
               (self.is_valid((nx + dx, ny - dy)) and self.is_blocked(nx, ny - dy)):
                return (nx, ny)
        else:  # horizontal or vertical
            if dx != 0:
                if (self.is_valid((nx + dx, ny + 1)) and self.is_blocked(nx, ny + 1)) or \
                   (self.is_valid((nx + dx, ny - 1)) and self.is_blocked(nx, ny - 1)):
                    return (nx, ny)
            else:
                if (self.is_valid((nx + 1, ny + dy)) and self.is_blocked(nx + 1, ny)) or \
                   (self.is_valid((nx - 1, ny + dy)) and self.is_blocked(nx - 1, ny)):
                    return (nx, ny)

        # Recursive jump
        if dx != 0 and dy != 0:
            if self.jump((nx, ny), (nx + dx, ny), goal) is not None or \
               self.jump((nx, ny), (nx, ny + dy), goal) is not None:
                return (nx, ny)

        return self.jump((nx, ny), (nx + dx, ny + dy), goal)

    def build_path_message(self, path):
        path[0] = self.world_to_map(self.robot_pose.position.x, self.robot_pose.position.y)

        path = self.interpolate_linear_path(path, points_per_segment=5)

        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            wx, wy = self.map_to_world(x, y)
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def interpolate_linear_path(self, path, points_per_segment=5):
        interpolated_path = []

        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]

            for j in range(points_per_segment):
                t = j / points_per_segment
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                interpolated_path.append((x, y))

        interpolated_path.append(path[-1])  # ajoute le dernier point
        return interpolated_path


def main(args=None):
    rclpy.init(args=args)
    node = JPSPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
