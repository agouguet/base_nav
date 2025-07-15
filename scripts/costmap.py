#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

import numpy as np
import math
import time


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np
import math

class LocalCostmapNode(Node):
    def __init__(self):
        super().__init__('local_costmap_node')

        # Paramètres de la costmap
        self.resolution = 0.05  # m/cell
        self.size = 80  # 80x80 → 4m x 4m
        self.origin_x = -2.0  # origine centrée sur robot
        self.origin_y = -2.0

        self.costmap = -np.ones((self.size, self.size), dtype=np.int8)

        self.create_subscription(LaserScan, '/base_scan', self.scan_callback, 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/local_costmap', 10)

        self.timer = self.create_timer(0.2, self.publish_costmap)

        self.get_logger().info("Local costmap node started.")

    def scan_callback(self, msg: LaserScan):
        # self.costmap.fill(0)  # Réinitialiser à free space

        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                mx = int((x - self.origin_x) / self.resolution)
                my = int((y - self.origin_y) / self.resolution)

                if 0 <= mx < self.size and 0 <= my < self.size:
                    self.costmap[my, mx] = 100  # Obstacle

            angle += msg.angle_increment

    def publish_costmap(self):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'base_link'  # costmap locale autour du robot

        grid.info.resolution = self.resolution
        grid.info.width = self.size
        grid.info.height = self.size
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.orientation.w = 1.0

        grid.data = self.costmap.flatten().tolist()

        self.costmap_pub.publish(grid)
        self.get_logger().info("Local costmap publish.")


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
