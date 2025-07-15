import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('base_nav'),
        'config',
        'params.yaml'
        )

    global_plan_node=Node(
        package = 'base_nav',
        name = 'global_planner',
        executable = 'global_planner.py',
        parameters = [config]
    )
        
    controller_node=Node(
        package = 'base_nav',
        name = 'controller',
        executable = 'dwa_controller_social.py',
        parameters = [config]
    )

    multiplexer_node=Node(
        package = 'pkg-nav',
        name = 'multiplexer',
        executable = 'velocity_multiplexer.py',
        parameters = [config]
    )

    ld.add_action(global_plan_node)
    ld.add_action(controller_node)
    return ld