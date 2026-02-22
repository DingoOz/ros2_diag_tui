"""Launch file for ROS2 Diagnostics TUI."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for ROS2 Diagnostics TUI."""
    return LaunchDescription(
        [
            Node(
                package="ros2_diag_tui",
                executable="diag_tui",
                name="ros2_diag_tui",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
