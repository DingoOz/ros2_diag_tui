from setuptools import find_packages, setup
import os
from glob import glob

package_name = "ros2_diag_tui"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nigel_H-S",
    maintainer_email="1388693+DingoOz@users.noreply.github.com",
    description="A curses-based TUI dashboard for real-time ROS2 robot diagnostics",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "diag_tui = ros2_diag_tui.diag_tui:main",
        ],
    },
)
