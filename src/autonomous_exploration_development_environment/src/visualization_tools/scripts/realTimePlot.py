#!/usr/bin/env python3

"""
realTimePlot.py - Real-time visualization for TARE exploration metrics
Compatible with ROS 2 Humble

Subscribes to:
  - /explored_volume     (std_msgs/Float32): Total explored volume in m^3
  - /traveling_distance  (std_msgs/Float32): Cumulative traveling distance in m
  - /time_duration       (std_msgs/Float32): Elapsed time since start in s
  - /runtime             (std_msgs/Float32): Algorithm processing time per cycle in s

Usage:
  ros2 run visualization_tools realTimePlot.py

  Or with remapping if topic names differ:
  ros2 run visualization_tools realTimePlot.py --ros-args -r /explored_volume:=/tare/explored_volume
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import threading
import time


class RealTimePlotNode(Node):
    def __init__(self):
        super().__init__('real_time_plot')

        # Data storage
        self.time_data = []
        self.explored_volume_data = []
        self.traveling_distance_data = []
        self.runtime_data = []

        self.explored_volume = 0.0
        self.traveling_distance = 0.0
        self.time_duration = 0.0
        self.runtime = 0.0

        self.start_time_set = False
        self.start_time_value = 0.0

        # Graceful shutdown flags
        self.shutdown_requested = False
        self.spin_thread = None

        self.get_logger().info('RealTimePlot node started, waiting for data...')

        # Create subscriptions
        self.create_subscription(
            Float32,
            '/time_duration',
            self.time_duration_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/runtime',
            self.runtime_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/explored_volume',
            self.explored_volume_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/traveling_distance',
            self.traveling_distance_callback,
            10
        )

        # Create matplotlib figure
        self.setup_plot()

        # Connect window close event for graceful shutdown
        self.fig.canvas.mpl_connect('close_event', self.on_figure_close)

        # Start spin thread
        self.spin_thread = threading.Thread(target=self.spin_loop, daemon=True)
        self.spin_thread.start()

        # Animation timer - update plot at 10Hz
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=100,
            blit=False
        )

        plt.show()

    def spin_loop(self):
        """Background thread for ROS spinning."""
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)

        while rclpy.ok() and not self.shutdown_requested:
            executor.spin_once(timeout_sec=0.01)
            plt.pause(0.001)

        self.get_logger().info('Spin thread exiting gracefully')

    def on_figure_close(self, event):
        """Handle matplotlib window close event for graceful shutdown."""
        self.get_logger().info('Matplotlib window closed, initiating graceful shutdown...')
        self.shutdown_requested = True

        # Give spin thread time to exit
        if self.spin_thread is not None and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)

        # Shutdown rclpy
        if rclpy.ok():
            rclpy.shutdown()

        self.get_logger().info('Graceful shutdown complete')

    def setup_plot(self):
        """Initialize the matplotlib figure with 3 subplots."""
        # DEFENSIVE style loading: NEVER let style loading crash the node
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                try:
                    plt.style.use('ggplot')
                except Exception:
                    plt.style.use('default')

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(10, 9), sharex=False
        )
        self.fig.canvas.manager.set_window_title('TARE Exploration Metrics')
        self.fig.tight_layout(rect=[0, 0.08, 1, 0.96])

        # Subplot 1: Explored Volume
        self.ax1.set_ylabel('Explored Volume (m\u00b3)', fontsize=12)
        self.ax1.set_title('TARE Exploration Metrics - Real-time Monitoring', fontsize=14, fontweight='bold')
        self.line1, = self.ax1.plot([], [], color='#E74C3C', linewidth=2, label='Explored Volume')
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, 60)
        self.ax1.set_ylim(0, 1000)

        # Subplot 2: Traveling Distance
        self.ax2.set_ylabel('Traveling Distance (m)', fontsize=12)
        self.line2, = self.ax2.plot([], [], color='#3498DB', linewidth=2, label='Traveling Distance')
        self.ax2.legend(loc='upper left')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, 60)
        self.ax2.set_ylim(0, 200)

        # Subplot 3: Algorithm Runtime
        self.ax3.set_ylabel('Runtime (s)', fontsize=12)
        self.ax3.set_xlabel('Time (s)', fontsize=12)
        self.line3, = self.ax3.plot([], [], color='#2ECC71', linewidth=2, label='Algorithm Runtime')
        self.ax3.legend(loc='upper left')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(0, 60)
        self.ax3.set_ylim(0, 5)

        plt.ion()
        plt.draw()
        plt.pause(0.01)

    def time_duration_callback(self, msg):
        """Callback for /time_duration topic."""
        self.time_duration = msg.data
        if not self.start_time_set and self.time_duration > 0:
            self.start_time_value = self.time_duration
            self.start_time_set = True
            self.get_logger().info(
                f'Received first time_duration: {self.time_duration:.2f}s'
            )

    def runtime_callback(self, msg):
        """Callback for /runtime topic."""
        self.runtime = msg.data

    def explored_volume_callback(self, msg):
        """Callback for /explored_volume topic."""
        self.explored_volume = msg.data

    def traveling_distance_callback(self, msg):
        """Callback for /traveling_distance topic."""
        self.traveling_distance = msg.data

    def update_plot(self, frame):
        """Update the plot with new data. Called by FuncAnimation."""
        if not self.start_time_set or self.shutdown_requested:
            return

        current_time = self.time_duration - self.start_time_value
        if current_time < 0:
            return

        # Only update every few frames
        if len(self.time_data) == 0 or (current_time - self.time_data[-1]) > 0.2:
            self.time_data.append(current_time)
            self.explored_volume_data.append(self.explored_volume)
            self.traveling_distance_data.append(self.traveling_distance)
            self.runtime_data.append(self.runtime)

        if len(self.time_data) < 2:
            return

        # Limit data points
        max_points = 600
        if len(self.time_data) > max_points:
            self.time_data = self.time_data[-max_points:]
            self.explored_volume_data = self.explored_volume_data[-max_points:]
            self.traveling_distance_data = self.traveling_distance_data[-max_points:]
            self.runtime_data = self.runtime_data[-max_points:]

        time_arr = np.array(self.time_data)
        vol_arr = np.array(self.explored_volume_data)
        dist_arr = np.array(self.traveling_distance_data)
        rt_arr = np.array(self.runtime_data)

        # Update line data
        self.line1.set_data(time_arr, vol_arr)
        self.line2.set_data(time_arr, dist_arr)
        self.line3.set_data(time_arr, rt_arr)

        # Rescale axes
        max_time = max(time_arr[-1] + 10, 60) if len(time_arr) > 0 else 60

        self.ax1.set_xlim(0, max_time)
        self.ax2.set_xlim(0, max_time)
        self.ax3.set_xlim(0, max_time)

        vol_max = max(np.max(vol_arr) * 1.2 if len(vol_arr) > 0 else 1000, 100)
        dist_max = max(np.max(dist_arr) * 1.2 if len(dist_arr) > 0 else 200, 20)
        rt_max = max(np.max(rt_arr) * 1.2 if len(rt_arr) > 0 else 5, 0.5)

        self.ax1.set_ylim(0, vol_max)
        self.ax2.set_ylim(0, dist_max)
        self.ax3.set_ylim(0, rt_max)

        # Status text
        status_text = (
            f'Time: {current_time:.1f}s  |  '
            f'Volume: {self.explored_volume:.1f} m\u00b3  |  '
            f'Distance: {self.traveling_distance:.1f} m  |  '
            f'Runtime: {self.runtime:.4f} s'
        )
        self.fig.suptitle(status_text, fontsize=10, y=0.99)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None):
    """Main entry point."""
    try:
        rclpy.init(args=args)
        node = RealTimePlotNode()

        # Main thread will exit when window is closed
        # Spin thread handles ROS callbacks

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in realTimePlot: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
