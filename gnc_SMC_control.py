import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
import math
from pyproj import Geod
import numpy as np
import time
from gpsd_client.msg import GpsFix
from sbg_driver.msg import SbgGpsPos

geodesic = Geod(ellps='WGS84')
origin_latitude = 2
origin_longitude = 6
target_x = 56
target_y = 78
target_heading = 23

allocation_matrix = np.array([[1, 4, 5], [4, 7, 9], [2, 3, 4], [6, 8, 9]])

class GPSIMUHandler(Node):
    def __init__(self):
        super().__init__('gps_imu_handler')
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        
        self.create_subscription(GpsFix, "/gps/fix", self.gps_callback, 10)
        self.create_subscription(SbgGpsPos, "/sbg/gps_pos", self.imu_callback, 10)

    def gps_callback(self, data):
        my_latitude = data.latitude
        my_longitude = data.longitude
        result = geodesic.inv(origin_longitude, origin_latitude, my_longitude, my_latitude)
        self.x = result[0]
        self.y = result[1]

    def imu_callback(self, data):
        quaternion = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        euler_angles = euler_from_quaternion(quaternion)
        self.heading = euler_angles[2]  

class SlidingModeController:
    def __init__(self, lambda_gain, switching_gain):
        self.lambda_gain = lambda_gain  
        self.switching_gain = switching_gain  
        self.previous_error = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        sigma = error + self.lambda_gain * (error - self.previous_error) / dt if dt > 0 else 0.0
        control_signal = -self.switching_gain * np.sign(sigma)
        self.previous_error = error

        return control_signal

def main(args=None):
    rclpy.init(args=args)
    node = GPSIMUHandler()

    desired_x = target_x
    desired_y = target_y
    desired_heading = target_heading

    smc_controller = SlidingModeController(lambda_gain=0.5, switching_gain=1.0)

    while rclpy.ok():
        error_x = desired_x - node.x
        error_y = desired_y - node.y
        error_heading = desired_heading - node.heading
        error_matrix = np.array([error_x, error_y, error_heading])
        control_signals = [smc_controller.update(error) for error in error_matrix]
        thrust_output = np.dot(allocation_matrix, control_signals)
        node.get_logger().info(f"Thrust Allocation: {thrust_output}")
        rclpy.spin_once(node)

if __name__ == '__main__':
    main()
