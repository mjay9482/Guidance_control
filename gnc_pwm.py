#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float32
from pyproj import Geod
import math
import time
import tf_transformations

geo_calc = Geod(ellps='WGS84')

class PIDController(Node):

    def __init__(self, kp, kd, ki):
        super().__init__('pid_controller')
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.previous_error = 0
        self.integral = 0
        self.previous_time = self.get_clock().now().seconds_nanoseconds()[0]

        self.transformation_matrix = np.array([[0.707, -0.707, -0.707, 0.707],
                                               [0.707, 0.707, 0.707, 0.707],
                                               [0.35,  0.35,  0.35,  0.35]])

        self.inverse_transformation_matrix = np.linalg.pinv(self.transformation_matrix)
        
        self.coeff_a1 = np.array([1.52503318e-05, -4.91422601e-02,  3.93300223e+01])
        self.coeff_a2 = np.array([1.61019834e-05, -4.23940889e-02,  2.71121120e+01])
        self.coeff_b1 = np.array([1.67759057e-05, -5.46903629e-02,  4.41751206e+01])
        self.coeff_b2 = np.array([2.12621123e-05, -5.82516683e-02,  3.93836637e+01])

        self.gps_subscription = self.create_subscription(NavSatFix, '/mavros/global_position/raw/fix', self.gps_callback, 10)
        self.imu_subscription = self.create_subscription(Imu, '/mavros/imu/data', self.imu_callback, 10)

    def compute_control(self, error):
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        dt = current_time - self.previous_time
        dt = max(dt, 0.1)  
        
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        self.previous_error = error
        self.previous_time = current_time

        return (self.kp * error) + (self.kd * derivative) + (self.ki * self.integral)

    def calculate_fin_curve_l(self, thrust):
        return (0.4 * (self.coeff_a1[0] * (thrust ** 2) + self.coeff_a1[1] * thrust + self.coeff_a1[2])) + \
               (0.6 * (self.coeff_b1[0] * (thrust ** 2) + self.coeff_b1[1] * thrust + self.coeff_b1[2]))

    def calculate_fin_curve_r(self, thrust):
        return (0.4 * (self.coeff_a2[0] * (thrust ** 2) + self.coeff_a2[1] * thrust + self.coeff_a2[2])) + \
               (0.6 * (self.coeff_b2[0] * (thrust ** 2) + self.coeff_b2[1] * thrust + self.coeff_b2[2]))

    def inverse_fin_function(self, force):
        s1 = self.solve_quadratic(-self.coeff_a1[0], -self.coeff_a1[1], -self.coeff_a1[2] - force)
        s2 = self.solve_quadratic(self.coeff_a2[0], self.coeff_a2[1], self.coeff_a2[2] - force)

        if 0 < force < 0.5:
            for val in s2:
                if val > 1500:
                    pwm_output = val
        elif -0.5 < force < 0:
            for val in s1:
                if val < 1500:
                    pwm_output = val
        elif force <= -0.5:
            pwm_output = 1400
        elif force >= 0.5:
            pwm_output = 1600
        else:
            pwm_output = 1500

        return pwm_output

    def solve_quadratic(self, a, b, c):
        discriminant = b ** 2 - 4 * a * c
        sqrt_val = math.sqrt(abs(discriminant))
        if discriminant > 0:
            return [(-b + sqrt_val) / (2 * a), (-b - sqrt_val) / (2 * a)]
        elif discriminant == 0:
            return [(-b / (2 * a))]
        else:
            return [1500]

    def gps_callback(self, gps_data):
        lat = gps_data.latitude
        lon = gps_data.longitude

        waypoint = (self.origin_lat, self.origin_lon)  
        target_coords = geo_calc.inv(self.origin_lat, self.origin_lon, waypoint[0], waypoint[1])
        current_coords = geo_calc.inv(self.origin_lat, self.origin_lon, lon, lat)
        
        error_x = target_coords[0] - current_coords[0]
        error_y = target_coords[1] - current_coords[1]

    def imu_callback(self, imu_data):
        orientation = imu_data.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler_angles = tf_transformations.euler_from_quaternion(quaternion)
        yaw = euler_angles[2]
        yaw_error = self.desired_yaw - yaw  
        self.compute_control(yaw_error)


def main(args=None):
    rclpy.init(args=args)
    pid_controller = PIDController(0.3, 0.6, 0.2)  
    rclpy.spin(pid_controller)
    pid_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
