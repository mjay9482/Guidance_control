#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
from scipy.optimize import curve_fit
from controller_pwm import PIDController
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Int32MultiArray, Float32, Float32MultiArray
from pyproj import Geod
import time

g = Geod(ellps='WGS84')

pidc = PIDController(0.1, 0.2, 0.05)

goal_position = np.array([0, 0, 0])  
initial_position = np.array([0, 0])  
initialized = False
circle_of_acceptance = 1  
look_ahead_distance = 2
current_x = 0  
current_y = 0  
current_yaw = 0  

previous_goal = np.array([0, 0, 0])
thrust_values = [1500, 1500, 1500, 1500]

def guider_callback(data):
    global goal_position, previous_goal
    x_goal = data.pose.position.x  
    y_goal = data.pose.position.y  
    if x_goal != previous_goal[0] or y_goal != previous_goal[1]:
        previous_goal[:2] = goal_position[:2]
    goal_position[:2] = [x_goal, y_goal]

def test(p, a, b):
    return (a * p + b)

def distance(x1, y1, x2, y2):
    return math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))

def cross_track(a, b, c, d, e, f):  
    origin = [a, b]
    goal = [c, d]
    p = np.array([goal[0], origin[0]])
    q = np.array([goal[1], origin[1]])
    param, _ = curve_fit(test, p, q)
    xt = e + param[0] * (abs(param[0] * e + param[1] - f) /
                         math.sqrt(param[0]**2 + param[1]**2))
    yt = f + param[1] * (abs(param[0] * e + param[1] - f) /
                         math.sqrt(param[0]**2 + param[1]**2))
    return xt, yt

def is_inside_circle(circle_x, circle_y, radius, x, y):  
    return (x - circle_x) ** 2 + (y - circle_y) ** 2 <= radius ** 2

def imu_callback(data):
    quaternion = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
    euler = tf.transformations.euler_from_quaternion(quaternion)  
    global current_yaw
    current_yaw = euler[2]  
    yaw_message = Float32()
    yaw_message.data = current_yaw
    yaw_publisher.publish(yaw_message)

def gps_callback(data):
    global current_x, current_y, initialized

    latitude = float(data.latitude)
    longitude = float(data.longitude)

    if not initialized:
        initial_position[:] = [latitude, longitude]
        previous_goal[:2] = [0, 0]
        goal_position[:2] = previous_goal[:2]
        initialized = True

    coord = g.inv(initial_position[0], initial_position[1], latitude, longitude)
    current_x, current_y = coord[0], coord[1]

def calculate_slope(x1, y1, x2, y2):
    if x1 - x2 == 0:
        return math.pi / 2
    else:
        return math.atan(abs((y1 - y2) / (x1 - x2)))

def stop():
    thrust_message = Int32MultiArray(data=[1500, 1500, 1500, 1500])
    thrust_publisher.publish(thrust_message)

def guide():
    global current_x, current_y
    xt, yt = cross_track(previous_goal[0], previous_goal[1], goal_position[0], goal_position[1], current_x, current_y)

    alpha = math.atan(distance(xt, yt, current_x, current_y) / look_ahead_distance)
    path_slope = calculate_slope(previous_goal[0], previous_goal[1], goal_position[0], goal_position[1])
    desired_yaw = path_slope - alpha

    error_x = xt - current_x  
    error_y = yt - current_y  
    error_yaw = desired_yaw - current_yaw  

    Fx = pidc.update(error_x)
    Fy = pidc.update(error_y)
    Fn = pidc.update(error_yaw)

    control_pwm = np.dot(pidc.pseudo_TA_inv, np.array([Fx, Fy, Fn]))

    left_front_thrust = control_pwm[0]  
    right_front_thrust = control_pwm[1]  
    left_aft_thrust = control_pwm[2] 
    right_aft_thrust = control_pwm[3]  

    thrust_values[0] = int(pidc.inverse_thrust(left_front_thrust))
    thrust_values[1] = int(pidc.inverse_thrust(right_front_thrust))
    thrust_values[2] = int(pidc.inverse_thrust(right_aft_thrust))
    thrust_values[3] = int(pidc.inverse_thrust(left_aft_thrust))

    thrust_message = Int32MultiArray(data=thrust_values)
    thrust_publisher.publish(thrust_message)

    debug_publisher.publish(Float32(data=left_front_thrust))

def main():
    if is_inside_circle(goal_position[0], goal_position[1], circle_of_acceptance, current_x, current_y):
        print("Goal Reached!")
        stop()

if __name__ == '__main__':
    rclpy.init()
    node = rclpy.create_node('guider')

    rate = node.create_rate(5)
    
    yaw_publisher = node.create_publisher(Float32, '/imu/yaw', 10)
    thrust_publisher = node.create_publisher(Int32MultiArray, '/thrust_val', 10)
    debug_publisher = node.create_publisher(Float32, '/debug_val', 10)

    node.create_subscription(NavSatFix, '/mavros/global_position/raw/fix', gps_callback, 10)
    node.create_subscription(PoseStamped, '/guider/goalpose', guider_callback, 10)
    node.create_subscription(Imu, '/mavros/imu/data', imu_callback, 10)

    while rclpy.ok():
        try:
            guide()
            main()
            rate.sleep()
        except rclpy.exceptions.ROSInterruptException:
            pass
