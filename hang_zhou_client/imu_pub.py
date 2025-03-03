# -*- coding: utf-8 -*-
"""

    用于提取imu 的特征， 再以消息的机制发出去
"""

import subprocess
import time
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32
from collections import deque 
import socket
import os

tmp_queue_size = 10

imu_x_publisher = rospy.Publisher('/data/data_x', Float32, queue_size=tmp_queue_size)
imu_y_publisher = rospy.Publisher('/data/data_y', Float32, queue_size=tmp_queue_size)
imu_z_publisher = rospy.Publisher('/data/data_z', Float32, queue_size=tmp_queue_size)


def imu_callback(data, index): # 可以不断的获取 imu 数据的函数
    
    if index == 1:
        x = data.linear_acceleration.x
        y = data.linear_acceleration.y
        z = data.linear_acceleration.z

        imu_x_publisher.publish(x)
        imu_y_publisher.publish(y)
        imu_z_publisher.publish(z)
            
        with open("data.txt" , "a") as f:
            f.write("{},{},{}\n".format(x, y, z))
    

def imu_listener():
    # 初始化节点
    rospy.init_node('shack_check_listener', anonymous=True)

    # 订阅 /imu/data 主题
    rospy.Subscriber('/imu/data', Imu, lambda data: imu_callback(data, 1))
    # rospy.Subscriber('/ltl_robot_basic_states', Int32, lambda data: imu_callback(data, 2))

    # 循环等待回调
    rospy.spin()

if __name__ == '__main__':
    try:
        imu_listener()
    except rospy.ROSInterruptException:
        pass
