# -*- coding: utf-8 -*-
"""

    用于提取imu 的特征， 再以消息的机制发出去
"""

import subprocess
import time
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float64
from collections import deque 
import socket
import os


x_buffer = deque(maxlen=30, iterable=[0 for _ in range(30)])



host = '192.168.1.103'
socket_port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, socket_port))


def power_callback(data): # 可以不断的获取 imu 数据的函数
    # 发布 x 值
    global client_socket
    x = data.data

    # print(x)

    if x < 20.0:
        send_data = "power " + "1 0"  #  1 表示 没电了
        time.sleep(2)
    else:
        send_data = "power " + "0 0"  #  0 表示 正常
        time.sleep(2)
    client_socket.sendall(send_data.encode('utf-8'))
            
    

def power_listener():
    # 初始化节点
    rospy.init_node('power_checker', anonymous=True)

    # 订阅 /imu/data 主题
    rospy.Subscriber('/ltl_battery', Float64, power_callback)

    # 循环等待回调
    rospy.spin()

if __name__ == '__main__':
    try:
        power_listener()
    except rospy.ROSInterruptException:
        pass
