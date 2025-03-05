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


imu_x_publisher = rospy.Publisher('/queue/sum_x', Float32, queue_size=10)
imu_y_publisher = rospy.Publisher('/queue/sum_y', Float32, queue_size=10)
# imu_z_publisher = rospy.Publisher('/queue/sum_z', Float32, queue_size=10)

x_buffer = deque(maxlen=30, iterable=[0 for _ in range(30)])
y_buffer = deque(maxlen=30, iterable=[0 for _ in range(30)]) # 长一点， 这样的话，更新的慢一点 

current_time = time.time()
last_trigger_time = current_time
trigger_duration = 2.0 # 秒内触发一次



host = '192.168.1.103'
socket_port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, socket_port))

IMU_Touching = 1            # 抚摸 * 2
IMU_Hit = 2                 # 拍打

is_static = True
def imu_callback(data, index): # 可以不断的获取 imu 数据的函数
    # 发布 x 值
    global last_trigger_time, current_time, trigger_duration, is_static, client_socket # , IMU_Touching, IMU_Hit
    
    """ if index == 2:
        if data.data == 1 or data.data == 6:
            is_static = True
        else:
            is_static = False """

    if index == 1 and is_static == True:
        x = data.linear_acceleration.x
        y = data.linear_acceleration.y
        z = data.linear_acceleration.z
        
        x_buffer.append(abs(x)) # 都取正
        y_buffer.append(abs(y))

        temp_average_x = sum(x_buffer) / len(x_buffer)
        temp_average_y = sum(y_buffer) / len(y_buffer)

        imu_x_publisher.publish(temp_average_x)
        imu_y_publisher.publish(temp_average_y)

        current_time = time.time()

        if current_time - last_trigger_time > trigger_duration: #  限制了触发间隔
            
            if abs(abs(z)-9.8) > 2:
                print("有坏人，快跑!!!")
                # shuo_zhong_wen("有坏人，快跑？")
                print("")
                last_trigger_time = current_time
                # subprocess.Popen("/home/ysc/.pyenv/shims/python3 /home/ysc/ltl/youhuairen_kuaipao.py", shell=True)

                data = "imu " + "2 0 0"  # 2 敲打
                client_socket.sendall(data.encode('utf-8'))
                
            elif temp_average_x > 1.0 or temp_average_y > 1.2: # 这里的 坐下的时候， 加速度还要重新标定以前是1.2和1.4
                print("there is a touching !!")
                #shuo_zhong_wen("是谁在摸我？")
                last_trigger_time = current_time
                # subprocess.Popen("/home/ysc/.pyenv/shims/python3 /home/ysc/ltl/shishei_zaimowo.py", shell=True)

                data = "imu " + "1 0 0"  #  1 表示 抚摸
                client_socket.sendall(data.encode('utf-8'))
            

            
    

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
