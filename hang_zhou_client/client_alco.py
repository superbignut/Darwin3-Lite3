# -*- coding: utf-8 -*-
from time import sleep
import matplotlib.pyplot as plt
import serial
import time
import numpy as np
import random  # 模拟酒精浓度数据
import serial.tools.list_ports
import socket
import sys 
import threading
import traceback
# print(sys.version.split()[0].split('.')[0] == '3')

plist = list(serial.tools.list_ports.comports())
# if sys.version.split()[0].split('.')[0] == '3':
# print plist[0][0]



def send_data_thread(host='192.168.1.103', socket_port=12345):

    # 设置串口参数
    port = plist[0][0]  # 串口设备路径
    print port
    baudrate = 9600  # 波特率
    ser = serial.Serial(port, baudrate, timeout=1)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, socket_port))


    try:
        last_a = 5000
        while True:
            message = ser.readline().strip()

            if message:
                
                parsed_values = [byte for byte in message] 

                if len(parsed_values) == 9:
                    
                    a = ord(parsed_values[4]) * 256 + ord(parsed_values[5])
                    b = ord(parsed_values[6]) * 256 + ord(parsed_values[7])

                    print a

                    if a - last_a >=200:
                        data = "alco " + str(1) + " " + str(0) 
                        client_socket.sendall(data.encode('utf-8'))

                    # print "alcohol is: {}/{}".format(a, b)
                    last_a =a
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
        # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
        # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
        print(traceback.format_exception(
            exc_type,
            exc_value,
            exc_traceback
        ))
    finally:
        client_socket.close()
        ser.close()

        

if __name__=="__main__":
    alcohol_thread = threading.Thread(target=send_data_thread, name="send_data_thread")
    alcohol_thread.start()



