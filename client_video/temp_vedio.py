"""
    client_video的本地版，用来调试代码
"""
import cv2 as cv
import numpy as np
import struct
import threading
import time
import os 
import sys
from opencv_zoo.models.person_detection_mediapipe.mp_persondet import MPPersonDet
from opencv_zoo.models.palm_detection_mediapipe.mp_palmdet import MPPalmDet
from opencv_zoo.models.handpose_estimation_mediapipe.mp_handpose import MPHandPose

from utils.RoI import RoIHumanDetMP
from utils.HandGesture import HandGesture

# socket 第一个参数
Color_Red = 1               # 红颜色
Color_Blue = 2              # 蓝颜色
Color_Black = 3             # 黑颜色

# socket 第二个参数
Gesture_Like = 4            # 点赞手势 
Gesture_Dislike = 5         # 点踩手势 / 拳头 0 
Gesture_Palm = 6            # 手掌手势 5 

# socket 第三个参数
Cmd_LieDown = 7             # 趴下指令 1 
Cmd_StandUp = 8             # 站起来指令 2
Cmd_GoAhead = 9             # 向前走指令 3
Cmd_GoBack = 10             # 向后走指令 4
Cmd_Woof = 11               # 往往叫指令 



if __name__ == '__main__':

    cap = cv.VideoCapture(0) # 这里测试了在狗上也能捕捉到

    if cv.cuda.getCudaEnabledDeviceCount() != 0:
        backend = cv.dnn.DNN_BACKEND_CUDA
        target = cv.dnn.DNN_TARGET_CUDA
    else:
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        print('CUDA is not set, will fall back to CPU.')

    # human detector, used to determine where a person is to reduce the area of interest
    human_detector_mp = MPPersonDet(
        modelPath='utils/person_detection_mediapipe_2023mar.onnx',
        nmsThreshold=0.3,
        scoreThreshold=0.3,  # lower to prevent missing human body
        topK=1,  # just only one person
        backendId=backend,
        targetId=target)

    # palm detector
    palm_detector = MPPalmDet(
        modelPath='utils/palm_detection_mediapipe_2023feb.onnx',
        nmsThreshold=0.3,
        scoreThreshold=0.4,  # lower to  prevent missing palms
        topK=5,  # maximum 2 palms to make sure right hand can be detected # origin=500
        backendId=backend,
        targetId=target)
    # handpose detector
    handpose_detector = MPHandPose(
        modelPath='utils/handpose_estimation_mediapipe_2023feb.onnx',
        confThreshold=0.6,  # higher to prevent mis-estimation
        backendId=backend,
        targetId=target)

    human_RoI_mp = RoIHumanDetMP(human_detector_mp) # 这两个都是检测人体 先用一个
    
    
    hand_gesture = HandGesture(palm_detector, handpose_detector) # 这个使用 mediapipe 检测手势
    
    
    gesture_buffer = [None] * 3 # 这个长度用来检测手势的判定

    pixels_buffer = np.zeros(3) 

    lower_red = np.array([100, 0, 0])   # 红色的最低范围
    upper_red = np.array([255, 80, 80]) # 红色的最高范围

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])


    while True:
        args1 = 0   # socket 第一个参数
        args2 = 0   # socket 第二个参数
        args3 = 0   # socket 第三个参数

        ret, frame = cap.read()
        if ret is None or not ret:
            continue

        # detect RoI by human detection
        bbox = human_RoI_mp.detect(frame)
        image = frame
        gestures = None

        if bbox is not None: # 其实这个人体检测 ，如果没有的话，似乎会更快一点

            upper_body_RoI = human_RoI_mp.get_full_RoI() # 这里如果要使用全屏检测手势的话， 需要改成[[0,0],[640, 480]] 

            gestures, area_list = hand_gesture.estimate(frame, upper_body_RoI)
            # print(gestures, "1111") # gestures 是字符串的列表

            cloth = human_RoI_mp.get_cloth_RoI()
            # print(cloth)
            
            if cloth is not None:
                        
                # 重塑cloth为(2, 2)并确保为整数
                cloth = cloth.reshape(-1).astype(int)
                x1, y1, x2, y2 = cloth
                # print(x1, y1, x2, y2)
                
                # 提取布料的RoI
                cloth_image = frame[y1:y2, x1:x2]
    
                # 将RoI转换为HSV颜色空间
                image_rgb = cv.cvtColor(cloth_image, cv.COLOR_BGR2RGB)
    
                # 创建掩码
                red_mask = cv.inRange(image_rgb, lower_red, upper_red) # 在范围内的是255 其余变成0
                blue_mask = cv.inRange(image_rgb, lower_blue, upper_blue) # 在范围内的是255 其余变成0
                black_mask = cv.inRange(image_rgb, lower_black, upper_black) # 在范围内的是255 其余变成0

                # 计算红色区域的像素数量
                pixels_buffer[0] = cv.countNonZero(red_mask) # 计算非零区域
                pixels_buffer[1] = cv.countNonZero(blue_mask)
                pixels_buffer[2] = cv.countNonZero(black_mask)

                # 计算总像素数量
                total_pixels = image_rgb.shape[0] * image_rgb.shape[1] # 统计总像素数

                # 最大颜色编号
                max_color = pixels_buffer.argmax()
                
                # 计算最大颜色的占比
                _ratio = pixels_buffer[max_color] * 5 *100 // total_pixels # 多乘了5 作为放大系数 python2 是整数
                
                if _ratio > 8:  # 这个参数用来调节颜色的判定阈值

                    data = "color " + str(max_color + 1) + " " + str(0) # 红1 蓝2 黑3
                    args1 = max_color + 1
                    # print(data)
                    # Todo  这里把手势加到第二个参数上
                
                    # client_socket.sendall(data.encode('utf-8'))
                # print( "红色占比, " , _ratio)
                # time.sleep(0.4)
                # cv.rectangle(image, (int(cloth[0]), int(cloth[1])), (int(cloth[2]), int(cloth[3])), (0, 255, 0), 1)

        cv.imshow("Demo", image)
        k = cv.waitKey(1) # 画图必备
        # control robot dog
        if gestures is not None and gestures.shape[0] != 0: # gestures有两个维度 第一个应该是 图像 第二个是分类结果
            # only use the biggest area right hand
            idx = area_list.argmax()
            gesture_buffer.insert(0, gestures[idx]) # 
            gesture_buffer.pop() # 每插入一个就要 pop一个
            # only if the gesture is the same 3 times, the corresponding command will be executed
            if gesture_buffer[0] is not None and all(ges == gesture_buffer[0] for ges in gesture_buffer):
                # print(gesture_buffer[0])
                # 下面是指令手势
                if gesture_buffer[0] == 'one':
                    args3 = Cmd_LieDown
                elif gesture_buffer[0] == 'two':
                    args3 = Cmd_StandUp
                elif gesture_buffer[0] == 'three':
                    args3 = Cmd_GoAhead
                elif gesture_buffer[0] == 'four':
                    args3 = Cmd_GoBack
                
                # 下面是非指令手势
                elif gesture_buffer[0] == 'like':
                    args2 = Gesture_Like
                elif gesture_buffer[0] == 'zero':
                    args2 = Gesture_Dislike
                elif gesture_buffer[0] == 'five':
                    args2 = Gesture_Palm
        
        if args1 !=0 or args2 != 0 or args3 != 0: # 只要有一个参数有就发
            
            data = "vedio " + str(args1) + " " + str(args2) + " " + str(args3) # 红1 蓝2 黑3
            print(data)
            # time.sleep(0.5)
# 给models 加了一个init