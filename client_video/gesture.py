# -*- coding: utf-8 -*-
"""
这个文件是用来 做 手势的初步测试的，因为以前一直是用 高翔的， 看看opencv 能不能做的简单一点
"""
import cv2 as cv
import numpy as np
import struct
import threading
import time

from opencv_zoo.models.person_detection_mediapipe.mp_persondet import MPPersonDet
from opencv_zoo.models.object_detection_nanodet.nanodet import NanoDet
from opencv_zoo.models.face_detection_yunet.yunet import YuNet
from opencv_zoo.models.palm_detection_mediapipe.mp_palmdet import MPPalmDet
from opencv_zoo.models.handpose_estimation_mediapipe.mp_handpose import MPHandPose

from utils.RoI import RoIHumanDetMP, RoIObjDetNano, RoIFaceDetYuNet
from utils.HandGesture import HandGesture
# from Controller import Controller
from utils.ColorDetection import ColorDetection

Develop_Mode = True  # True means use computer camera. False means use dog camera

if __name__ == '__main__':

    # get raw video frame
    if Develop_Mode:
        cap = cv.VideoCapture(0) # 这里测试了在狗上也能捕捉到
    # from Robot Dog
    else:
        cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)


    # try to use CUDA
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
    # nano detector
    human_detector_nano = NanoDet(
        modelPath='utils/object_detection_nanodet_2022nov.onnx',
        prob_threshold=0.5,
        iou_threshold=0.6,
        backend_id=backend,
        target_id=target)
    # face detector
    face_detector = YuNet(modelPath='utils/face_detection_yunet_2023mar.onnx',  # 2022-> 2023
                          confThreshold=0.6,  # lower to make sure mask face can be detected
                          nmsThreshold=0.3,
                          topK=5000,  # only one face
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
    # human_RoI_nano = RoIObjDetNano(human_detector_nano)


    # face_RoI_yunet = RoIFaceDetYuNet(face_detector)
    hand_gesture = HandGesture(palm_detector, handpose_detector) # 这个使用 mediapipe 检测手势
    # mask_detector = ColorDetection(np.array([86, 28, 141]), np.array([106, 128, 225]))

    # gesture will be recognized only if the gesture is the same 2 times in a row
    gesture_buffer = [None] * 3
    while True:
        ret, frame = cap.read()
        if ret is None or not ret:
            continue

        # detect RoI by human detection
        bbox = human_RoI_mp.detect(frame)
        image = frame
        gestures = None
        # find a person by MediaPipe human detection
        if bbox is not None:
            # usually upper body RoI can be gotten
            upper_body_RoI = human_RoI_mp.get_upper_RoI() # 这里如果要使用全屏检测手势的话， 需要改成[[0,0],[640, 480]] 
            # print("this is :", upper_body_RoI)
            # time.sleep(1)
            gestures, area_list = hand_gesture.estimate(frame, upper_body_RoI)
            # face_RoI_yunet.detect(frame, upper_body_RoI)
            # face_RoI = face_RoI_yunet.get_face_RoI()
            # mask_detector.detect(frame, face_RoI)
            # print(bbox)
            cloth = human_RoI_mp.get_cloth_RoI()
            # print(bbox)
            if cloth is not None:
                cloth = cloth.reshape(-1)
                cv.rectangle(image, (int(cloth[0]), int(cloth[1])), (int(cloth[2]), int(cloth[3])), (0, 255, 0), 1) # 这里把人的位置画出来

            image = hand_gesture.visualize(image) # 手势可视化 与连线
        # if human detection can't find a person, try NanoDet # 另一种方法
        """ else:
            continue # 这里暂时先用一种方法
            bbox = human_RoI_nano.detect(frame)
            human_RoI = human_RoI_nano.get_human_RoI()
            gestures, area_list = hand_gesture.estimate(frame, human_RoI) """
            # face_RoI_yunet.detect(frame, human_RoI)
            # face_RoI = face_RoI_yunet.get_face_RoI()
            # mask_detector.detect(frame, face_RoI)

        # visualize
            
        # image = mask_detector.visualize(image, "mask")

        cv.imshow("Demo", image)
        k = cv.waitKey(1)
        if k == 113 or k == 81:  # q or Q to quit
            if not Develop_Mode:
                # controller.drive_dog("squat")
                pass
            cap.release()
            cv.destroyWindow("Demo")
            break

        # control robot dog
        if gestures is not None and gestures.shape[0] != 0: # gestures有两个维度 第一个应该是 图像 第二个是分类结果
            # only use the biggest area right hand
            idx = area_list.argmax()
            gesture_buffer.insert(0, gestures[idx])
            gesture_buffer.pop()
            # only if the gesture is the same 3 times, the corresponding command will be executed
            if not Develop_Mode or (
                    gesture_buffer[0] is not None and all(ges == gesture_buffer[0] for ges in gesture_buffer)):
                # controller.drive_dog(gesture_buffer[0])
                pass