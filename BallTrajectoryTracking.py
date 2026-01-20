import math
import cv2 as cv
import numpy as np
import time

import aprilTagDetection

BALL_RADIUS = 0.1501 / 2.0
fovY = 29.99
realPosList = [[], [], [], []]  # 0 - x, 1 - y, 2 - z, 3 - t
framePosList = []
polyCoefList = [[],[],[]]
cap = cv.VideoCapture(1, cv.CAP_DSHOW)
frame_hieght = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

def ballMain():
    TrackBallPos()
    calc_ball_trajecktory_polynom_on_all_axis()


def TrackBallPos():
    global framePosList, realPosList, polyCoefList
    start = False
    timeStampList = []
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        aprilTagDetection.aprilTag3dPosDetection(frame)

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])

        mask = cv.inRange(hsv_frame, lower, upper)
        mask = cv.medianBlur(mask, 9)

        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            ball = max(contours, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(ball)
            print(calc_ball_3d_pos(x, y, frame_width, frame_hieght, radius))
            px,py,pz = calc_ball_3d_pos(x,y,frame_width,frame_hieght,radius)
            X,Y,pr = project_ball_3d_pos_to_screen(px,py,pz)
            cv.circle(frame, (int(X), int(Y)), int(pr), (0, 255, 255), 5)

            if start:
                framePosList.append((x, y))
                timeStampList.append(time.time() - startTime)
                record_ball_3d_pos(x, y, frame_width, frame_hieght, radius, timeStampList[-1])

            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        for i in range(1, len(framePosList)):
            pos = framePosList[i]
            pos = (int(pos[0]), int(pos[1]))
            cv.circle(frame, (int(pos[0]), int(pos[1])), 2, (0, 255, 0), cv.FILLED)
            if (len(framePosList) > 0):
                cv.line(frame, pos, (int(framePosList[i - 1][0]), int(framePosList[i - 1][1])), (0, 0, 255), 1)
        cv.imshow('screen', mask)
        if len(realPosList[0]) >= 2:
            calc_ball_trajecktory_polynom_on_all_axis()
            draw_polynom_on_frame(frame, startTime, time.time())
        cv.imshow("pain", frame)
        k = cv.waitKey(1)
        if k == ord('q'):
            print(calc_ball_trajecktory_polynom_on_all_axis())
            if len(realPosList[0]) >= 2:
                calc_ball_trajecktory_polynom_on_all_axis()
                draw_polynom_on_frame(frame, startTime, time.time())
                cv.imshow("pain", frame)
                cv.waitKey(0)
            break
        if k == ord('s'):
            start = True
            startTime = time.time()
        if k == ord('r'):
            framePosList = []
            realPosList = [[], [], [], []]
            polyCoefList = [[],[],[]]


def calc_ball_3d_pos(Fx, Fy, frame_width, frame_height, ball_radius):

    plainY = (2.0 * BALL_RADIUS * frame_height) / (2.0 * ball_radius)
    real_z = plainY / (2.0 * math.tan(math.radians(fovY) / 2.0))
    real_x = (2.0 * BALL_RADIUS * (Fx - (frame_width / 2.0))) / (2.0 * ball_radius)
    real_y = -(2.0 * BALL_RADIUS * (Fy - (frame_height / 2.0))) / (2.0 * ball_radius)

    return (real_x, real_y, real_z)


def project_ball_3d_pos_to_screen(x, y, z):
    plainY = z * 2.0 * math.tan(math.radians(fovY) / 2.0)
    ball_radius = (BALL_RADIUS * frame_hieght) / plainY
    frame_x = (x * 2.0 * ball_radius) / (2.0 * BALL_RADIUS) + (frame_width / 2.0)
    frame_y = -(y * 2.0 * ball_radius) / (2.0 * BALL_RADIUS) + (frame_hieght / 2.0)
    return (frame_x, frame_y,ball_radius)


def record_ball_3d_pos(Fx, Fy, frame_width, frame_height, ball_radius, time):

    real_x, real_y, real_z = calc_ball_3d_pos(Fx, Fy, frame_width, frame_height, ball_radius)
    realPosList[0].append(real_x)
    realPosList[1].append(real_y)
    realPosList[2].append(real_z)
    realPosList[3].append(time)


def calc_ball_trajecktory_polynom_on_all_axis():
    polyCoefList[0] = np.poly1d(np.polyfit(np.array(realPosList[3]), np.array(realPosList[0]), 3))
    polyCoefList[1] = np.poly1d(np.polyfit(np.array(realPosList[3]), np.array(realPosList[1]), 3))
    polyCoefList[2] = np.poly1d(np.polyfit(np.array(realPosList[3]), np.array(realPosList[2]), 3))
    print([a[0] for a in realPosList])
    print([c(realPosList[3][0]) for c in polyCoefList])

def draw_polynom_on_frame(frame,startTime,endTime):

    ptsList = []
    for i,t in enumerate(realPosList[3]): #(int((endTime - startTime) / 0.1)):
        x = polyCoefList[0](t)
        y = polyCoefList[1](t)
        z = polyCoefList[2](t)
        Px, Py, pr = project_ball_3d_pos_to_screen(x, y, z)
        cv.circle(frame, (int(Px), int(Py)), abs(int(pr)), (0, 255, 255), cv.FILLED)
        ptsList.append((Px, Py))
        if i > 0 :
            intPos = (int(ptsList[i][0]),int(ptsList[i][1]))
            prevIntPos = (int(ptsList[i-1][0]),int(ptsList[i-1][1]))
            cv.line(frame,intPos,prevIntPos,(255,0,0))