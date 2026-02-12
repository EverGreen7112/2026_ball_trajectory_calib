import math
#from tty import ISPEED

import cv2 as cv
import numpy as np
import time

import aprilTagDetection
import consts
import recordData
#from consts import cap

ISPEED = 115200

BALL_RADIUS = 0.1501 / 2.0
realPosList = [[], [], [], []]  #0 - x, 1 - y, 2 - z, 3 - t
framePosList = []
polyCoefList = [[], [], []]
frame_height = 720
frame_width = 1280
fovY = math.degrees(2 * math.atan((frame_height) / (2 * consts.CAM_MTX[1][1])))


def TrackBallPos(cap, speed):
    global framePosList, realPosList, polyCoefList

    # Good: Data resets at the start of every video
    framePosList = []
    realPosList = [[], [], [], []]
    polyCoefList = [None, None, None]

    tag_to_cam_mtx = np.eye(4)
    startTime = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            # FIX 1: Break the loop so process.py can move to the next video
            print("Finished processing video.")
            break

        cam_to_tag_mtx = aprilTagDetection.aprilTag3dPosDetection(frame)
        if cam_to_tag_mtx is not None:
            tag_to_cam_mtx = np.linalg.inv(cam_to_tag_mtx)

        # ... (HSV and Contour logic) ...
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower = np.array([16, 220, 112])
        upper = np.array([26, 255, 255])
        mask = cv.inRange(hsv_frame, lower, upper)
        mask = cv.medianBlur(mask, 9)
        mask = cv.erode(mask, None, iterations=4)
        mask = cv.dilate(mask, None, iterations=4)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            ball = max(contours, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(ball)

            # Record data
            time_now = time.time() - startTime
            record_ball_3d_pos(Fx=x, Fy=y, frame_width=frame_width, frame_height=frame_height,
                               ball_radius=radius, time=time_now, tag_to_cam_mtx=tag_to_cam_mtx)

            framePosList.append((x, y))
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        # Draw trail
        for i in range(1, len(framePosList)):
            cv.line(frame, (int(framePosList[i - 1][0]), int(framePosList[i - 1][1])),
                    (int(framePosList[i][0]), int(framePosList[i][1])), (0, 0, 255), 1)

        # Good: Only calculate when we have enough data
        if len(realPosList[3]) >= 4:
            calc_ball_trajecktory_polynom_on_all_axis()

            # Draw the yellow mathematical prediction path
            draw_polynom_on_frame(frame, tag_to_cam_mtx)

            # Save results to JSON
            recordData.write_data(polyCoefList, 0, 0, 0, speed, 0, 0)

        # FIX 2: You need these lines to see the video on Windows
        cv.imshow('Processing...', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to skip a video
            break

    # Clean up window before starting the next video
    cv.destroyAllWindows()
def calc_ball_3d_pos(tag_to_cam_mtx, Fx, Fy, frame_width, frame_height, ball_radius):
    plainY = (2.0 * BALL_RADIUS * frame_height) / (2.0 * ball_radius)

    real_z = plainY / (2.0 * math.tan(math.radians(fovY) / 2.0))
    real_x = (2.0 * BALL_RADIUS * (Fx - (frame_width / 2.0))) / (2.0 * ball_radius)
    real_y = (2.0 * BALL_RADIUS * (Fy - (frame_height / 2.0))) / (2.0 * ball_radius)
    (real_x, real_y, real_z) = transform_ball_cam_space_to_abs_space(tag_to_cam_mtx, real_x, real_y, real_z)
    return (real_x, real_y, real_z)

def project_ball_3d_pos_to_screen(tag_to_cam_mtx, x, y, z):
    (x, y, z) = transform_ball_abs_space_to_cam_space(tag_to_cam_mtx, x, y, z)

    plainY = z * 2.0 * math.tan(math.radians(fovY) / 2.0)
    ball_radius = (BALL_RADIUS * frame_height) / plainY
    # frame_x = (x * 2.0 * ball_radius) / (2.0 * BALL_RADIUS) + (frame_width / 2.0)
    # frame_y = (y * 2.0 * ball_radius) / (2.0 * BALL_RADIUS) + (frame_hieght / 2.0)

    frame_x = (x * consts.CAM_MTX[0][0] / z) + (frame_width / 2)
    frame_y = (y * consts.CAM_MTX[1][1] / z) + (frame_height / 2)
    return (frame_x, frame_y, ball_radius)


def record_ball_3d_pos(Fx, Fy, frame_width, frame_height, ball_radius, time, tag_to_cam_mtx):
    real_x, real_y, real_z = calc_ball_3d_pos(tag_to_cam_mtx, Fx, Fy, frame_width, frame_height, ball_radius)
    realPosList[0].append(real_x)
    realPosList[1].append(real_y)
    realPosList[2].append(real_z)
    realPosList[3].append(time)


def calc_ball_trajecktory_polynom_on_all_axis():
    if len(realPosList[3]) < 4:
        return
    polyCoefList[0] = np.poly1d(np.polyfit(np.array(realPosList[3]), np.array(realPosList[0]), 3))
    polyCoefList[1] = np.poly1d(np.polyfit(np.array(realPosList[3]), np.array(realPosList[1]), 3))
    polyCoefList[2] = np.poly1d(np.polyfit(np.array(realPosList[3]), np.array(realPosList[2]), 3))
    print([a[0] for a in realPosList])
    print([c(realPosList[3][0]) for c in polyCoefList])


def Draw_polynom_on_frame(frame, tag_to_cam_mtx):
    ptsList = []
    for i, t in enumerate(realPosList[3]):
        x = polyCoefList[0](t)
        y = polyCoefList[1](t)
        z = polyCoefList[2](t)
        Px, Py, pr = project_ball_3d_pos_to_screen(tag_to_cam_mtx, x, y, z)
        cv.circle(frame, (int(Px), int(Py)), abs(int(pr)), (0, 255, 255), cv.FILLED)
        ptsList.append((Px, Py))
        if i > 0:
            intPos = (int(ptsList[i][0]), int(ptsList[i][1]))
            prevIntPos = (int(ptsList[i - 1][0]), int(ptsList[i - 1][1]))
            cv.line(frame, intPos, prevIntPos, (255, 0, 0))


def draw_polynom_on_frame(frame, tag_to_cam_mtx):
    ptsList = []

    # Check if we even have coefficients
    if polyCoefList[0] is None:
        print("No coefficients found!")
        return

    for i, t in enumerate(realPosList[3]):
        x = polyCoefList[0](t)
        y = polyCoefList[1](t)
        z = polyCoefList[2](t)

        # DEBUG: Let's see what the math is actually producing
        # print(f"3D Pos: {x:.2f}, {y:.2f}, {z:.2f}")

        Px, Py, pr = project_ball_3d_pos_to_screen(tag_to_cam_mtx, x, y, z)

        # DEBUG: See where the pixels are landing
        # print(f"Pixel Pos: {Px}, {Py}")

        # TEMPORARILY REMOVE the bounds check to see if pixels are just slightly off-screen
        cv.circle(frame, (int(Px), int(Py)), 5, (0, 255, 255), -1)

        ptsList.append((Px, Py))
        if len(ptsList) > 1:
            cv.line(frame, (int(ptsList[-2][0]), int(ptsList[-2][1])),
                    (int(ptsList[-1][0]), int(ptsList[-1][1])), (255, 0, 0), 2)

def transform_ball_cam_space_to_abs_space(tag_to_cam_mtx, cam_x, cam_y, cam_z):
    cam_ball_pos = np.array([cam_x, cam_y, cam_z, 1.0])
    ball_absolute_pos = tag_to_cam_mtx @ cam_ball_pos

    return ball_absolute_pos[:3]


def transform_ball_abs_space_to_cam_space(tag_to_cam_mtx, tag_x, tag_y, tag_z):
    inv_mtx = np.linalg.inv(tag_to_cam_mtx)
    cam_ball_pos = np.array([tag_x, tag_y, tag_z, 1])
    ball_absolute_pos = inv_mtx @ cam_ball_pos

    return ball_absolute_pos[:3]
def calc_intial_speed():
    x = polyCoefList[0][1]
    y = polyCoefList[1][1]
    z = polyCoefList[2][1]

    return (x,y,z)