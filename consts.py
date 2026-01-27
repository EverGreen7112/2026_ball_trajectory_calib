import numpy as np
import cv2 as cv
CAM_MTX = np.array([[1.33084514e+03, 0.00000000e+00, 6.28944423e+02],
                    [0.00000000e+00, 1.33752328e+03, 3.13907314e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DIST_COEF = np.array([[ 1.89464525e-01, -8.72067867e-01, -7.56249011e-03, -4.57008985e-05,
                        9.36110394e-01]])
TAG_SIZE = 0.1651

frame_height = 720
frame_width = 1280

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv.CAP_PROP_EXPOSURE,-7.5 )