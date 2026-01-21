import numpy as np
import cv2

# Setup: Change this to match your board (internal corners)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

CHESSBOARD_SIZE = (9, 6)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane



while True:
    ok,img = cap.read()
    if not ok :
        continue

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        cv2.circle(img, (int(corners[0][0][0]), int(corners[0][0][1])), 10, (255, 0, 0), 2)
        if k == ord('s'):
            objpoints.append(objp)
            imgpoints.append(corners)
            if len(imgpoints) > 100:
                break
    cv2.imshow("frame", img)
print("starting processing, this may take a while")
# The Magic Step: returns the Camera Matrix (mtx) and Distortion (dist)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix:\n", mtx)
print("Distortion Coeffs:\n", dist)