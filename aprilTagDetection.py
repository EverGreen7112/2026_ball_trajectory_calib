import math

import cv2 as cv

import consts
import cv2
import numpy as np
import pupil_apriltags as apriltag

at_detector = apriltag.Detector(families='tag36h11')


def aprilTag3dPosDetection(frame):


    # Load image and convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Camera parameters [fx, fy, cx, cy] and tag size in meters
    camera_params = [consts.CAM_MTX[0][0], consts.CAM_MTX[1][1], consts.CAM_MTX[0][2], consts.CAM_MTX[1][2]]

    results = at_detector.detect(gray, estimate_tag_pose=True,
                                 camera_params=camera_params,
                                 tag_size=consts.TAG_SIZE)

    for r in results:
        if r.tag_id != 4:
            break
        rotation_matrix = r.pose_R  # 3x3 rotation matrix
        translation_vector = r.pose_t  # 3x1 translation vector

        # Corners for visualization
        (ptA, ptB, ptC, ptD) = r.corners
        half_size = consts.TAG_SIZE / 2.0
        object_points = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(object_points, r.corners, consts.CAM_MTX, consts.DIST_COEF,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        transformation_matrix = None
        if success:
            cv2.drawFrameAxes(frame, consts.CAM_MTX, consts.DIST_COEF, rvec, tvec, 0.05)
            tvec[1] *= -1
            # Convert rotation vector (rvec) to rotation matrix (R)
            R, _ = cv2.Rodrigues(rvec)

            # Create a 4x4 homogeneous transformation matrix
            # transformation_matrix = np.eye(4)
            # transformation_matrix[:3, :3] = R
            # transformation_matrix[:3, 3] = tvec.flatten()
            # Combine R and t into a 3x4 extrinsic matrix
            extrinsic_matrix = np.hstack((R, tvec))

            # Create the bottom row [0, 0, 0, 1]
            bottom_row = np.array([0, 0, 0, 1]).reshape(1, 4)

            # Combine to form the 4x4 homogeneous transformation matrix
            transformation_matrix = np.vstack((extrinsic_matrix, bottom_row))

            #print("\nTransformation Matrix (Tag relative to Camera):\n", transformation_matrix)
            #print(f"\nTranslation Vector (tvec in meters): {tvec.flatten()}")
            abs = math.sqrt(sum([t**2 for t in tvec]))
            # print(abs)
            #print(r.tag_id)
            # Optional: draw the axes on the image
            return transformation_matrix

