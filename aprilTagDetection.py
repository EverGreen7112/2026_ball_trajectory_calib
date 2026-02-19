import math

import consts
import cv2
import numpy as np
import pupil_apriltags as apriltag

at_detector = apriltag.Detector(families='tag36h11', quad_sigma=0.8)
R_TOLERANCE = 0.0008
T_TOLERANCE = 0.05

prev_tvec = None
prev_rvec = None
def aprilTag3dPosDetection(frame):
    global prev_rvec, prev_tvec
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

        success_v1, rvec_v1, tvec_v1 = cv2.solvePnP(object_points, r.corners, consts.CAM_MTX, consts.DIST_COEF,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        success_v2, rvec_v2, tvec_v2 = cv2.solvePnP(object_points, r.corners, consts.CAM_MTX, consts.DIST_COEF,
                                           flags=cv2.SOLVEPNP_SQPNP)
        transformation_matrix = None
        if (success_v1 and success_v2 and
            np.linalg.norm(rvec_v1 - rvec_v2) % (2 * np.pi) < R_TOLERANCE and
            np.linalg.norm(tvec_v1 - tvec_v2) < T_TOLERANCE):
            tvec = 0.5 * (tvec_v1 + tvec_v2)
            rvec = rvec_v2 #0.5 * ((rvec_v1 % (2 * np.pi))  + (rvec_v2 % (2 * np.pi)))

            raw_tvec = tvec
            raw_rvec = rvec

            if (prev_rvec is not None and prev_tvec is not None):
                tvec = (prev_tvec + tvec) / 2
                rvec = (prev_rvec + rvec) / 2



            cv2.drawFrameAxes(frame, consts.CAM_MTX, consts.DIST_COEF, rvec, tvec, 0.05)
            #tvec[1] *= -1
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
            abs = np.sqrt(np.sum([t**2 for t in tvec]))
            # print(abs)
            #print(r.tag_id)
            # Optional: draw the axes on the image
            prev_tvec = raw_tvec
            prev_rvec = raw_rvec

            transformation_matrix = transformation_matrix @ consts.SHOOTER_MTX
        else:
            if prev_tvec is not None and prev_rvec is not None:
                cv2.drawFrameAxes(frame, consts.CAM_MTX, consts.DIST_COEF, prev_rvec, prev_tvec, 0.05)
        return transformation_matrix

def get_transformation(cap: cv2.VideoCapture) -> np.ndarray | None:
    global prev_rvec, prev_tvec
    prev_tvec = None
    prev_rvec = None
    tvec_sum = np.array([[0.0], [0.0], [0.0]])
    rvec_sum = np.array([[0.0], [0.0], [0.0]])
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("finished calculating matrix")
            break
        transform = aprilTag3dPosDetection(frame)
        if transform is not None:
            tvec_sum += prev_tvec
            rvec_sum += prev_rvec
            count += 1
    if count > 0:
        tvec = tvec_sum / count
        rvec = rvec_sum / count
        R, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((R, tvec))

        # Create the bottom row [0, 0, 0, 1]
        bottom_row = np.array([0, 0, 0, 1]).reshape(1, 4)

        # Combine to form the 4x4 homogeneous transformation matrix
        transformation_matrix = np.vstack((extrinsic_matrix, bottom_row))
        return transformation_matrix # @ consts.SHOOTER_MTX
    return None

def draw_axis(frame: np.ndarray):
    if prev_tvec is not None and prev_rvec is not None:
        cv2.drawFrameAxes(frame, consts.CAM_MTX, consts.DIST_COEF, prev_rvec, prev_tvec, 0.05)