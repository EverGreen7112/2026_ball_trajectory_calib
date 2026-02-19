import os

import BallTrajectoryTracking
import consts
from aprilTagDetection import get_transformation
import numpy as np

def process():
    folder_path = r".\film\learn\learn edited"
    for filename in os.listdir(folder_path):
        video_path = os.path.join(folder_path, filename)
        cap_for_transform = consts.setup(video_path)
        transform = get_transformation(cap_for_transform)
        print(f"determinant: {np.linalg.det(transform)}")
        cap =  consts.setup(video_path)
        name_without_extension = os.path.splitext(filename)[0]
        speed = float(name_without_extension.split('_')[0])
        angle = float(name_without_extension.split('_')[1].split('.')[0])
        BallTrajectoryTracking.TrackBallPos(cap, speed, angle, transform)