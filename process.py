import os

import BallTrajectoryTracking
import consts


def process():
    folder_path = r".\film\learn\learn edited"
    for filename in os.listdir(folder_path):
        video_path = os.path.join(folder_path, filename)
        cap = consts.setup(video_path)
        name_without_extension = os.path.splitext(filename)[0]
        speed = float(name_without_extension.split('_')[0])
        angle = float(name_without_extension.split('_')[1].split('.')[0])
        BallTrajectoryTracking.TrackBallPos(cap, speed, angle)