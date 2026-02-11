import os

import BallTrajectoryTracking
import consts


def process():
    folder_path = "path/to/your/videos"
    for filename in os.listdir(folder_path):
        video_path = os.path.join(folder_path, filename)
        cap = consts.VideoCapture(video_path)
        name_without_extension = os.path.splitext(filename)[0]
        speed = name_without_extension.split('_')[0]
        BallTrajectoryTracking.TrackBallTrajectory(cap,speed)