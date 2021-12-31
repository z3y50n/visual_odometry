#!/usr/bin/env python3
import sys

import cv2
import numpy as np

from display import Display
from frame import Frame
from estimator import Estimator

W, H = 1164 // 2, 874 // 2
F = 910
K = np.asarray([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
disp = Display(W, H)
est = Estimator(K)


def process_frame(frame):
    frame.extract()
    if frame.prev is None:
        return

    matches = frame.match()
    Rt, inliers = est.estimate(matches)
    frame.pose = Rt @ frame.prev.pose
    roll, pitch, yaw = est.euler(frame.pose)

    print(frame.pose)
    # print(roll, pitch, yaw)

    for match in matches[inliers]:
        cv2.circle(frame.img, match[0], radius=3, color=(0, 255, 0))
        cv2.arrowedLine(frame.img, match[0], match[1], color=(255, 0, 0))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./vo.py <video_path>")
        exit(0)
    video = sys.argv[1]
    vid = cv2.VideoCapture(video)

    prev = None
    while vid.isOpened():
        ret, frame = vid.read()

        if not ret:
            break

        frame = Frame(frame, prev, W, H)
        process_frame(frame)
        disp.draw(frame.img)
        prev = frame
