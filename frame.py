import cv2
import numpy as np


class Frame:
    def __init__(self, img, prev, W, H):
        self._img = cv2.resize(img, (W, H))
        self._prev = prev
        self.W, self.H = W, H
        self._kps = None
        self._des = None
        self._pose = np.eye(4)

    @property
    def img(self):
        return self._img

    @property
    def prev(self):
        return self._prev

    @property
    def kps(self):
        return self._kps

    @property
    def des(self):
        return self._des

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose

    def extract(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()

        feats = cv2.goodFeaturesToTrack(gray, 1000, qualityLevel=0.01, minDistance=10)

        kps = [cv2.KeyPoint(*f[0], size=3) for f in feats]
        kps, des = orb.compute(self.img, kps)
        self._kps = kps
        self._des = des

    def match(self):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(self.des, self.prev.des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                pt1 = np.int0(self.kps[m.queryIdx].pt)
                pt2 = np.int0(self.prev.kps[m.trainIdx].pt)
                good.append((pt1, pt2))
        return np.asarray(good)
