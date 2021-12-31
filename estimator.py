import math

import cv2
import numpy as np
from skimage.transform import EssentialMatrixTransform
from skimage.measure import ransac

np.set_printoptions(suppress=True)


class Estimator:
    def __init__(self, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def estimate(self, matches):
        pts1 = matches[:, 0, :]
        pts2 = matches[:, 1, :]

        pts1 = self.normalize(pts1)
        pts2 = self.normalize(pts2)

        model, inliers = ransac(
            (pts1, pts2),
            # FundamentalMatrixTransform,
            EssentialMatrixTransform,
            min_samples=8,
            residual_threshold=0.001,
            max_trials=500,
        )
        Rt = self.extractRt(model.params, pts1, pts2)

        return Rt, inliers

    def extractRt(self, E, pts1, pts2):
        ret, R, t, mask = cv2.recoverPose(E, pts1, pts2)
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.reshape(3)
        return Rt

    def euler(self, Rt):
        roll = math.atan2(Rt[2, 1], Rt[2, 2])
        pitch = math.atan2(-Rt[2, 0], math.sqrt(Rt[2, 1] ** 2 + Rt[2, 2] ** 2))
        yaw = math.atan2(Rt[1, 0], Rt[0, 0])
        return roll, pitch, yaw

    def _add_ones(self, pts):
        return np.vstack((pts.T, np.ones(pts.shape[0])))

    def normalize(self, pts):
        return (self.Kinv @ self._add_ones(pts)).T[:, :2]

    def denormalize(self, pts):
        return (self.K @ self._add_ones(pts)).T[:, :2].astype(np.int0)
