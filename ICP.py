from scipy.spatial import KDTree
import numpy as np

def calcRigidTranformation(MatA, MatB):
    A, B = np.copy(MatA), np.copy(MatB)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A -= centroid_A
    B -= centroid_B

    H = np.dot(A.T, B)
    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T, U.T)
    T = np.dot(-R, centroid_A) + centroid_B

    return R, T


class ICP(object):
    def __init__(self, targetPoints, sourcePoints):
        self.targetPoints = targetPoints
        self.sourcePoints = sourcePoints
        self.kdtree = KDTree(self.targetPoints)

    def calculate(self, iter):
        old_points = np.copy(self.sourcePoints)
        new_points = np.copy(self.sourcePoints)

        for i in range(iter):
            neighbor_idx = self.kdtree.query(old_points)[1]
            targets = self.targetPoints[neighbor_idx]
            R, T = calcRigidTranformation(old_points, targets)
            new_points = np.dot(R, old_points.T).T + T

            if  np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = np.copy(new_points)

        return new_points