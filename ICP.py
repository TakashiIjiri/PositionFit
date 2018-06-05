# from
# http://nghiaho.com/?page_id=671
# http://clientver2.hatenablog.com/entry/2015/11/27/160814

from scipy.spatial import KDTree
import numpy as np
import random
import time




# MatA, MatB --> pair of vertices (same size and has correspondences)
# return rigid transformation (Rotation and transformation)
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


class ICP( object ):
    def __init__(self, targetPoints, sourcePoints):
        self.targetPoints = targetPoints
        self.sourcePoints = sourcePoints
        self.kdtree = KDTree(self.targetPoints)

    def calculate(self, iter, LIMIT_POINT_NUM=10**5,THRESHOLD=0.001):
        old_points = np.copy(self.sourcePoints)
        new_points = np.copy(self.sourcePoints)

        if len( self.sourcePoints ) >LIMIT_POINT_NUM:
            SAMPLE_POINT_NUM = LIMIT_POINT_NUM
        else:
            SAMPLE_POINT_NUM = len( self.sourcePoints )

        for i in range(iter):
            start = time.time()

            sourceSamplesIndices = random.sample( range( len(self.sourcePoints) ), SAMPLE_POINT_NUM )
            samplePoints = old_points[sourceSamplesIndices]

            neighbor_idx = self.kdtree.query(samplePoints)[1]
            targets      = self.targetPoints[neighbor_idx]
            R, T         = calcRigidTranformation(samplePoints, targets)
            new_points   = np.dot(R, old_points.T).T + T

            end = time.time() - start
            print ("\nelapsed_time:{0}".format(end) + "[sec]\n")

            if  np.mean( np.abs(old_points - new_points) ) < THRESHOLD:
                break

            old_points = np.copy(new_points)

        return new_points
