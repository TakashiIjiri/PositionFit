import numpy as np
import time
import scipy.spatial as ss
import random

def ScaleFit(SourcePs,TargetPs):
    print("scale fitting...")

    target_kdtree = ss.KDTree(TargetPs)
    LIMIT_POINT_NUM = 10**5
    if SourcePs.shape[0] > LIMIT_POINT_NUM:
        SamplePointNum = LIMIT_POINT_NUM
    else :
        SamplePointNum = SourcePs.shape[0]

    preerr = 10000

    result_xscale = 1.0
    result_yscale = 1.0
    result_zscale = 1.0

    TRESHOLD = 0.0001
    LIMIT_ITR_COUNT = 5
    count = 0
    noMoveCount = 0

    while(True):
        start = time.time()
        SourceSamplesIndices = random.sample(range(SourcePs.shape[0]),SamplePointNum)

        nearPointIndices = []
        for index in SourceSamplesIndices:
            nearPointIndices.append(target_kdtree.query(SourcePs[index])[1])

        x_scale=0
        y_scale=0
        z_scale=0
        bunbo = np.zeros(3)
        for i,index in enumerate(SourceSamplesIndices):
            targetPoint = TargetPs[nearPointIndices[i]]
            x_scale    += targetPoint[0]*SourcePs[index][0]
            y_scale    += targetPoint[1]*SourcePs[index][1]
            z_scale    += targetPoint[2]*SourcePs[index][2]
            bunbo      += np.square(SourcePs[index])
        
        x_scale /= bunbo[0]
        y_scale /= bunbo[1]
        z_scale /= bunbo[2]

        x_scale = abs(x_scale)
        y_scale = abs(y_scale)
        z_scale = abs(z_scale)

        SourceCenter = np.zeros(3)
        for vertex in SourcePs:
            SourceCenter += vertex
        SourceCenter /= len(SourcePs)
        scaleMatrix = np.array([[x_scale,0      ,      0],
                                [0      ,y_scale,      0],
                                [0      ,0      ,z_scale]])


        for i in range(len(SourcePs)):
            SourcePs[i] -= SourceCenter
            SourcePs[i]  = np.dot(scaleMatrix,SourcePs[i])
            SourcePs[i] += SourceCenter


        err = 0.0
        
        for i,index in enumerate(SourceSamplesIndices):
            targetPoint = TargetPs[nearPointIndices[i]]
            err += np.linalg.norm(targetPoint - SourcePs[index])
        err /= len(SourceSamplesIndices)
            
        end = time.time() - start

        if abs(err - preerr) < TRESHOLD or count > LIMIT_ITR_COUNT:
            noMoveCount += 1
            print(str(x_scale)+"  "+str(y_scale)+"  "+str(z_scale))
            print("no move")
            print ("\nelapsed_time:{0}".format(end) + "[sec]\n")
            if noMoveCount > 3 or count > LIMIT_ITR_COUNT:
                return result_xscale , result_yscale , result_zscale
        
        else :
            noMoveCount = 0
            count += 1
            preerr = err
            result_xscale *= x_scale
            result_yscale *= y_scale
            result_zscale *= z_scale

            print(str(x_scale)+"  "+str(y_scale)+"  "+str(z_scale))
            print ("\nelapsed_time:{0}".format(end) + "[sec]\n")

