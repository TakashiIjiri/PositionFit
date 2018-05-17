import numpy as np
import scipy.spatial as ss
import concurrent.futures
from multiprocessing import Pool,current_process
import time
import multiprocessing as multi
import random

def OnlyTransformICP(sourcePs,targetPs):
    prederr   = 0.0
    count     = 0
    no_move_count = 0
    threshold = 0.001
    k_d_targetPs = ss.KDTree(targetPs)
    result_t = np.zeros(3)
    
    LIMIT_POINT_NUM = 10**5
    LIMIT_ITR_COUNT = 30
    LIMIT_NO_MOVE_COUNT = 3
    if sourcePs.shape[0] > LIMIT_POINT_NUM:
        SamplePointNum = LIMIT_POINT_NUM
    else :
        SamplePointNum = sourcePs.shape[0]
    print("サンプリング数:",SamplePointNum)
    while(True):
        start = time.time()

        sourceSamplesIndices = random.sample(range(sourcePs.shape[0]),SamplePointNum)

        sourcePsCenterOfMass = np.zeros(3)
        nearPointindices     = []
        for index in sourceSamplesIndices:
            sourcePsCenterOfMass += sourcePs[index]
            nearPointindices.append(k_d_targetPs.query(sourcePs[index])[1])

        sourcePsCenterOfMass /= len(sourceSamplesIndices)

        delta      = np.zeros((len(nearPointindices),3))
        nearPsCenterOfMass  = np.zeros(3)
        for i,index in enumerate(nearPointindices):
            delta[i] = targetPs[index] - sourcePs[sourceSamplesIndices[i]]
            nearPsCenterOfMass += targetPs[index]

        nearPsCenterOfMass /= len(nearPointindices)

        transform = nearPsCenterOfMass - sourcePsCenterOfMass
        result_t  += transform
        sourcePs = sourcePs + transform

        err = 0.0
        for p in delta:
            err += np.linalg.norm(p)

        err /= delta.shape[0]

        if abs(err-prederr) < threshold:
            no_move_count += 1
            print("no move itr:",no_move_count)
        else:
            no_move_count = 0
        
        count += 1

        print(str(count)+"イテレーション")
        print("transform =",transform)
        print("err =",err)
        print("|prederr-err| =",abs(prederr-err))
        end = time.time()- start
        print ("elapsed_time:{0}".format(end) + "[sec]\n")

        prederr = err

        if no_move_count >= LIMIT_NO_MOVE_COUNT or count > LIMIT_ITR_COUNT:
            print("\nICP終了")
            print("result transform",result_t)
            return result_t



def ICP(sourcePs,targetPs):
    prederr   = 0.0
    count     = 0
    no_move_count = 0
    threshold = 0.001
    k_d_targetPs = ss.KDTree(targetPs)
    result_ts = []
    result_mats = []
    
    LIMIT_POINT_NUM = 10**5
    LIMIT_ITR_COUNT = 15
    LIMIT_NO_MOVE_COUNT = 3
    if sourcePs.shape[0] > LIMIT_POINT_NUM:
        SamplePointNum = LIMIT_POINT_NUM
    else :
        SamplePointNum = sourcePs.shape[0]
    print("サンプリング数:",SamplePointNum)

    while(True):
        start = time.time()

        sourceSamplesIndices = random.sample(range(sourcePs.shape[0]),SamplePointNum)

        sourcePsCenterOfMass = np.zeros(3)
        nearPointindices = []
        for index in sourceSamplesIndices:
            sourcePsCenterOfMass += sourcePs[index]
            nearPointindices.append(k_d_targetPs.query(sourcePs[index])[1])

        sourcePsCenterOfMass /= len(sourceSamplesIndices)


        nearPsCenterOfMass  = np.zeros(3)
        for index in nearPointindices:
            nearPsCenterOfMass += targetPs[index]

        nearPsCenterOfMass /= len(nearPointindices)


        #calc covariance
        cov = np.zeros( ( len(sourcePs[0]),len(sourcePs[0]) ) )
        for i,sourceindex in enumerate(sourceSamplesIndices):
            alpha = sourcePs[sourceindex]-sourcePsCenterOfMass  
            beta  = targetPs[nearPointindices[i]] - nearPsCenterOfMass 
            cov += np.dot( alpha.transpose() , beta )
        cov /= len(sourceSamplesIndices)

        U,s,V = np.linalg.svd(cov)

        m = np.diag([1,1,np.linalg.det(np.dot(U,V.transpose()))])

        rotMatrix = np.dot(U,np.dot(m,V.transpose()))
        transform = nearPsCenterOfMass - np.dot(rotMatrix,sourcePsCenterOfMass)

        result_ts.append(transform)
        result_mats.append(rotMatrix)

        for i in range(len(sourcePs)):
            sourcePs[i]  = np.dot(rotMatrix,sourcePs[i])
            sourcePs[i] += transform

        #ScaleFit
        x_scale=0
        y_scale=0
        z_scale=0
        bunbo = np.zeros(3)
        for i,index in enumerate(sourceSamplesIndices):
            targetPoint = targetPs[nearPointindices[i]]
            x_scale    += targetPoint[0]*sourcePs[index][0]
            y_scale    += targetPoint[1]*sourcePs[index][1]
            z_scale    += targetPoint[2]*sourcePs[index][2]
            bunbo      += np.square(sourcePs[index])
        
        x_scale /= bunbo[0]
        y_scale /= bunbo[1]
        z_scale /= bunbo[2]

        sourcePsCenterOfMass = np.zeros(3)
        for p in sourcePs:
            sourcePsCenterOfMass += p
        sourcePsCenterOfMass /= len(sourcePs)

        scaleMatrix = np.array([[x_scale,0      ,      0],
                                [0      ,y_scale,      0],
                                [0      ,0      ,z_scale]])

        for i in range(sourcePs.shape[0]):
            sourcePs[i] -= sourcePsCenterOfMass
            sourcePs[i]  = np.dot(scaleMatrix,sourcePs[i])
            sourcePs[i] += sourcePsCenterOfMass



        err = 0.0
        for i in range(len(sourceSamplesIndices)):
            delta = sourcePs[sourceSamplesIndices[i]] - targetPs[nearPointindices[i]]
            err += np.linalg.norm( delta )**2
        err /= len(sourceSamplesIndices)

        if abs(err-prederr) < threshold:
            no_move_count += 1
            print("no move itr:",no_move_count)
        else:
            no_move_count = 0
        
        count += 1

        print(str(count)+"イテレーション")
        print("transform =",transform)
        print("rotation ")
        print(rotMatrix)
        print("err =",err)
        print("|prederr-err| =",abs(prederr-err))
        end = time.time()- start
        print ("elapsed_time:{0}".format(end) + "[sec]\n")

        prederr = err

        if no_move_count >= LIMIT_NO_MOVE_COUNT or count > LIMIT_ITR_COUNT:
            return result_ts,result_mats


