# -*- coding: utf-8 -*-

from objLoader import loadOBJ
from objWriter import saveOBJ
from ICP import OnlyTransformICP
from ICP import ICP
import numpy as np
import math
import wx
from sklearn.decomposition import PCA
from multiprocessing import Pool,current_process
import time
import scipy.spatial as ss
import random
from Object_3D import Object_3D

#1が基準。2が変換をかけるほう
def varRatio(vectorArray1,vectorArray2) :
    mean = np.zeros(3)
    for i in vectorArray1:
        mean += i
    mean /= len(vectorArray1)

    var1 = 0.0
    for i in vectorArray1:
        i = i - mean
        var1 += np.dot(i,i)
    var1 /= len(vectorArray1)

    mean = np.zeros(3)
    for j in vectorArray2:
        mean += j
    mean /= len(vectorArray2)

    var2 = 0.0
    for j in vectorArray2:
        j = j - mean
        var2 += np.dot(j,j)
    var2 /= len(vectorArray2)
    return var1/var2

def triangleMean(vertexarray,faceVertexIds):
    trianglemean = []
    trimeanave = np.zeros(3)
    for index in faceVertexIds:
        v1 = vertexarray[index[0]]
        v2 = vertexarray[index[1]]
        v3 = vertexarray[index[2]]
        trimean = (v1 + v2 + v3)/3.0
        trianglemean.append(trimean)
        trimeanave += trimean

    trimeanave /= len(trianglemean)
    return trianglemean,trimeanave

def calcVertex(v):
    vertex = v[0]
    CTtrimeanave = v[1]
    transMat = v[2]
    Texturetrimeanave = v[3]

    r = vertex - CTtrimeanave
    r = np.dot(transMat,r)
    r = r + Texturetrimeanave
    return r


def useVertexCheck(vertices,faceVertIDs):
    UseCheck = [False] * len(vertices)
    for vertexIDs in faceVertIDs:
        for ID in vertexIDs:
            UseCheck[ID] = True

    print([i for i in UseCheck if i==False])
    useIDs = []
    for i,check in enumerate(UseCheck):
        if(check):
            useIDs.append(i)

    return useIDs

def Loss(args):
    start = time.time()
    Model = args[0]
    target_k_d_tree = args[1]
    err = 0.0
    LIMIT_POINT_NUM = 10**5
    SamplePointNum = 0

    if len(Model) > LIMIT_POINT_NUM:
        SamplePointNum = LIMIT_POINT_NUM
    else :
        SamplePointNum = len(Model)

    sourceSamplesIndices = random.sample(range( len(Model) ),SamplePointNum)
    for index in sourceSamplesIndices:
        nearlest = target_k_d_tree.query(Model[index])
        err += nearlest[0]
    err /= SamplePointNum
    
    end = time.time() - start
    print ("\nelapsed_time:{0}".format(end) + "[sec]\n")

    return err

def nearlestConversion(sourceModels,targetModel,Conversions):
    target_k_d_tree = ss.KDTree(targetModel)
    args = []

    start = time.time()
    for i in range(len(sourceModels)):
        args.append( [ sourceModels[i],target_k_d_tree,Conversions[i] ] )

    p = Pool(4)
    result = p.map(Loss,args)
    end = time.time() - start
    print ("\nelapsed_time:{0}".format(end) + "[sec]\n")
    minE = -1
    for i,err in enumerate(result):
        if err < minE or minE < 0:
            minE = err
            ret_transform = Conversions[i]
        
    p.close()
    return ret_transform


def positionfit(CTfilepath,Texturefilepath,Savefilepath,check,var = 1.0):
    try:
        CT_Object  = Object_3D(CTfilepath)
        Tex_Object = Object_3D(Texturefilepath) 
    except :
        print("ファイル入力エラー\n")
        return False

    TexPCA = PCA()
    TexPCA.fit(Tex_Object.getVertices())

    CTPCA = PCA()
    CTPCA.fit(CT_Object.getVertices())

    cov1 = TexPCA.get_covariance()
    cov2 = CTPCA.get_covariance ()

    eig1_val,eig1_vec = np.linalg.eig(cov1)
    eig2_val,eig2_vec = np.linalg.eig(cov2)

    eig1_val = np.sort(eig1_val)
    eig2_val = np.sort(eig2_val)

    print("Texture")
    print(eig1_val )
    print("\nCT"   )
    print(eig2_val )

    if not(check):
        var = math.sqrt(varRatio(Tex_Object.getVertices(),CT_Object.getVertices()))
    print("\n" + str(var))

    scaleMat  = np.array([ [var,0.0,0.0,0.0],
                           [0.0,var,0.0,0.0],
                           [0.0,0.0,var,0.0],
                           [0.0,0.0,0.0,1.0] ])

    CT_Center  = CT_Object .getPosition()

    transMat1 = np.array([ [1.0,0.0,0.0,-CT_Center[0]],
                           [0.0,1.0,0.0,-CT_Center[1]],
                           [0.0,0.0,1.0,-CT_Center[2]],
                           [0.0,0.0,0.0,1.0          ] ])

    transMat2 = np.array([ [1.0,0.0,0.0,CT_Center[0]],
                           [0.0,1.0,0.0,CT_Center[1]],
                           [0.0,0.0,1.0,CT_Center[2]],
                           [0.0,0.0,0.0,1.0         ] ])

    CT_Object.linerConversion( np.dot( transMat2,np.dot(scaleMat,transMat1) ) )


    TexPCARot = np.array(TexPCA.components_).transpose()

    TexPCARot = np.array([ [TexPCARot[0][0],TexPCARot[0][1],TexPCARot[0][2],0.0],
                           [TexPCARot[1][0],TexPCARot[1][1],TexPCARot[1][2],0.0],
                           [TexPCARot[2][0],TexPCARot[2][1],TexPCARot[2][2],0.0],
                           [0.0            ,0.0            ,0.0            ,1.0]])

    CTPCARot  = np.array(CTPCA.components_ )
    
    CTPCARot = np.array([ [CTPCARot[0][0],CTPCARot[0][1],CTPCARot[0][2],0.0],
                          [CTPCARot[1][0],CTPCARot[1][1],CTPCARot[1][2],0.0],
                          [CTPCARot[2][0],CTPCARot[2][1],CTPCARot[2][2],0.0],
                          [0.0           ,0.0           ,0.0           ,1.0] ])


    start = time.time()



    end   = time.time() - start
    print ("\nelapsed_time:{0}".format(end) + "[sec]\n")
    
    #この時点で複数のモデルのうち最もターゲットに近いモデル(正確には変換)を採用
    transMat= nearlestConversion(newvs,useTextureVertices,transMats)


    #デバッグ
    for t in transMats:
        print(t)
        print("\n")

    print(transMat)
    print("\n")
    #ここまで

    newv = np.zeros( (len(useCTVertices),3) )
    for i,v in enumerate(useCTVertices):
        args = [v,CTCenter,transMat,TexCenter]
        newv[i] = calcVertex(args)

    t = OnlyTransformICP(newv,useTextureVertices)

    newv = newv + t



    #実際のデータに変換を適用
    newv = np.zeros( (len(CTvertices),3) )
    newNormal = np.zeros( (len(normals2),3) )
    for i,v in enumerate(CTvertices):
        args = [v,CTCenter,transMat,TexCenter]
        newv[i] = calcVertex(args)

    #鏡面変換の検出
    if np.linalg.det(transMat)<0:
        for i in range(len(faceVertIDs2)):#CT
            tmp = faceVertIDs2[i][0]
            faceVertIDs2[i][0] = faceVertIDs2[i][1]
            faceVertIDs2[i][1] = tmp

    newv = newv + t

    try:
        print("ファイルセーブ")
        saveOBJ(Savefilepath, newv, uvs2, normals2, faceVertIDs2, uvIDs2, normalIDs2, vertexColors2)
    except:
        print("ファイルセーブエラー\n")
        return False

    paths = Savefilepath.split("\\")
    savef = paths[len(paths)-1]

    dialog = wx.MessageDialog(None, savef + 'を保存しました', 'メッセージ', style=wx.OK)
    dialog.ShowModal()
    return True
