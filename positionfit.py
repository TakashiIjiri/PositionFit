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
        vertices1, uvs1, normals1, faceVertIDs1, uvIDs1, normalIDs1, vertexColors1 = loadOBJ(Texturefilepath)
        vertices2, uvs2, normals2, faceVertIDs2, uvIDs2, normalIDs2, vertexColors2 = loadOBJ(CTfilepath)
    except :
        print("ファイル入力エラー\n")
        return False

    Texturevertices = np.array(vertices1)
    CTvertices      = np.array(vertices2)

    useTextureVIndices = useVertexCheck(Texturevertices,faceVertIDs1)
    useCTVIndices      = useVertexCheck(CTvertices     ,faceVertIDs2)

    useTextureVertices = np.zeros( (len(useTextureVIndices ), 3) )
    useCTVertices      = np.zeros( (len(useCTVIndices      ), 3) ) 

    #定義されているがメッシュ作成に使われない頂点が存在する場合があるため
    #使われている頂点のみから変換を計算する
    for i,index in enumerate(useTextureVIndices):
        useTextureVertices[i] = Texturevertices[index]

    for i,index in enumerate(useCTVIndices):
        useCTVertices[i] = CTvertices[index]

    CTCenter  = np.zeros(3)
    TexCenter = np.zeros(3)

    for v in useCTVertices:
        CTCenter  += v
    CTCenter  /= len(useCTVertices)

    for v in Texturevertices:
        TexCenter += v
    TexCenter /= len(useTextureVertices)

    TexPCA = PCA()
    TexPCA.fit(useTextureVertices)

    CTPCA = PCA()
    CTPCA.fit(useCTVertices)

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
        var = math.sqrt(varRatio(useTextureVertices,useCTVertices))
    print("\n" + str(var))

    TexPCARot = np.array(TexPCA.components_).transpose()
    CTPCARot  = np.array(CTPCA.components_ )
    scaleMat  = np.array([ [var,0.0,0.0],
                          [0.0,var,0.0],
                          [0.0,0.0,var] ])
    rotMat1   = np.dot(TexPCARot,CTPCARot)
    transMats = []
    transMats.append(np.dot(scaleMat,rotMat1))

    milerMat = np.diag([-1.0,1.0,-1.0]) #y軸回転
    rotMat2  = np.dot(TexPCARot,np.dot(milerMat,CTPCARot))
    transMats.append(np.dot(scaleMat,rotMat2))

    milerMat = np.diag([-1.0,-1.0,1.0]) #z軸回転
    rotMat3  = np.dot(TexPCARot,np.dot(milerMat,CTPCARot))
    transMats.append(np.dot(scaleMat,rotMat3))
    
    milerMat = np.diag([1.0,-1.0,-1.0]) #x軸回転
    rotMat4  = np.dot(TexPCARot,np.dot(milerMat,CTPCARot))
    transMats.append(np.dot(scaleMat,rotMat4))

    start = time.time()

    newvs = []
    for j,transMat in enumerate(transMats):
        newvs.append(np.zeros( (len(useCTVertices),3) ))
        for i,v in enumerate(useCTVertices):
            args    = [v,CTCenter,transMat,TexCenter]
            newvs[j][i] = calcVertex(args)

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
