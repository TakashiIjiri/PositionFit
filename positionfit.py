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

def ScaleFit(SourcePs,TargetPs):
    print("scale fitting...")

    target_kdtree = ss.KDTree(TargetPs)
    LIMIT_POINT_NUM = 10**5
    if SourcePs.shape[0] > LIMIT_POINT_NUM:
        SamplePointNum = LIMIT_POINT_NUM
    else :
        SamplePointNum = SourcePs.shape[0]

    preerr = 0

    result_xscale = 1.0
    result_yscale = 1.0
    result_zscale = 1.0

    TRESHOLD = 0.001
    LIMIT_ITR_COUNT = 30
    count = 0

    while(True):
        start = time.time()
        SourceSamplesIndices = random.sample(range(SourcePs.shape[0]),SamplePointNum)

        nearPointIndices = []
        for index in SourceSamplesIndices:
            nearPointIndices.append(TargetPs[target_kdtree.query(SourcePs[index])[1]])


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

        if abs(err - preerr) < TRESHOLD or count > LIMIT_ITR_COUNT:
            return result_xscale , result_yscale , result_zscale
        
        else :
            count += 1
            preerr = err
            result_xscale *= x_scale
            result_yscale *= y_scale
            result_zscale *= z_scale

            end = time.time() - start
            print ("\nelapsed_time:{0}".format(end) + "[sec]\n")


def useVertexCheck(vertices,faceVertIDs):
    useVertices = np.copy(vertices)
    UseCheck = [False] * len(vertices)
    for vertexIDs in faceVertIDs:
        for ID in vertexIDs:
            UseCheck[ID] = True

    print([i for i in UseCheck if i==False])
    nonUseIDs = []
    for i,check in enumerate(UseCheck):
        if(not(check)):
            nonUseIDs.append(i)

    if len(nonUseIDs) > 0:
        useVertices = np.delete(useVertices,nonUseIDs,0)
    return useVertices

def Loss(args):
    start = time.time()
    Model = args[0]
    target_k_d_tree = args[1]
    Conversion = args[2]
    err = 0.0
    LIMIT_POINT_NUM = 5**6
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
        

    return ret_transform


def positionfit(CTfilepath,Texturefilepath,Savefilepath,check,var = 1.0):
    P = Pool(2)
    try:
        vertices1, uvs1, normals1, faceVertIDs1, uvIDs1, normalIDs1, vertexColors1 = loadOBJ(Texturefilepath)
        vertices2, uvs2, normals2, faceVertIDs2, uvIDs2, normalIDs2, vertexColors2 = loadOBJ(CTfilepath)
    except :
        print("ファイル入力エラー\n")
        return False

    Texturevertices = np.array(vertices1)
    CTvertices      = np.array(vertices2)

    useTextureVertices = useVertexCheck(Texturevertices,faceVertIDs1)
    useCTVertices      = useVertexCheck(CTvertices     ,faceVertIDs2)

    TexturetriangleMean,Texturetrimeanave = triangleMean(Texturevertices,faceVertIDs1)
    CTtriangleMean     ,CTtrimeanave      = triangleMean(CTvertices     ,faceVertIDs2)

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

    rot1 = np.array(TexPCA.components_).transpose()
    rot2 = np.array(CTPCA.components_ )
    scaleMat = np.array([[var,0.0,0.0],
                         [0.0,var,0.0],
                         [0.0,0.0,var]])
    rotMat1 = np.dot(rot1,rot2)
    transMats = []
    transMats.append(np.dot(scaleMat,rotMat1))

    milerMat = np.diag([-1.0,1.0,1.0])
    rotMat2 = np.dot(rot1,np.dot(milerMat,rot2))
    transMats.append(np.dot(scaleMat,rotMat2))

    milerMat = np.diag([1.0,-1.0,1.0])
    rotMat3  = np.dot(rot1,np.dot(milerMat,rot2))
    transMats.append(np.dot(scaleMat,rotMat3))
    
    milerMat = np.diag([1.0,1.0,-1.0])
    rotMat4  = np.dot(rot1,np.dot(milerMat,rot2))
    transMats.append(np.dot(scaleMat,rotMat4))

    start = time.time()

    newvs = []
    for j,transMat in enumerate(transMats):
        newvs.append(np.zeros((len(useCTVertices),3)))
        for i,v in enumerate(useCTVertices):
            args    = [v,CTtrimeanave,transMat,Texturetrimeanave]
            newvs[j][i] = calcVertex(args)

    end   = time.time() - start
    print ("\nelapsed_time:{0}".format(end) + "[sec]\n")
    
    #この時点で複数のモデルのうち最もターゲットに近いモデル(正確には変換)を採用
    transMat= nearlestConversion(newvs,useTextureVertices,transMats)

    
    newv = np.zeros( (len(useCTVertices),3) )
    for i,v in enumerate(useCTVertices):
        args = [v,CTtrimeanave,transMat,Texturetrimeanave]
        newv[i] = calcVertex(args)

    t = OnlyTransformICP(newv,useTextureVertices)

    newv = newv + t

    if not(check):
        x_scale,y_scale,z_scale = ScaleFit(newv,useTextureVertices)
    else:
        print("no check")
        x_scale,y_scale,z_scale = 1.0,1.0,1.0

    print(x_scale,y_scale,z_scale)
    scaleMatrix = np.array([[x_scale,0      ,      0],
                            [0      ,y_scale,      0],
                            [0      ,0      ,z_scale]])




    newv = np.zeros( (len(CTvertices),3) )
    for i,v in enumerate(CTvertices):
        args = [v,CTtrimeanave,transMat,Texturetrimeanave]
        newv[i] = calcVertex(args)

    newv = newv + t

    newvCenterOfMass = np.zeros(3)
    for i in newv:
        newvCenterOfMass += i
    newvCenterOfMass /= newv.shape[0]

    for i in range(newv.shape[0]):
        newv[i] -= newvCenterOfMass
        newv[i]  = np.dot(scaleMatrix,newv[i])
        newv[i] += newvCenterOfMass

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
