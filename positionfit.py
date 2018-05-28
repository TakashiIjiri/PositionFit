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


def varRatio(vectorArray1,vectorArray2) :
    mean = np.zeros(3)
    for i in vectorArray1:
        mean += i
    mean /= len(vectorArray1)

    var1 = 0.0
    vec= np.zeros(3)
    for i in vectorArray1:
        vec = i - mean
        var1 += np.dot(vec,vec)
    var1 /= len(vectorArray1)

    mean = np.zeros(3)
    for j in vectorArray2:
        mean += j
    mean /= len(vectorArray2)

    var2 = 0.0
    vec= np.zeros(3)
    for j in vectorArray2:
        vec = j - mean
        var2 += np.dot(vec,vec)
    var2 /= len(vectorArray2)
    return var1/var2


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

def Loss(CT_Object,Tex_Object):
    CT_vertices  = CT_Object .getVertices_3D()
    Tex_vertices = Tex_Object.getVertices_3D()

    target_KDTree = ss.KDTree(Tex_Object.getVertices_3D())

    LIMIT_POINT_NUM = 10**5

    if len( CT_vertices ) >LIMIT_POINT_NUM:
        SAMPLE_POINT_NUM = LIMIT_POINT_NUM
    else:
        SAMPLE_POINT_NUM = len( CT_vertices )

    sourceSamplesIndices = random.sample(range(len( CT_vertices )),SAMPLE_POINT_NUM)

    err = 0.0
    for index in sourceSamplesIndices:
        err += target_KDTree.query(CT_vertices[index])[0]

    return err / len(sourceSamplesIndices)


def nearlestModel(CT_Object,Tex_Object,CT_PCARot,TexPCARot):

    CT_Center = CT_Object.getPosition()
    #重心座標系に直すための行列
    transMat1 = np.array([ [1.0,0.0,0.0,-CT_Center[0]],
                           [0.0,1.0,0.0,-CT_Center[1]],
                           [0.0,0.0,1.0,-CT_Center[2]],
                           [0.0,0.0,0.0,       1.0   ]
                        ])

    transMat2 = np.array([ [1.0,0.0,0.0,CT_Center[0]],
                           [0.0,1.0,0.0,CT_Center[1]],
                           [0.0,0.0,1.0,CT_Center[2]],
                           [0.0,0.0,0.0,       1.0  ]
                        ])

    CT_Object.linerConversion( np.dot( transMat2,np.dot( CT_PCARot,transMat1 ) ) )

    I_Mat = np.diag([ 1.0, 1.0, 1.0,1.0])
    Rot_X = np.diag([ 1.0,-1.0,-1.0,1.0])
    Rot_Y = np.diag([-1.0, 1.0,-1.0,1.0])
    Rot_Z = np.diag([-1.0,-1.0, 1.0,1.0])

    nearlestConversionMat = None
    minErr = -1.0
    for Rot in [I_Mat,Rot_X,Rot_Y,Rot_Z]:
        Rot = np.dot(TexPCARot,Rot)

        CT_Object.linerConversion( np.dot( transMat2,np.dot( Rot,transMat1 ) ) )

        err = Loss(CT_Object,Tex_Object)
        print("err = ",err)
        if err < minErr or minErr < 0:
            minErr = err
            nearlestConversionMat =  Rot
            print(nearlestConversionMat)

        CT_Object.linerConversion( np.dot( transMat2,np.dot( Rot.transpose(),transMat1 ) ) )

    CT_Object.linerConversion( np.dot( transMat2,np.dot( nearlestConversionMat,transMat1 ) ) )


def positionfit(CTfilepath,Texturefilepath,Savefilepath,check,var = 1.0):
    try:
        CT_Object  = Object_3D(CTfilepath)
        Tex_Object = Object_3D(Texturefilepath)
    except :
        print("file input err\n")
        return False

    TexPCA = PCA()
    TexPCA.fit(Tex_Object.getVertices_3D())

    CTPCA = PCA()
    CTPCA.fit(CT_Object.getVertices_3D())

    Tex_Cov = TexPCA.get_covariance()
    CT_Cov  = CTPCA.get_covariance ()

    eig1_val,eig1_vec = np.linalg.eig(Tex_Cov)
    eig2_val,eig2_vec = np.linalg.eig(CT_Cov )

    eig1_val = np.sort(eig1_val)
    eig2_val = np.sort(eig2_val)

    print("Texture")
    print(eig1_val )
    print("\nCT"   )
    print(eig2_val )

    if not(check):
        var = varRatio(CT_Object.getVertices_3D(),Tex_Object.getVertices_3D())
    print("\n" + str(var))

    scaleMat  = np.array([ [var,0.0,0.0,0.0],
                           [0.0,var,0.0,0.0],
                           [0.0,0.0,var,0.0],
                           [0.0,0.0,0.0,1.0] ])

    CT_Center  = CT_Object .getPosition()
    Tex_Center = Tex_Object.getPosition()

    transMat1 = np.array([ [1.0,0.0,0.0,-CT_Center[0]],
                           [0.0,1.0,0.0,-CT_Center[1]],
                           [0.0,0.0,1.0,-CT_Center[2]],
                           [0.0,0.0,0.0,1.0          ] ])

    transMat2 = np.array([ [1.0,0.0,0.0,Tex_Center[0]],
                           [0.0,1.0,0.0,Tex_Center[1]],
                           [0.0,0.0,1.0,Tex_Center[2]],
                           [0.0,0.0,0.0,1.0          ] ])

    CT_Object.linerConversion( np.dot( transMat2,np.dot(scaleMat,transMat1) ) )


    TexPCARot = np.array(TexPCA.components_).transpose()

    TexPCARot = np.array([ [TexPCARot[0][0],TexPCARot[0][1],TexPCARot[0][2],0.0],
                           [TexPCARot[1][0],TexPCARot[1][1],TexPCARot[1][2],0.0],
                           [TexPCARot[2][0],TexPCARot[2][1],TexPCARot[2][2],0.0],
                           [0.0            ,0.0            ,0.0            ,1.0]])

    CT_PCARot  = np.array(CTPCA.components_ )

    CT_PCARot = np.array([ [CT_PCARot[0][0],CT_PCARot[0][1],CT_PCARot[0][2],0.0],
                           [CT_PCARot[1][0],CT_PCARot[1][1],CT_PCARot[1][2],0.0],
                           [CT_PCARot[2][0],CT_PCARot[2][1],CT_PCARot[2][2],0.0],
                           [0.0            ,0.0            ,0.0            ,1.0] ])


    start = time.time()

    nearlestModel(CT_Object,Tex_Object,CT_PCARot,TexPCARot)

    end   = time.time() - start
    print ("\nelapsed_time:{0}".format(end) + "[sec]\n")

    t = OnlyTransformICP(CT_Object.getVertices_3D(),Tex_Object.getVertices_3D())

    transformMat = np.array([ [1.0,0.0,0.0,t[0]],
                              [0.0,1.0,0.0,t[1]],
                              [0.0,0.0,1.0,t[2]],
                              [0.0,0.0,0.0,1.0 ]
                            ])

    CT_Object.linerConversion(transformMat)


    try:
        print("file save")
        CT_Object.saveOBJ(Savefilepath)
    except:
        print("file save err\n")
        return False

    paths = Savefilepath.split("\\")
    savef = paths[len(paths)-1]

    dialog = wx.MessageDialog(None, savef + 'を保存しました', 'メッセージ', style=wx.OK)
    dialog.ShowModal()
    return True
