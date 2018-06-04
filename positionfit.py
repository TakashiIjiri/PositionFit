# -*- coding: utf-8 -*-
from ICP import ICP
import numpy as np
import math
import wx
from sklearn.decomposition import PCA
import time
import scipy.spatial as ss
import random
from Object_3D import Object_3D


def varRatio(targetVertices,sourceVertices) :
    targetMean = np.mean( targetVertices, axis=0 )
    targetVar  = 0.0
    targetGravityCoordinateVs = targetVertices - targetMean
    for vertex in targetGravityCoordinateVs:
        targetVar += np.dot(vertex, vertex)
    targetVar /= len(targetVertices)


    sourceMean = np.mean( sourceVertices, axis=0 )
    sourceVar  = 0.0
    sourceGravityCoordinateVs = sourceVertices - sourceMean
    for vertex in sourceGravityCoordinateVs:
        sourceVar += np.dot(vertex,vertex)
    sourceVar /= len(sourceVertices)


    return math.sqrt(targetVar/sourceVar)


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

    target_KDTree = ss.KDTree(Tex_Object.getVertices_3D())

    LIMIT_POINT_NUM = 2*10**5

    if len( CT_vertices ) >LIMIT_POINT_NUM:
        SAMPLE_POINT_NUM = LIMIT_POINT_NUM
    else:
        SAMPLE_POINT_NUM = len( CT_vertices )

    sourceSamplesIndices = random.sample( range( len( CT_vertices ) ),SAMPLE_POINT_NUM )

    err = 0.0
    err = np.mean( target_KDTree.query(CT_vertices[sourceSamplesIndices])[0] )

    return err


def nearlestModel(CT_Object,Tex_Object,CT_PCARot,TexPCARot):

    CT_Center = CT_Object.getPosition_3D()
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
    count = 0
    for Rot in [I_Mat,Rot_X,Rot_Y,Rot_Z]:
        Rot = np.dot(TexPCARot,Rot)

        CT_Object.linerConversion( np.dot( transMat2,np.dot( Rot,transMat1 ) ) )

        err = Loss(CT_Object,Tex_Object)

        #debug
        try:
            filename = "C:\\Users\\光\\Desktop\\debugOBJ" + str(count) + ".obj"
            CT_Object.saveOBJ(filename)
            count += 1
        except:
            pass

        print("err = ",err)
        if err < minErr or minErr < 0:
            minErr = err
            nearlestConversionMat =  Rot
            print(nearlestConversionMat)

        CT_Object.linerConversion( np.dot( transMat2,np.dot( Rot.transpose(),transMat1 ) ) )

    #debug
    try:
        filename = "C:\\Users\\光\\Desktop\\debugOBJ" + str(count) + ".obj"
        CT_Object.saveOBJ(filename)
        count += 1
    except:
        pass


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
        var = varRatio(Tex_Object.getVertices_3D(), CT_Object.getVertices_3D())
    print("\n" + str(var))

    scaleMat  = np.array([ [var,0.0,0.0,0.0],
                           [0.0,var,0.0,0.0],
                           [0.0,0.0,var,0.0],
                           [0.0,0.0,0.0,1.0] ])

    CT_Center  = CT_Object .getPosition_3D()
    Tex_Center = Tex_Object.getPosition_3D()

    transMat1 = np.array([ [1.0,0.0,0.0,-CT_Center[0]],
                           [0.0,1.0,0.0,-CT_Center[1]],
                           [0.0,0.0,1.0,-CT_Center[2]],
                           [0.0,0.0,0.0,1.0          ] ])

    transMat2 = np.array([ [1.0,0.0,0.0,Tex_Center[0]],
                           [0.0,1.0,0.0,Tex_Center[1]],
                           [0.0,0.0,1.0,Tex_Center[2]],
                           [0.0,0.0,0.0,1.0          ] ])
    #拡大縮小だけ最初にやっておく
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


    #debug
    if np.linalg.det(TexPCARot) < 0:
        print("tex debug")
        TexPCARot[:,0] = (-1 * TexPCARot[:,0]).T

    if np.linalg.det(CT_PCARot) < 0:
        print("CT debug")
        CT_PCARot[0] = -1 * CT_PCARot[0]


    start = time.time()

    nearlestModel(CT_Object,Tex_Object,CT_PCARot,TexPCARot)

    end   = time.time() - start
    print ("\nelapsed_time:{0}".format(end) + "[sec]\n")


    icp = ICP(Tex_Object.getVertices_3D(), CT_Object.getVertices_3D())
    new_vertices = icp.calculate(30)
    CT_Object.setVertices_3D(new_vertices)


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
