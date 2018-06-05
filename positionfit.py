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



def meanDistFromOrigin( verts ) :
    sum = 0
    for v in verts :
        sum += math.sqrt( np.dot(v, v) )
    return sum / len(targetVertices)

def meanDistRatio_FromGravityCenter(trgtVerts, srcVerts) :
    return meanDistFromOrigin( trgtVerts - calcCenter(trgtVerts))
         / meanDistFromOrigin( srcVerts  - calcCenter(srcVerts))

"""
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
"""

#
# calc ICP loss between a pair of vertices (srcVerts, trgtVerts)
#
def Loss(srcVerts, trgtVerts):
    LIMIT_POINT_NUM = 2*10**5
    SAMPLE_POINT_NUM = min(LIMIT_POINT_NUM, len( srcVerts ))
    sourceSamplesIndices = random.sample( range( len( srcVerts ) ), SAMPLE_POINT_NUM )

    target_KDTree = ss.KDTree( trgtVerts )
    err = np.mean( target_KDTree.query( srcVerts[sourceSamplesIndices] )[0] )
    return err



#
# Rotate( CT_Object ) = Tex_Objectとなる Rotationを計算
#
def nearestModel(CT_Object, Tex_Object, CT_PCARot, TexPCARot):

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

        err = Loss(CT_Object.getVertices_3D(), Tex_Object.getVertices_3D())


        #debug
        try:
            filename = "C:\\Users\\pikap\\Desktop\\debugOBJ" + str(count) + ".obj"
            #CT_Object.saveOBJ(filename)
            count += 1
        except:
            pass


        print("err = ",err)
        if err < minErr or minErr < 0:
            minErr = err
            nearlestConversionMat = np.copy(Rot)
            print(nearlestConversionMat)
            print("\n")

        CT_Object.linerConversion( np.dot( transMat2,np.dot( Rot.transpose(),transMat1 ) ) )


    CT_Object.linerConversion( np.dot( transMat2,np.dot( nearlestConversionMat,transMat1 ) ) )


    #debug
    try:
        filename = "C:\\Users\\pikap\\Desktop\\debugOBJ" + str(count) + ".obj"
        #CT_Object.saveOBJ(filename)
        count += 1
    except:
        pass






def positionfit(CTfilepath,　Texturefilepath,　Savefilepath,　doManualScaling = False,　manualScaling = 1.0):
    #load files
    try:
        CT_Object  = Object_3D(CTfilepath     )
        Tex_Object = Object_3D(Texturefilepath)
    except :
        print("file input err\n")
        return False

    #0. 両者を原点に移動
    texVerts = Tex_Object.getVertices()
    texCenter= calcCenter(texVerts)
    texVerts-= texCenter

    ctVerts  = CT_Object .getVertices()
    ctCenter = calcCenter(ctVerts)
    ctVerts -= ctCenter

    #1. scalingする
    if ( doManualScaling ):
        scl = manualScaling
    else :
        scl = meanDistRatio_FromGravityCenter(Tex_Object.getVertices(), CT_Object.getVertices())
    ctVerts *= scl

    #2. PCA計算 --> rotation
    texPCA = PCA()
    ctPCA  = PCA()
    texPCA.fit( Tex_Object.getVertices())
    ctPCA.fit ( CT_Object.getVertices())
    texRot = np.array( texPCA.components_) # (v1,v2,v3)^T
    ct_Rot = np.array( ctPCA.components_ ) # (v1,v2,v3)^T

    if np.linalg.det(texRot) < 0:
        print("invert tex rot")
        texRot[2] = (-1 * texRot[2])

    if np.linalg.det(ct_Rot) < 0:
        print("invert ct rot")
        ct_Rot[2] = (-1 * ct_Rot[2])

    ctVerts = np.dot(texRot.T * ct_Rot, ctVerts.T).T

    #3. 180 deg rotation
    I_Mat = np.diag([ 1.0, 1.0, 1.0,1.0])
    Rot_X = np.diag([ 1.0,-1.0,-1.0,1.0])
    Rot_Y = np.diag([-1.0, 1.0,-1.0,1.0])
    Rot_Z = np.diag([-1.0,-1.0, 1.0,1.0])

    nearlestConversionMat = None
    minErr = -1.0
    tmptmpCtVerts = np.zeros(0)

    for Rot in [I_Mat,Rot_X,Rot_Y,Rot_Z]:
        tmpCtVerts = np.dot( Rot, cvVerts.deepcopy().T).T
        err = Loss(tmpCtVerts, texVerts)

        print("err = ",err)
        if err < minErr or minErr < 0:
            minErr = err
            tmptmpCtVerts = tmpCtVerts.deepcopy()

    ctVerts = tmptmpCtVerts

    #icp
    icp = ICP( texVerts, ctVerts)
    new_vertices = icp.calculate(30)
    new_vertices += texCenter
    CT_Object.setVertices(new_vertices)

    #todo normal

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
