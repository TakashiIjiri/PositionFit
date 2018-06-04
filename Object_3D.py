import numpy as np
from objLoader import loadOBJ_4D
from objWriter import saveOBJ
import copy


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2


def calcTriangleArea(pos1,pos2,pos3):
    if len(pos1) == 4:
        pos1 /= pos1[3]
        pos1 = np.array([ pos1[0],pos1[1],pos1[2] ])

    if len(pos2) == 4:
        pos2 /= pos2[3]
        pos2 = np.array([ pos2[0],pos2[1],pos2[2] ])

    if len(pos3) == 4:
        pos3 /= pos3[3]
        pos3 = np.array([ pos3[0],pos3[1],pos3[2] ])


    vector1_2 = pos2 - pos1
    vector1_3 = pos3 - pos1
    cross     = np.cross(vector1_2,vector1_3)

    return np.linalg.norm(cross)/2



def calcCenter(vertices):
    center = np.zeros(4)
    for v in vertices:
        if len(v)==4:
            v /= v[3]
        else:
            v = np.append( v, [1] )

        center += v

    return center/len(vertices)



class MaterialPoint:

    def __init__(self,position,mass):
        self.m_position = position
        self.m_mass     = mass



#頂点は同次座標
class Object_3D():

    def __init__(self,OBJ_FilePath=""):
        try:
            vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = loadOBJ_4D(OBJ_FilePath)
        except:
            print("no input obj")
            self.__m_vertices     = None
            self.__m_uvs          = None
            self.__m_normals      = None
            self.__m_faceIDs      = None
            self.__m_uvIDs        = None
            self.__m_normalIDs    = None
            self.__m_vertexColors = None
            return

        self.__m_vertices     = np.array(vertices)
        self.__m_uvs          = np.array(uvs)
        self.__m_normals      = np.array(normals)
        self.__m_faceIDs      = np.array(faceVertIDs)
        self.__m_uvIDs        = np.array(uvIDs)
        self.__m_normalIDs    = np.array(normalIDs)
        self.__m_vertexColors = np.array(vertexColors)
        self.__m_center       = calcCenter(self.__m_vertices)

        self.defineMaterialPoints()


    def loadOBJ(self,OBJ_FilePath):
        try:
            vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = loadOBJ_4D(OBJ_FilePath)
        except:
            print("OBJ input err")
            return

        self.__m_vertices     = np.array(vertices)
        self.__m_uvs          = np.array(uvs)
        self.__m_normals      = np.array(normals)
        self.__m_faceIDs      = np.array(faceVertIDs)
        self.__m_uvIDs        = np.array(uvIDs)
        self.__m_normalIDs    = np.array(normalIDs)
        self.__m_vertexColors = np.array(vertexColors)
        self.__m_center       = calcCenter(self.__m_vertices)

        self.defineMaterialPoints()


    def saveOBJ(self,OBJ_FilePath):
        try:
            saveOBJ(
                     OBJ_FilePath         ,
                     self.__m_vertices    ,
                     self.__m_uvs         ,
                     self.__m_normals     ,
                     self.__m_faceIDs     ,
                     self.__m_uvIDs       ,
                     self.__m_normalIDs   ,
                     self.__m_vertexColors
                    )
        except:
            print("OBJ save err")
            return


    #各三角形メッシュの重心を位置、面積を質量とした頂点群を定義
    def defineMaterialPoints(self):
        self.__m_material_points        = []
        self.__m_material_points_center = np.zeros(4)
        self.__m_mass                   = 0.0

        for indices in self.__m_faceIDs:
            center = np.zeros(4)
            for index in indices:
                pos_4D  = self.__m_vertices[index]
                pos_4D /= pos_4D[3] #通常は1で割るだけのはず

                center += pos_4D

            center /= len(indices)
            mass    = calcTriangleArea(self.__m_vertices[indices[0]],self.__m_vertices[indices[1]],self.__m_vertices[indices[2]])

            Material_P = MaterialPoint(center,mass)
            self.__m_material_points.append(Material_P)
            self.__m_mass += mass
            self.__m_material_points_center += mass * center

        self.__m_material_points_center   /= self.__m_mass
        self.__m_material_points_center[3] = 1.0



    def linerConversion(self,conversionMatrix):
        try:
            self.__m_vertices = np.dot(conversionMatrix, self.__m_vertices.T).T
            for i in range( len(self.__m_vertices) ):
                self.__m_vertices[i] /= self.__m_vertices[i][3]

            self.__m_center = np.mean( self.__m_vertices, axis = 0 )

            conversionMat_3D = np.array([ [conversionMatrix[0][0],conversionMatrix[0][1],conversionMatrix[0][2]],
                                          [conversionMatrix[1][0],conversionMatrix[1][1],conversionMatrix[1][2]],
                                          [conversionMatrix[2][0],conversionMatrix[2][1],conversionMatrix[2][2]] ])

            if len(self.__m_normals) > 0:
                self.__m_normals = np.dot(conversionMat_3D,self.__m_normals.T).T
                self.__m_normals = normalize(self.__m_normals,axis=1)

        except Exception as e:
            print ('type:' + str(type(e)))
            print ('args:' + str(e.args))
            print ('e自身:' + str(e))
            print("conversion err")



    def getVertices_4D(self):
        return copy.deepcopy(self.__m_vertices)


    def getVertices_3D(self):
        vertices = np.zeros( (len(self.__m_vertices),3) )
        for i in range( len(vertices) ):
            vertices[i][0] = self.__m_vertices[i][0] / self.__m_vertices[i][3]
            vertices[i][1] = self.__m_vertices[i][1] / self.__m_vertices[i][3]
            vertices[i][2] = self.__m_vertices[i][2] / self.__m_vertices[i][3]

        return vertices


    def getPosition_4D(self):
        return self.__m_center

    def getPosition_3D(self):
        center = np.zeros(3)
        center[0] = self.__m_center[0] / self.__m_center[3]
        center[1] = self.__m_center[1] / self.__m_center[3]
        center[2] = self.__m_center[2] / self.__m_center[3]

        return center


    def getMaterialPoints(self):
        return self.__m_material_points


    def getMaterialPointsCenter(self):
        return self.__m_material_points_center


    def getMass(self):
        return self.__m_mass

    def setVertices_3D(self,vertices):
        vertices_4D = np.zeros( (len(vertices), 4) )

        for i in range(len(vertices_4D)):
            vertices_4D[i][0] = vertices[i][0]
            vertices_4D[i][1] = vertices[i][1]
            vertices_4D[i][2] = vertices[i][2]
            vertices_4D[i][3] = 1.0

        self.__m_vertices = vertices_4D

if __name__ =="__main__":
    object3D = Object_3D("C:\\Users\\光\\Documents\\GitHub\\PositionFit\\test.obj")
