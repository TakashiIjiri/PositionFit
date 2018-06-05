import numpy as np
from objLoader import loadOBJ_4D
from objWriter import saveOBJ
import copy

#normalize all vertices
def normalizeVertices( verts):
    l2 = np.linalg.norm(verts, ord = 2, axis=1, keepdims=True)
    l2[ l2==0 ] = 1
    return v / l2


def calcTriangleArea(pos1, pos2, pos3):
    cross = np.cross(pos2 - pos1, pos3 - pos1)
    return np.linalg.norm(cross) / 2


def calcCenter(vertices):
    center = np.zeros(3)
    for v in vertices:
        center += v
    return center/len(vertices)


#頂点は同次座標
class Object_3D():

    def __init__(self, OBJ_FilePath="") :
        self.loadOBJ(OBJ_FilePath)

    def loadOBJ(self, OBJ_FilePath):
        try:
            vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = loadOBJ(OBJ_FilePath)
        except:
            print("OBJ input err")
            return False

        self.__m_vertices     = np.array(vertices)
        self.__m_uvs          = np.array(uvs)
        self.__m_normals      = np.array(normals)
        self.__m_vertexColors = np.array(vertexColors)
        self.__m_faceIDs      = np.array(faceVertIDs)
        self.__m_uvIDs        = np.array(uvIDs)
        self.__m_normalIDs    = np.array(normalIDs)

        return True

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

    def rotate (self, rot) :
        self.__m_vertices = np.dot( rot, self.__m_vertices.T).T
        self.__m_normals  = np.dot( rot, self.__m_normals.T ).T

    def translate(self, trans ) :
        self.__m_vertices += trans

    def getGravityCenter(self) :
        return calcCenter(self.__m_vertices)

    def getVertices(self) :
        return copy.deepcopy(self.__m_vertices)

    def setVertices(self, verts) :
        self.__m_vertices = copy.deepcopy(verts)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!ここでnormalの再計算!!!")


if __name__ =="__main__":
    object3D = Object_3D("C:\\Users\\光\\Documents\\GitHub\\PositionFit\\test.obj")
