import numpy as np
from objLoader import loadOBJ_4D
from objWriter import saveOBJ

def calcTriangleArea(pos1,pos2,pos3):
    if len(pos1) == 4:
        pos1 /= pos1[3]

    if len(pos2) == 4:
        pos2 /= pos2[3]

    if len(pos3) == 4:
        pos3 /= pos3[3]


    vector1_2 = pos2 - pos1
    vector1_3 = pos3 - pos1

    cross = np.cross(vector1_2,vector1_3)
    #デバッグ
    print(cross)

    return np.linalg.norm(cross)/2


class MaterialPoint:

    def __init__(self,position,mass):
        self.m_position = position
        self.m_mass     = mass


def calcCenter(vertices):
    center = np.zeros(4)
    for v in vertices:
        if len(v)==4:
            v /= v[3]
        else:
            v = np.append( v, [1] )

        center += v

    return center/len(vertices)


#頂点は同次座標
class Object_3D():

    def __init__(self,OBJ_FilePath=""):
        self.__m_material_point = []
        try:
            vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = loadOBJ_4D(OBJ_FilePath)
        except:
            self.__m_vertices     = None
            self.__m_uvs          = None
            self.__m_normals      = None
            self.__m_faceIDs      = None
            self.__m_uvIDs        = None
            self.__m_normalIDs    = None
            self.__m_vertexColors = None
            return

        self.__m_vertices     = vertices
        self.__m_uvs          = uvs
        self.__m_normals      = normals
        self.__m_faceIDs      = faceVertIDs
        self.__m_uvIDs        = uvIDs
        self.__m_normalIDs    = normalIDs
        self.__m_vertexColors = vertexColors
    
    
    def loadOBJ(self,OBJ_FilePath):
        try:
            vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = loadOBJ_4D(OBJ_FilePath)
        except:
            print("OBJ input err")
            return

        self.__m_vertices     = vertices
        self.__m_uvs          = uvs
        self.__m_normals      = normals
        self.__m_faceIDs      = faceVertIDs
        self.__m_uvIDs        = uvIDs
        self.__m_normalIDs    = normalIDs
        self.__m_vertexColors = vertexColors

    
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


    def linerConversion(self,conversionMatrix):
        try:
            for i in range(len(self.__m_vertices)):
                self.__m_vertices[i] = np.dot( conversionMatrix, self.__m_vertices )
        except:
            print("conversion err")


    def defineMaterialPoints(self):
        for indices in self.__m_faceIDs:
            center = np.zeros(4)
            for index in indices:
                pos_4D  = self.__m_vertices[index]
                pos_4D /= pos_4D[3] #通常は1で割るだけのはず

                center += pos_4D

            center /= len(indices)
            mass    = calcTriangleArea(self.__m_vertices[indices[0]],self.__m_vertices[indices[1]],self.__m_vertices[indices[2]])
            
            Material_P = MaterialPoint(center,mass)
            self.__m_material_point.append(Material_P)

        #デバッグ
        print(self.__m_material_point)





    def getVertices(self):
        return self.__m_vertices


