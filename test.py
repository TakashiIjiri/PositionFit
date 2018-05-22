import numpy as np
from Object_3D import Object_3D

def conversion(obj,matrix):
    obj.linerConversion(a)
    

obj = Object_3D("./test.obj")

print(obj.getVertices())

a = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])

conversion(obj,a)

print(obj.getVertices())