import numpy as np

def loadOBJ(fliePath):
    files = fliePath.split("\\")
    if not(files[len(files)-1].endswith(".obj")):
        raise IOError


    numVertices  = 0
    numUVs       = 0
    numNormals   = 0
    numFaces     = 0
    vertices     = []
    uvs          = []
    normals      = []
    vertexColors = []
    faceVertIDs  = []
    uvIDs        = []
    normalIDs    = []

    for line in open(fliePath, "r"):
        vals = line.split(" ")
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            a = np.array([float(vals[1]),float(vals[2]),float(vals[3])])
            vertices.append(a)
            if len(vals) == 7:
                a = np.array([float(vals[4]),float(vals[5]),float(vals[6])])
                vertexColors.append(a)
            numVertices += 1
        if vals[0] == "vt":
            vt = np.array([float(vals[1]),float(vals[2])])
            uvs.append(vt)
            numUVs += 1
        if vals[0] == "vn":
            vn = np.array([float(vals[1]),float(vals[2]),float(vals[3])])
            normals.append(vn)
            numNormals += 1
        if vals[0] == "f":
            fvID = []
            uvID = []
            nvID = []
            for f in vals[1:]:
                w = f.split("/")
                indices = []
                for i in range(len(w)):
                    try:
                        int(w[i])
                        indices.append(i)
                    except ValueError:
                        pass

                count = 0
                if numVertices > 0:
                    fvID.append(int(w[ indices[count] ])-1)
                    count += 1

                if numUVs > 0 and len(w) < count :
                    uvID.append(int(w[ indices[count] ])-1)
                    count += 1

                if numNormals > 0 and len(w) < count:
                    nvID.append(int(w[ indices[count] ])-1)

            faceVertIDs.append(fvID)
            uvIDs.append(uvID)
            normalIDs.append(nvID)
            numFaces += 1
    print (fliePath)
    print ("numVertices: ", numVertices)
    print ("numUVs: ", numUVs)
    print ("numNormals: ", numNormals)
    print ("numFaces: ", numFaces)
    print ("\n")
    return vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors
