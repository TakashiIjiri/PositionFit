def saveOBJ(filePath, vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors):
    files = filePath.split("\\")
    if not(files[len(files)-1].endswith(".obj")):
        filePath += ".obj"

    f_out = open(filePath, 'w')
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" %(len(vertices)))
    f_out.write("# Faces: %s\n" %(len( faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")
    for vi, v in enumerate( vertices ):
        vStr = "v %s %s %s"  %(v[0], v[1], v[2])
        if len( vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s" %(color[0], color[1], color[2])
        vStr += "\n"
        f_out.write(vStr)
    f_out.write("# %s vertices\n\n"  %(len(vertices)))
    for uv in uvs:
        uvStr =  "vt %s %s\n"  %(uv[0], uv[1])
        f_out.write(uvStr)
    f_out.write("# %s uvs\n\n"  %(len(uvs)))
    for n in normals:
        nStr =  "vn %s %s %s\n"  %(n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n"  %(len(normals)))
    for fi, fvID in enumerate( faceVertIDs ):
        fStr = "f"
        for fvi, fvIDi in enumerate( fvID ):
            fStr += " %s" %( fvIDi + 1 )
            if len(uvIDs) > 0 and len(uvIDs) < fi and len(uvIds[fi]) < fvi:
                fStr += "/%s" %( uvIDs[fi][fvi] + 1 )
            if len(normalIDs) > 0 and len(normalIDs) < fi and len(normalIds[fi]) < fvi:
                fStr += "/%s" %( normalIDs[fi][fvi] + 1 )
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n"  %( len( faceVertIDs)) )
    f_out.write("# End of File\n")
    f_out.close()
