import numpy as np



a = np.array([[1,1,1],[2,2,2],[3,3,3]])

b = np.diag( [1,1,1,1] )

print(a+b)