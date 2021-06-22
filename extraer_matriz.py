import pandas as pd
import numpy as np
 
mat = pd.read_csv("matriz1.csv")

a = []
for i in mat:
    for j in i:
        a.append(j)
b = []
b.append(a)

c = []
for p in mat.index:
    for pix in mat.iloc[p]:
        for x in pix:
            c.append(x)
        b.append(c)
        c = []

b = np.array(b, int)    

