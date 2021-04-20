
import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt

def Proba(pixel, hist):
    return hist[pixel, 0]/31376

image = cv2.imread('Imagen3.jpg', 0)
# cv2.imshow('Imagen 3', image)

hist = cv2.calcHist([image], [0], None, [256], [0, 256])

hi = hist.cumsum()
hi_norm = hi*255/hi[-1]
matriz_bruta = image.reshape(1,31376)
matriz = matriz_bruta[0]

M_matriz = []
# M_matriz = matriz.reshape(1,31376)
r_max = np.amax(matriz)
r_min = np.amin(matriz)


for i in matriz:     
    # d = acumulada(aux)
    operacion = (r_max/r_min)*hi_norm[i]
    a = i - round(operacion)
    #print(a)
    M_matriz.append(a)
    


nueva_matriz = np.array([M_matriz]).reshape(148,212)

cv2.imwrite('filename.png', nueva_matriz)

print("despues del cambio")


plt.hist(image.ravel(), 256, [0,256])
plt.show()

plt.hist(nueva_matriz.ravel(), 256, [0,256])
plt.show()

