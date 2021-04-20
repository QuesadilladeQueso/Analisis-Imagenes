
import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt

image = cv2.imread('Imagen3.jpg', 0)

def acumulada(i):
    sum = 0
    for index in range(i):
        sum = sum + probas[index]
    return sum

def EQNYH(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    probas = []
    for i in range(len(hist)):
        probas.append(hist[i,0]/31376)

    matriz_bruta = image.reshape(1,31376)
    matriz = matriz_bruta[0]

    M_matriz = []

    r_max = 255
    r_min = 20
    for i in matriz:     
        d = acumulada(i)  
        operacion = (r_max/r_min) * d
        a = r_min * round(operacion)
        M_matriz.append(a)
    nueva_matriz = np.array([M_matriz]).reshape(148,212)

#cv2.imwrite('filename.png', nueva_matriz)
#print("despues del cambio")
#plt.hist(image.ravel(), 256, [0,256])
#plt.show()
#plt.hist(nueva_matriz.ravel(), 256, [0,256])
#plt.show()

