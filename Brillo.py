from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
# Ajuste de brillo y contraste
#Beta corresponde a 
#Lectura de Imagen

# parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
# parser.add_argument('--input', help='Direccion de la imagen.', default='Imagen3.jpg')
# args = parser.parse_args()

print('''
      -------------------------------------------------------------
        Código para cambiar el contraste y el brilo de una imagen
      -------------------------------------------------------------
      ''')

filename = input('Ingresa el nombre de la imagen')

image = cv.imread(filename)
if image is None:
    print('Error en la imagen ', filename)
    exit(0)
#Lectura de Imagen
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0 # Contraste
beta = 0    # Brillo 

# Terminal
print(' Basic Linear Transforms ')
print('-------------------------')
try:
    alpha = float(input('* Ingrese ganancia (contraste) [1.0-3.0]: '))
    beta = int(input('* Ingrese cambio del brillo [0-100]: '))
except ValueError:
    print('Ingrese numero')
#Terminal
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
#Muestra de imagenes
cv.imshow('Original Image', image)
img = cv.imread('Imagen3.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()
#Primer imagen con histograma fin
cv.waitKey()
cv.imshow('New Image', new_image)
plt.hist(new_image.ravel(),256,[0,256]); plt.show()
# Wait until user press some key
cv.waitKey()