#Otsu
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('noisy2.png',0)
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])


#Adaptativa
import numpy as np
import cv2

gray = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('tutorial umbral', gray)

# umbral fijo
_, dst1 = cv2.threshold(gray, 96, 255, cv2.THRESH_BINARY)

cv2.imshow('umbral fijo', dst1)

# umbral adaptable
gray = cv2.medianBlur(gray, 5)
dst2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('umbral adaptable', dst2)

cv2.waitKey(0)

# void cv::adaptiveThreshold(
#         cv::InputArray src,  // Imagen de entrada
#         cv::OutputArray dst, // Imagen de salida
#         double maxValue,     // Valor que se asigna si se cumple con el umbral
#         int adaptiveMethod,  // Método a utilizar (mean, Gaussian)
#         int thresholdType,   // Tipo de umbralización
#         int blockSize,       // Tamaño de bloque: 3, 5, 7, ...
#         double C             // Constante
# );

gray = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)
gray = cv2.medianBlur(gray, 5)
dst2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


#####binarización de imágenes python + opencv
#Método de binarización de imagen: 1. Umbral global 2. Umbral local
#Método de binarización de imágenes en OpenCV: 1. OTSU 2. Triángulo 3. Automático y manual 4. Umbral adaptativo

import cv2 as cv
import numpy as np
 
 
# Encuentra automáticamente el umbral de acuerdo con el método seleccionado
def threshold_demo(image):
         # Imagen en escala de grises
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
         # Imagen binaria
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print('threshold value %s' % ret)
    cv.imshow('binary', binary)
 
 
 # Umbral local
def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
         # blockSize debe ser un número impar, lo siguiente se establece en 25, mayor que el valor promedio 10 (establecido por usted mismo) se establece en blanco o negro, y dentro de 10 se establece en otro color
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow('binary', dst)
 
 
 # Umbral adaptativo
def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
         # Mean
    mean = m.sum() / (w*h)
    print('mean:', mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow('binary', binary)
 
 
 # Establecer manualmente el umbral
def threshold_demo_1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    print('threshold value %s' % ret)
    cv.imshow('binary', binary)
 
 
src = cv.imread('C:/Users/Y/Pictures/Saved Pictures/demo.png')
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow('input image', src)
custom_threshold(src)
cv.waitKey(0)
cv.destroyAllWindows()

#####Operaciones morfologicas

#Eroción 
import cv2
import numpy as np
 
img = cv2.imread('A.png',0)
kernel = np.ones((7,7),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

#Dilatación
dilatacion = cv2.dilate(img,kernel,iterations = 1)

#Apertura
apertura = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#Cierre
cierre = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#Gradiente Morfológico
gradiente = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


#####Tranformada Hit or Miss
import cv2 as cv
import numpy as np
input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 255, 255, 0],
    [0,255, 0, 255, 0, 0, 255, 0],
    [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")
kernel = np.array((
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]), dtype="int")
output_image = cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel)
rate = 50
kernel = (kernel + 1) * 127
kernel = np.uint8(kernel)
kernel = cv.resize(kernel, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv.imshow("kernel", kernel)
cv.moveWindow("kernel", 0, 0)
input_image = cv.resize(input_image, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv.imshow("Original", input_image)
cv.moveWindow("Original", 0, 200)
output_image = cv.resize(output_image, None , fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv.imshow("Hit or Miss", output_image)
cv.moveWindow("Hit or Miss", 500, 200)
cv.waitKey(0)
cv.destroyAllWindows()

