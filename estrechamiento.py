import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('imagen3.jpg',0) #abre la imagen
hist,bins = np.histogram(img.flatten(),256,[0,256]) #saca el histograma
cdf = hist.cumsum() #se hace el filtro del histograma
cdf_normalized = cdf * hist.max()/ cdf.max() #se normaliza
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')                #Para mostrar el primer histograma
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())     #Aplicación del filtro
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
img = cv2.imread('imagen3.jpg',0)                               #Lectura y creación de la nueva imagen e histograma con el filtro
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)                                      #Guardado de la nueva imagen con la comparativa entre imagenes
plt.plot(cdf_normalized, color = 'b')
plt.hist(res.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()