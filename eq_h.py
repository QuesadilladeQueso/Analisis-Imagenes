
import cv2
import numpy as np 
import matplotlib.cm as cm
from matplotlib import pyplot as plt


#st.title("Abre una imagen :3")

#original = Image.open('Imagen3.jpg')

#st.image(original, use_column_width=True)

image = cv2.imread('Imagen3.jpg',0)
# cv2.imshow('Imagen 3', image)

hist = cv2.calcHist([image], [0], None, [256], [0,256])

def Proba(pixel, hist):
    return hist[i]/31376
# print(hist)
matriz = np.array(image)
M_matriz = []
# M_matriz = matriz.reshape(1,31376)
c = 0 
d = 0


for i in matriz:
    d = Proba(i, hist)
    c = c + d
    a = i - c
    M_matriz.append(a)
    

           

             

plt.imsave('filename.png', np.array(M_matriz).reshape(212,148), cmap=cm.gray)
print("despues del cambio")
Image. open('filename.png')
# # print('-----------------')

# hist2 = cv2.calcHist([image], [0], None, [256], [0,256])

# print(hist2)
# plt.plot(hist2)

# cv2.imshow("window",image)
# cv2.waitKey(0)

# plt.plot(hist, color='gray')

# plt.xlabel('intensidad de iluminacion')
# plt.ylabel('cantidad de pixeles')
# plt.show()



matriz = np.array(image)
n_matriz = matriz.reshape(1,31376)


#for pixel in n_matriz:
 #   print("pixel: ",pixel)
  #  print("probabilidad: ", Proba(pixel, hist))

# for x in n_matriz:
#     print(x


    
    
    

# cv2.destroyAllWindows()







