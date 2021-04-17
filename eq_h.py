

from PIL import Image
import cv2
import numpy as np 
from matplotlib import pyplot as plt


#st.title("Abre una imagen :3")

#original = Image.open('Imagen3.jpg')

#st.image(original, use_column_width=True)

image = cv2.imread('Imagen3.jpg',0)
cv2.imshow('Imagen 3', image)


hist = cv2.calcHist([image], [0], None, [256], [0,256])

print(hist)
plt.plot(hist)

largo, alto = image.shape
 
for l in range(largo):
    for a in range(alto):
            #print("antes del cambio", image[l,a])
            image[l,a] = image[l,a] + 100
            #print("despues del cambio", image[l,a])
print('-----------------')

hist2 = cv2.calcHist([image], [0], None, [256], [0,256])

print(hist2)
plt.plot(hist2)

cv2.imshow("window",image)
cv2.waitKey(0)

# plt.plot(hist, color='gray')

# plt.xlabel('intensidad de iluminacion')
# plt.ylabel('cantidad de pixeles')
# plt.show()
matriz = np.array(image)
n_matriz = matriz.reshape(148,212)


# for x in n_matriz:
#     print(x)



# cv2.destroyAllWindows()







