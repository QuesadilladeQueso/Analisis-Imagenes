

import streamlit as st
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
# plt.plot(hist, color='gray')

# plt.xlabel('intensidad de iluminacion')
# plt.ylabel('cantidad de pixeles')
# plt.show()
matriz = np.array(image)
n_matriz = matriz.reshape(148,212)


# for x in n_matriz:
#     print(x)



# cv2.destroyAllWindows()







