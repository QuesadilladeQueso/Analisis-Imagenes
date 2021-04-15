

import streamlit as st
from PIL import Image
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd


st.title("Abre una imagen :3")

original = Image.open('Imagen3.jpg')

st.image(original, use_column_width=True)

image = cv2.imread('Imagen3.jpg',0)
histograma =  plt.hist(image.ravel(), 256,[0,256])
plt.show()


segunda = cv2.equalizeHist(original)
st.image(segunda, use_column_width=True)

histograma2 =  plt.hist(segunda.ravel(), 256,[0,256])
plt.show()

















