

import streamlit as st
from PIL import Image
import cv2
import numpy as np 
from matplotlib import pyplot as plt


st.title("Abre una imagen :3")

original = Image.open('Imagen3.jpg')

st.image(original, use_column_width=True)

image = cv2.imread('Imagen3.jpg',0)
st.map(plt.hist(image.ravel(), 256,[0,256]))














