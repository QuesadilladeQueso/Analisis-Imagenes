from tkinter import *
from tkinter import simpledialog
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import imutils
import numpy as np
import tkinter as tk
import random
from matplotlib import pyplot as plt


#---------Selección de Imagen-----------------------------------------------------------------------------------------------------
def Seleccion_Imagen():
    
    
    ruta = filedialog.askopenfilename(filetypes= [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    
    if len(ruta) > 0:
        global image
        
        image = cv2.imread(ruta)
        image = imutils.resize(image, height=380)
        
        muestra = imutils.resize(image, height=280)
        muestra = cv2.cvtColor(muestra, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(muestra)
        img_tk = ImageTk.PhotoImage(image=img)
        labelImagenEntrada.configure(image=img_tk)
        labelImagenEntrada.image = img_tk
        
        labelinfo1 = Label(raiz, text="Imagen de entrada", bg="#363b40", fg="#ffffff", width=30, font=("Courier", 26))
        labelinfo1.grid(column=0, row=1, padx=5, pady=5)

        labelImagenSalida.image = ""
        seleccionado.set(0)

#---------Ecuación Hyperbolica-----------------------------------------------------------------------------------------------------
def acumulada(i,probas):
    sum = 0
    for index in range(i):
        sum = sum + probas[index]
    return sum        

def EQNYH():
    
    ruta = filedialog.askopenfilename(filetypes= [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
   
    image = cv2.imread(ruta, 0)

    
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
        d = acumulada(i, probas)  
        operacion = (r_max/r_min) * d
        a = r_min * round(operacion)
        M_matriz.append(a)
    
    
    nueva_matriz = np.array([M_matriz]).reshape(148,212)
    
    nueva_matriz = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((nueva_matriz).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

#---------Estrechamiento/Expansión-------------------------------------------------------------------------------------------------
def EST ():
    
    ruta = filedialog.askopenfilename(filetypes= [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    
    img = cv2.imread(ruta, 0) #abre la imagen
    hist,bins = np.histogram(img.flatten(),256,[0,256]) #saca el histograma
    cdf = hist.cumsum() #se hace el filtro del histograma
    cdf_normalized = cdf * hist.max()/ cdf.max() #se normaliza
    
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')                #Para mostrar el primer histograma
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())     #Aplicación del filtro
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    
    im2 = imutils.resize(img2.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

    
    # img = cv2.imread('imagen3.jpg',0)                               #Lectura y creación de la nueva imagen e histograma con el filtro
    # equ = cv2.equalizeHist(img)
    # res = np.hstack((img,equ)) #stacking images side-by-side
    # cv2.imwrite('res.png',res)                                      #Guardado de la nueva imagen con la comparativa entre imagenes
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(res.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
    
#---------Brillo/Desplazamiento----------------------------------------------------------------------------------------------------
def Desplazamiento():
    
    ruta = filedialog.askopenfilename(filetypes= [
            ("image", ".jpeg"),
            ("image", ".png"),
            ("image", ".jpg")])

    image = cv2.imread(ruta, 0) 
    matriz_bruta = image.reshape(1,31376)
    matriz = matriz_bruta[0]
    
    new_image = []
     
    alpha = 1.0 # Contraste
    beta = 0    # Brillo 
    # Terminal
    application_window = tk.Tk()
    beta = simpledialog.askstring("Dezplazamiento", "Ingrese cambio del brillo [0-100]", parent=application_window)
    beta = int(beta)
    
    for pixel in matriz:
        operacion = (alpha*pixel) + beta
        new_image.append(operacion)
        
    nueva_matriz = np.array([new_image]).reshape(148,212)
    
    #Terminal
    #for y in range(image.shape[0]):
     #   for x in range(image.shape[1]):
      #      for c in range(image.shape[2]):
       #         new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    im2 = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

    #Muestra de imagenes
    ##cv2.imshow('Original Image', image)
    ##img = cv2.imread('Imagen3.jpg',0)
    ##plt.hist(img.ravel(),256,[0,256]); plt.show()
    #Primer imagen con histograma fin
    ##cv2.waitKey()
    ##cv2.imshow('New Image', new_image)
    ##plt.hist(new_image.ravel(),256,[0,256]); plt.show()
    # Wait until user press some key
    ##cv2.waitKey()

# #---------Contracción--------------------------------------------------------------------------------------------------------------

def Contraccion():
    
    ruta = filedialog.askopenfilename(filetypes= [
            ("image", ".jpeg"),
            ("image", ".png"),
            ("image", ".jpg")])

    image = cv2.imread(ruta, 0) 
    matriz_bruta = image.reshape(1,31376)
    matriz = matriz_bruta[0]
    
    new_image = []
     
    alpha = 1.0 # Contraste
    beta = 0    # Brillo 
    # Terminal
    application_window = tk.Tk()
    alpha = simpledialog.askstring("Estrechamiento", "Ingrese cambio [0-100]", parent=application_window)
    alpha = int(alpha)
    
    for pixel in matriz:
        operacion = (alpha*pixel) + beta
        new_image.append(operacion)
        
    nueva_matriz = np.array([new_image]).reshape(148,212)
    
    im2 = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)
        

def SalPimienta():
    
    ruta = filedialog.askopenfilename(filetypes= [
            ("image", ".jpeg"),
            ("image", ".png"),
            ("image", ".jpg")])

    image = cv2.imread(ruta, 0) 
    matriz_bruta = image.reshape(1,31376)
    matriz = matriz_bruta[0]
    
    new_image = []
    
    application_window = tk.Tk()
    prob = float(simpledialog.askstring("Ruido SalPimienta", "Ingrese cambio [0-0.1]", parent=application_window))
    thresh = 1 - prob
    
    for pixel in matriz:
        rdn = random.random()
        if rdn < prob:
            new_image.append(0)
        elif rdn > thresh:
             new_image.append(255)
        else:
            new_image.append(pixel)
    
    nueva_matriz = np.array([new_image]).reshape(148,212)
    
    im2 = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)


#---------Main-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    image = None
    ruta = None
    
    raiz = Tk()
    

    raiz.title("Practica 1: Ajuste de Brillo")
    raiz.geometry("1000x720")
    raiz['bg'] = '#7090c4' 
    
    labelImagenEntrada = Label(raiz)
    labelImagenEntrada.grid(column = 0, row = 2)
    labelImagenSalida = Label(raiz)
    labelImagenSalida.grid(column = 1, row = 1, rowspan = 6)
    
    labelOpcion = Label(raiz, text="Elige una opción: ", bg="#363b40", fg="#ffffff",width=35, font=("Courier", 25))
    labelOpcion.grid(column = 0, row = 3, padx = 5, pady = 5)
    
    seleccionado = IntVar()
    rad1 = Radiobutton(raiz, text='Desplazamineto', bg="#7090c4", fg="#ffffff", width=35, font=("Courier", 21),value=1, variable=seleccionado, command=Desplazamiento)
    rad2 = Radiobutton(raiz, text='Expansión', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=2, variable=seleccionado, command=EST)
    rad3 = Radiobutton(raiz, text='Contracción', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=3, variable=seleccionado, command=Contraccion)
    rad4 = Radiobutton(raiz, text='Ecualización Logarimo Hiperbolica', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=4, variable=seleccionado, command=EQNYH)
    rad5 = Radiobutton(raiz, text='Ruido SalPimienta', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=5, variable=seleccionado, command=SalPimienta)
    rad1.grid(column=0, row=4, padx = 10, pady = 10)
    rad2.grid(column=0, row=5, padx = 10, pady = 10)
    rad3.grid(column=0, row=6, padx = 10, pady = 10)
    rad4.grid(column=0, row=7, padx = 10, pady = 10)
    rad5.grid(column=0, row=8, padx = 10, pady = 10)
    
    
    btn = Button(raiz, text="Selecciona una imagen", width=40, command=Seleccion_Imagen)
    btn['bg'] = '#363b40'
    btn['fg'] = '#ffffff'
    btn.grid(column=0, row=0, padx=5, pady=5)
    
    raiz.mainloop()
