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
from scipy import ndimage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


#---------Selección de Imagen-----------------------------------------------------------------------------------------------------
def Seleccion_Imagen():
    
    global ruta
    
    ruta = filedialog.askopenfilename(filetypes= [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    
    if len(ruta) > 0:
        global image
        
        image = cv2.imread(ruta)
        image = imutils.resize(image, height=380)
        
        muestra = imutils.resize(image, height=180)
        muestra = cv2.cvtColor(muestra, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(muestra)
        img_tk = ImageTk.PhotoImage(image=img)
        labelImagenEntrada.configure(image=img_tk)
        labelImagenEntrada.image = img_tk
        
        
        labelinfo1 = Label(raiz, text="Imagen de entrada", bg="#363b40", fg="#ffffff", width=30, font=("Courier", 26))
        labelinfo1.grid(column=0, row=1, padx=5, pady=5)

        framegrafico = Frame(bg="#ffffff", width="50", height="50")
        framegrafico.grid(column=0, row=3)
        
        eg1 = cv2.cvtColor(muestra, cv2.COLOR_BGR2GRAY)
        h1 = cv2.calcHist([eg1],[0],None,[256],[0,255])
        fig = plt.figure()
        fig.add_subplot(111).plot(h1)
        canvas = FigureCanvasTkAgg(fig, framegrafico)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=0)
        
        labelImagenSalida.image = "" 
        seleccionado.set(0)

#---------Ecuación Hyperbolica-----------------------------------------------------------------------------------------------------
def acumulada(i,probas):
    sum = 0
    for index in range(i):
        sum = sum + probas[index]
    return sum        

def EQNYH():
   
    image = cv2.imread(ruta, 0)    
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    h, w = image.shape
    probas = []
    for i in range(len(hist)):
        probas.append(hist[i,0]/h*w)

   
    matriz_bruta = image.reshape(1,h*w)
    matriz = matriz_bruta[0]

    M_matriz = []

    r_max = max(matriz)
    r_min = min(matriz)+1
    
    print(r_max)
    print(r_min)
    
    for i in matriz:     
        d = acumulada(i, probas)  
        operacion = (r_max/r_min) * d
        a = r_min * round(operacion)
        M_matriz.append(a)
    
    
    nueva_matriz = np.array([M_matriz]).reshape(h,w)
    
    nueva_matriz = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((nueva_matriz).astype(np.uint8))
    img.save("EQ_Hiperbolica.jpg")
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

#---------Estrechamiento/Expansión-------------------------------------------------------------------------------------------------
def EST ():
    
    img = cv2.imread(ruta, 0) #abre la imagen
    hist,bins = np.histogram(img.flatten(),256,[0,256]) #saca el histograma
    cdf = hist.cumsum() #se hace el filtro del histograma
    cdf_normalized = cdf * hist.max()/ cdf.max() #se normaliza
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())     #Aplicación del filtro
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    
    im2 = imutils.resize(img2.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img.save("Estrechamiento.jpg")
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)
    
#---------Brillo/Desplazamiento----------------------------------------------------------------------------------------------------
def Desplazamiento():
    
    image = cv2.imread(ruta, 0) 
    h, w = image.shape
    matriz_bruta = image.reshape(1,h*w)
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
        
    nueva_matriz = np.array([new_image]).reshape(h,w)
    
    im2 = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img.save("Desplazamiento.jpg")
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

# #---------Contracción--------------------------------------------------------------------------------------------------------------

def Contraccion():
    
    image = cv2.imread(ruta, 0) 
    h, w = image.shape
    matriz_bruta = image.reshape(1,h*w)
    matriz = matriz_bruta[0]
    
    new_image = []
     
    alpha = 1.0 # Contraste
    beta = 0    # Brillo 
    # Terminal
    application_window = tk.Tk()
    alpha = simpledialog.askstring("Estrechamiento", "Ingrese cambio [1-10]", parent=application_window)
    alpha = int(alpha)
    
    for pixel in matriz:
        operacion = (alpha*pixel) + beta
        new_image.append(operacion)
        
    nueva_matriz = np.array([new_image]).reshape(h,w)
    
    im2 = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img.save("Contraccion.jpg")
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)
        

def SalPimienta():
    
    image = cv2.imread(ruta, 0) 
    h, w = image.shape
    matriz_bruta = image.reshape(1,h*w)
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
    
    nueva_matriz = np.array([new_image]).reshape(h,w)
    
    im2 = imutils.resize(nueva_matriz.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img.save("Sal_Pimienta.jpg")
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

#---------Promediador-----------------------------------------------------------------------------------------------------        
def Promedio():
   
    im = cv2.imread(ruta, 0) 
        
    a = [ [ 1.0/9, 1.0/9, 1.0/9 ],
          [ 1.0/9, 1.0/9, 1.0/9 ],
          [ 1.0/9, 1.0/9, 1.0/9 ] ]
    kernel = np.asarray(a)
    dst = cv2.filter2D(im, -1, kernel)
    cv2.imwrite ('Promedio.jpg', dst)
   
    im2 = imutils.resize(dst.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)
     

#---------Promediado pesado-----------------------------------------------------------------------------------------------------     
def PromedioPesado():
    
    #val = random.randrange(0, 10)
    im = cv2.imread(ruta, 0) 
    
    #aux = val + 2
    # a = [ [ 1.0/aux, 1.0/aux, 1.0/aux ],
    #       [ 1.0/aux,  val   , 1.0/aux ],
    #       [ 1.0/aux, 1.0/aux, 1.0/aux ] ]
    
    a = [
        [0.05, 0.05, 0.05],
        [0.05, 0.1, 0.05],
        [0.05, 0.05, 0.05]]
    kernel = np.asarray(a)
    dst = cv2.filter2D(im, -1, kernel)
    cv2.imwrite ('PromedioPesado.jpg', dst)
    
    im2 = imutils.resize(dst.astype(np.uint8), height=400)
    
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)


def Marr_Hildreth():

    img = cv2.imread(ruta, 0) 
    
    sigma = 3
    
    size = int(2*(np.ceil(3*sigma))+1)
    
    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
 
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter
        
    kern_size = kernel.shape[0]
    log = np.zeros_like(img, dtype=float)
    
     # applying filter
    for i in range(img.shape[0]-(kern_size-1)):
        for j in range(img.shape[1]-(kern_size-1)):
            window = img[i:i+kern_size, j:j+kern_size] * kernel
            log[i, j] = np.sum(window)


    log = log.astype(np.int64, copy=False)
    zero_crossing = np.zeros_like(log)
         # Calculate 0 cross
    for i in range(log.shape[0]-(kern_size-1)):
        for j in range(log.shape[1]-(kern_size-1)):
            if log[i][j] == 0:
 
                if (log[i][j-1] < 0 and log[i][j+1] > 0) or (log[i][j-1] < 0 and log[i][j+1] < 0) or (log[i-1][j] < 0 and log[i+1][j] > 0) or (log[i-1][j] > 0 and log[i+1][j] < 0):
 
                    zero_crossing[i][j] = 255
 
            if log[i][j] < 0:
 
                if (log[i][j-1] > 0) or (log[i][j+1] > 0) or (log[i-1][j] > 0) or (log[i+1][j] > 0):
 
                    zero_crossing[i][j] = 255
 
    im2 = imutils.resize(zero_crossing.astype(np.uint8), height=400)
    img2 = Image.fromarray((im2).astype(np.uint8))
    img2.save("MarrHildreth.jpg")
    img_tk = ImageTk.PhotoImage(image=img2)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk    

    lblInfo3 = Label(raiz, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)


def Robert():
    
    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                [ 0, 1, 0 ],
                                [ 0, 0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                [ 0, 0, 1 ],
                                [ 0,-1, 0 ]] )
    
    img = cv2.imread(ruta, 0) 
    image2 = np.asarray(img, dtype="int32")
    
    vertical = ndimage.convolve(image2, roberts_cross_v)
    horizontal = ndimage.convolve(image2, roberts_cross_h)
    
    result = np.sqrt( np.square(horizontal) + np.square(vertical) )
    
    im2 = imutils.resize(result.astype(np.uint8), height=400)
    img2 = Image.fromarray((im2).astype(np.uint8))
    img2.save("Robert.jpg")
    img_tk = ImageTk.PhotoImage(image=img2)
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
    raiz.geometry("1180x720")
    raiz['bg'] = '#7090c4' 
    
    labelImagenEntrada = Label(raiz)
    labelImagenEntrada.grid(column = 0, row = 2)
    labelImagenSalida = Label(raiz)
    labelImagenSalida.grid(column = 1, row = 1, rowspan = 6)
    
    labelOpcion = Label(raiz, text="Elige una opción: ", bg="#363b40", fg="#ffffff",width=35, font=("Courier", 25))
    labelOpcion.grid(column = 0, row = 4, padx = 5, pady = 5)
    
    seleccionado = IntVar()
    rad1 = Radiobutton(raiz, text='Desplazamineto', bg="#7090c4", fg="#ffffff", width=35, font=("Courier", 21),value=1, variable=seleccionado, command=Desplazamiento)
    rad2 = Radiobutton(raiz, text='Expansión', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=2, variable=seleccionado, command=EST)
    rad3 = Radiobutton(raiz, text='Contracción', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=3, variable=seleccionado, command=Contraccion)
    rad4 = Radiobutton(raiz, text='Ecualización Logarimo Hiperbolica', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=4, variable=seleccionado, command=EQNYH)
    rad5 = Radiobutton(raiz, text='Ruido SalPimienta', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=5, variable=seleccionado, command=SalPimienta)
    rad6 = Radiobutton(raiz, text='Promediador', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=6, variable=seleccionado, command=Promedio)
    rad7 = Radiobutton(raiz, text='Promedio Pesado', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=7, variable=seleccionado, command=PromedioPesado)
    rad8 = Radiobutton(raiz, text='Marr Hildreth', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=8, variable=seleccionado, command=Marr_Hildreth)
    rad9 = Radiobutton(raiz, text='Robert', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=9, variable=seleccionado, command=Robert)
    rad1.grid(column=0, row=5)
    rad2.grid(column=0, row=6)
    rad3.grid(column=0, row=7)
    rad4.grid(column=0, row=8)
    rad5.grid(column=0, row=9)
    rad6.grid(column=0, row=10)
    rad7.grid(column=0, row=11)
    rad8.grid(column=0, row=12)
    rad9.grid(column=0, row=13)
    
    
    btn = Button(raiz, text="Selecciona una imagen", width=40, command=Seleccion_Imagen)
    btn['bg'] = '#363b40'
    btn['fg'] = '#ffffff'
    btn.grid(column=0, row=0, padx=7, pady=7)
    
    raiz.mainloop()
