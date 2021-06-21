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



def LimpiarSalida():
    widget = frame_ImagenEntrada.winfo_children()[4]
    widget.destroy()

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
                    
        labelinfo1 = Label(frame_ImagenEntrada, text="Imagen de entrada", bg="#363b40", fg="#ffffff", width=30, font=("Courier", 26))
        labelinfo1.pack(side=TOP)
    
        framegrafico = Frame(frame_ImagenEntrada,bg="#ffffff", width="30", height="30")
        framegrafico.pack()
        
        eg1 = cv2.cvtColor(muestra, cv2.COLOR_BGR2GRAY)
        h1 = cv2.calcHist([eg1],[0],None,[256],[0,255])
        fig = plt.figure(figsize=(3,3), dpi=80)
        
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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()
    
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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()
        

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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()
     

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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()


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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()


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

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()


def Otsu():
    
    img = cv2.imread(ruta, 0) 
    threshold1 = simpledialog.askstring("Thresholding", "Ingresa un valor 1 [0-255]: ")
    threshold1 = int(threshold1)
    threshold2 = simpledialog.askstring("Thresholding", "Ingresa un valor 2 [0-255]: ")
    threshold2 = int(threshold2)
    ret2,th2 = cv2.threshold(img,threshold1,threshold2,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imwrite ('Otsu.jpg', th2)
   
    im2 = imutils.resize(th2.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()
    
    
def Otsu_gauss():
    
    img = cv2.imread(ruta, 0) 
    
    blur = cv2.GaussianBlur(img,(5,5),0)
    threshold1 = simpledialog.askstring("Thresholding", "Ingresa un valor 1 [0-255]: ")
    threshold1 = int(threshold1)
    threshold2 = simpledialog.askstring("Thresholding", "Ingresa un valor 2 [0-255]: ")
    threshold2 = int(threshold2)
    ret3,th3 = cv2.threshold(blur,threshold1,threshold2,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imwrite ('Otsu_gauss.jpg', th3)
   
    im2 = imutils.resize(th3.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()
    

def Adaptativa():
    
    img = cv2.imread(ruta, 0) 
    
    img = cv2.medianBlur(img, 5)
    dst2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    cv2.imwrite ('Adaptativa.jpg', dst2)
   
    im2 = imutils.resize(dst2.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

def Erosion():
    kernel = np.ones((7,7),np.uint8)

    img = cv2.imread(ruta, 0) 
    
    erosion = cv2.erode(img,kernel,iterations = 1)
    
    cv2.imwrite ('Erocion.jpg', erosion)
   
    im2 = imutils.resize(erosion.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

def Dilatacion():
    kernel = np.ones((7,7),np.uint8)
    img = cv2.imread(ruta, 0) 
    
    dilatacion = cv2.dilate(img,kernel,iterations = 1)
    
    cv2.imwrite ('Dilatacion.jpg', dilatacion)
   
    im2 = imutils.resize(dilatacion.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

def Gradiente_Morfologico():
    kernel = np.ones((7,7),np.uint8)
    img = cv2.imread(ruta, 0) 
    
    gradiente = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    cv2.imwrite('Gradiente.jpg', gradiente)
   
    im2 = imutils.resize(gradiente.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()

def HitMiss():
    
    img = cv2.imread(ruta, 0)
    
    kernel1 = cv2.imread('fa.bmp',0)
    print(kernel1)
    ret2,kernel_bin = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(kernel_bin)
    
    kernel = np.array((
        [0,0,0,1,0,0,0],
        [0,0,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,0,0],
        [0,0,1,1,1,0,0],
        [0,0,1,1,1,0,0]
    ),dtype='int')
    
    output_image = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
    
    cv2.imwrite('Hit-or-Miss.jpg', output_image)
   
    im2 = imutils.resize(output_image.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()


def Op_Or():
    
    text_img1 = simpledialog.askstring("Operación resta", "Ingresa el nombre de la imagen 1")
    text_img2 = simpledialog.askstring("Operación resta", "Ingresa el nombre de la imagen 2")
    
    img1_original = cv2.imread(text_img1,0)   
    img2 = cv2.imread(text_img2,0)

    bitwise_not = cv2.bitwise_not(img2)    
    black = cv2.bitwise_or(img1_original, bitwise_not)
    
    cv2.imwrite('Imagen_Or.jpg', black)
   
    im2 = imutils.resize(black.astype(np.uint8), height=400)
   
    img = Image.fromarray((im2).astype(np.uint8))
    img_tk = ImageTk.PhotoImage(image=img)
    labelImagenSalida.configure(image=img_tk)
    labelImagenSalida.image = img_tk

    lblInfo3 = Label(frame_ImagenSalida, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.pack()


#---------Main-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    image = None
    ruta = None
    
    raiz = Tk()
    
    raiz.title("Practica 3 y 4")
    raiz.resizable(2,2)
    raiz['bg'] = '#7090c4' 
    
    frame_ImagenEntrada = Frame(raiz)
    frame_ImagenEntrada.config( bg='#7090c4')
    frame_ImagenEntrada.grid(column = 0, row = 0)
    
    frame_Opciones = Frame(raiz)
    frame_Opciones.config(bg='#7090c4')
    frame_Opciones.grid(column = 1, row = 0)
    
    frame_ImagenSalida = Frame(raiz)
    frame_ImagenSalida.config(bg='#7090c4')
    frame_ImagenSalida.grid(column = 0, row=1, columnspan=2)
    
    
    
    labelImagenEntrada = Label(frame_ImagenEntrada)
    labelImagenEntrada.pack(side=TOP)
    labelImagenSalida = Label(frame_ImagenSalida)
    labelImagenSalida.pack()
    
    labelOpcion = Label(frame_Opciones, text="Elige una opción: ", bg="#363b40", fg="#ffffff",width=35, font=("Courier", 25))
    labelOpcion.pack()
    
    seleccionado = IntVar()
    rad1 = Radiobutton(frame_Opciones, text='Otsu', bg="#7090c4", fg="#ffffff", width=35, font=("Courier", 21),value=1, variable=seleccionado, command=Otsu)
    rad2 = Radiobutton(frame_Opciones, text='Otsu después de FGauss', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=2, variable=seleccionado, command=Otsu_gauss)
    rad3 = Radiobutton(frame_Opciones, text='umbralización Adapt', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=3, variable=seleccionado, command=Adaptativa)
    rad4 = Radiobutton(frame_Opciones, text='Erosion', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=4, variable=seleccionado, command=Erosion)
    rad5 = Radiobutton(frame_Opciones, text='Dilatacion', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=5, variable=seleccionado, command=Dilatacion)
    rad6 = Radiobutton(frame_Opciones, text='Gradiente_Morfologico', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=6, variable=seleccionado, command=Gradiente_Morfologico)
    rad7 = Radiobutton(frame_Opciones, text='Hit or Miss', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=7, variable=seleccionado, command=HitMiss)
    rad8 = Radiobutton(frame_Opciones, text='Expansión', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=8, variable=seleccionado, command=EST)
    rad9 = Radiobutton(frame_Opciones, text='Operación Or', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=9, variable=seleccionado, command=Op_Or)
    # rad7 = Radiobutton(frame_Opciones, text='Promedio Pesado', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=7, variable=seleccionado, command=PromedioPesado)
    # rad8 = Radiobutton(frame_Opciones, text='Marr Hildreth', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=8, variable=seleccionado, command=Marr_Hildreth)
    # rad9 = Radiobutton(frame_Opciones, text='Robert', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=9, variable=seleccionado, command=Robert)
    rad1.pack()
    rad2.pack()
    rad3.pack()
    rad4.pack()
    rad5.pack()
    rad6.pack()
    rad7.pack()
    rad8.pack()
    rad9.pack()
    
    
    btn = Button(frame_ImagenEntrada, text="Selecciona una imagen", width=40, command=Seleccion_Imagen)
    btn['bg'] = '#363b40'
    btn['fg'] = '#ffffff'
    btn.pack(side=BOTTOM)
    
    btn_clean = Button(frame_ImagenEntrada, text="Limpiar entrada", width=40, command=LimpiarSalida)
    btn_clean['bg'] = '#363b40'
    btn_clean['fg'] = '#ffffff'
    btn_clean.pack(side=BOTTOM)
    
    raiz.mainloop()
