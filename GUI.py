from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import imutils
import numpy as np


image = None

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
        

if __name__ == '__main__':
    
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
    rad1 = Radiobutton(raiz, text='Desplazamineto', bg="#7090c4", fg="#ffffff", width=35, font=("Courier", 21),value=1, variable=seleccionado)
    rad2 = Radiobutton(raiz, text='Expansión', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=2, variable=seleccionado)
    rad3 = Radiobutton(raiz, text='Contracción', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=3, variable=seleccionado)
    rad4 = Radiobutton(raiz, text='Ecualización Logarimo Hiperbolica', bg="#7090c4",fg="#ffffff", width=35, font=("Courier", 21), value=4, variable=seleccionado)
    rad1.grid(column=0, row=4, padx = 10, pady = 10)
    rad2.grid(column=0, row=5, padx = 10, pady = 10)
    rad3.grid(column=0, row=6, padx = 10, pady = 10)
    rad4.grid(column=0, row=7, padx = 10, pady = 10)
    
    
    btn = Button(raiz, text="Selecciona una imagen", width=40, command=Seleccion_Imagen)
    btn['bg'] = '#363b40'
    btn['fg'] = '#ffffff'
    btn.grid(column=0, row=0, padx=5, pady=5)
    
    raiz.mainloop()