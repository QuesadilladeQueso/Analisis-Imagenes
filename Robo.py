import random
import cv2
import time 
import numpy as np
import collections
from collections import Counter
#import sys, os, random, Image
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

def FiltroGaussiano(nombre, val):
    tiempo_de_inicio = time.time()
    img = Image.open(nombre)
    img.show()
    Histograma(img)
    img2 = img.filter(ImageFilter.GaussianBlur(val))
    img2.show()
    Histograma(img2)
    size = (img2.size[0],img2.size[1])
    nueva = Image.new('RGB',size,"white")
    nueva.paste(img2,(0,0))
    nueva.save("RuidoGaussiano.jpg")
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def SalPimienta(nombre, prob):
    tiempo_de_inicio = time.time()
    img = Image.open(nombre)
    img.show()
    Histograma(img)
    imagen7 = img
    output = np.zeros(img.size,np.uint8)
    thres = 1 - prob 
    i=0
    while i < imagen7.size[0]:
        j=0
        while j < imagen7.size[1]:
            rdn = random.random()
            r, g, b = imagen7.getpixel((i,j))
            if rdn < prob:
                r = 0
                g = 0
                b = 0
                imagen7.putpixel((i,j),(r,g,b))
            elif rdn > thres:
                r = 255
                g = 255
                b = 255
                imagen7.putpixel((i,j),(r,g,b))
            else:
                imagen7.putpixel((i,j),(r,g,b))
            j+=1
        i+=1
    imagen7.show()
    Histograma(imagen7)
    size = (imagen7.size[0],imagen7.size[1])
    nueva = Image.new('RGB',size,"white")
    nueva.paste(imagen7,(0,0))
    nueva.save("RuidoSalPimienta.jpg")
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def Promedio(nombre):
    tiempo_de_inicio = time.time()
    im = cv2.imread(nombre)
    cv2.imshow('Original', im)
    a = [ [ 1.0/9, 1.0/9, 1.0/9 ],
          [ 1.0/9, 1.0/9, 1.0/9 ],
          [ 1.0/9, 1.0/9, 1.0/9 ] ]
    kernel = np.asarray(a)
    dst = cv2.filter2D(im, -1, kernel)
    cv2.imshow('Convolucion', dst)
    cv2.waitKey(0)
    cv2.imwrite ('Promedio.jpg', dst)
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def PromedioPesado(nombre, val):
    tiempo_de_inicio = time.time()
    im = cv2.imread(nombre)
    cv2.imshow('Original', im)
    aux = val + 8
    a = [ [ 1.0/aux, 1.0/aux, 1.0/aux ],
          [ 1.0/aux,  val   , 1.0/aux ],
          [ 1.0/aux, 1.0/aux, 1.0/aux ] ]
    kernel = np.asarray(a)
    dst = cv2.filter2D(im, -1, kernel)
    cv2.imshow('Convolucion', dst)
    cv2.waitKey(0)
    cv2.imwrite ('PromedioPesado.jpg', dst)
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def Mediana(nombre, val):
    tiempo_de_inicio = time.time()
    img = cv2.imread(nombre, 0)
    cv2.imshow('Original', img)
    img2 = cv2.medianBlur(img, val)
    cv2.imshow("Mediana", img2)
    cv2.waitKey(0)
    cv2.imwrite ('Mediana.jpg', img2)
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

#def EscalaDeGrises(nombre):
#    imagenGris = Image.new('RGB', nombre.size)
#    datosImg = Image.Image.getdata(nombre)
#    imagenGris.putdata(datosImg)
#    ancho, alto = nombre.size
#
#    for i in range(ancho):
#        for j in range(alto):
#            r, g, b = imagenGris.getpixel((i,j))
#            x = (r + g + b) / 3
#            intx = int (x)
#            pixel = tuple ([intx, intx, intx])
#            imagenGris.putpixel((i,j), pixel)
#    return imagenGris

def ObtenerVecinos(copia, i, j):
    pixel_list = []
    try: 
        pixel_list.append(copia.getpixel((i-1, j-1)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i, j-1)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i+1, j-1)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i-1, j)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i, j)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i+1, j)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i-1, j+1)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i, j+1)))
    except: 
        pixel_list.append((0, 0, 0))
    try: 
        pixel_list.append(copia.getpixel((i+1, j+1)))
    except: 
        pixel_list.append((0, 0, 0))
    return pixel_list

def Moda(nombre):
    tiempo_de_inicio = time.time()
    img = Image.open(nombre)
    img.show()
    Histograma(img)

    copia = Image.new('RGB', img.size)
    datosImg = Image.Image.getdata(img)
    copia.putdata(datosImg)
    ancho, alto = img.size

    for i in range(ancho):
        for j in range(alto):
            r, g, b = copia.getpixel((i,j))
            x = (r + g + b) / 3
            intx = int (x)
            pixel = tuple ([intx, intx, intx])
            copia.putpixel((i,j), pixel)

    for i in range(ancho):
        for j in range(alto):
            vecindades = ObtenerVecinos(img, i, j)
            moda = Counter(vecindades).most_common()[0][0]
            r, g, b = moda
            pixel = tuple([r, g, b])
            copia.putpixel((i, j), pixel)

    copia.show()
    copia.save("Moda.jpg")
    Histograma(copia)
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def Maximo(nombre):
    tiempo_de_inicio = time.time()
    img = Image.open(nombre)
    img.show()
    Histograma(img)

    copia = Image.new('RGB', img.size)
    datosImg = Image.Image.getdata(img)
    copia.putdata(datosImg)
    ancho, alto = img.size

    for i in range(ancho):
        for j in range(alto):
            r, g, b = copia.getpixel((i,j))
            x = (r + g + b) / 3
            intx = int (x)
            pixel = tuple ([intx, intx, intx])
            copia.putpixel((i,j), pixel)

    for i in range(ancho):
        for j in range(alto):
            vecindades = ObtenerVecinos(img, i, j)
            maxi = max(((vecindades[0][0]), (vecindades[1][0]), (vecindades[2][0]), 
                        (vecindades[3][0]), (vecindades[4][0]), (vecindades[5][0]), 
                        (vecindades[6][0]), (vecindades[7][0]), (vecindades[8][0])))
            res = maxi
            pixel = tuple([res, res, res])
            copia.putpixel((i, j), pixel)

    copia.show()
    copia.save("Max.jpg")
    Histograma(copia)
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def Minimo(nombre):
    tiempo_de_inicio = time.time()
    img = Image.open(nombre)
    img.show()
    Histograma(img)

    copia = Image.new('RGB', img.size)
    datosImg = Image.Image.getdata(img)
    copia.putdata(datosImg)
    ancho, alto = img.size

    for i in range(ancho):
        for j in range(alto):
            r, g, b = copia.getpixel((i,j))
            x = (r + g + b) / 3
            intx = int (x)
            pixel = tuple ([intx, intx, intx])
            copia.putpixel((i,j), pixel)

    for i in range(ancho):
        for j in range(alto):
            vecindades = ObtenerVecinos(img, i, j)
            mini = min(((vecindades[0][0]), (vecindades[1][0]), (vecindades[2][0]), 
                        (vecindades[3][0]), (vecindades[4][0]), (vecindades[5][0]), 
                        (vecindades[6][0]), (vecindades[7][0]), (vecindades[8][0])))
            res = mini
            pixel = tuple([res, res, res])
            copia.putpixel((i, j), pixel)

    copia.show()
    copia.save("Min.jpg")
    Histograma(copia)
    tiempo_de_terminacion = time.time()
    print ("Abrir la imagen tardo: ", tiempo_de_terminacion - tiempo_de_inicio, " segundos")

def Histograma(im):
    im = im.convert('L')
    [ren, col] = im.size
    total = ren * col
    a = np.asarray(im, dtype = np.float32)
    a = a.reshape(1, total)
    a = a.astype(int)
    a = max(a)
    valor = 0
    maxd = max(a)
    grises = maxd
    vec = np.zeros(grises + 1)
    for i in range(total - 1):
        valor = a[i]
        vec[valor] = vec[valor] + 1    
    plt.plot(vec)
    plt.show()

if __name__ == "__main__":
    nombre = input("Ingresa el nombre de la imagen + extension: ")
    opc = int(input('''
    Menu principal:
        1. Agregar ruido Sal y Pimienta
        2. Aplicar filtro Paso Bajas - Promedio
        3. Aplicar filtro Promedio Pesado
        4. Aplicar filtro Gaussiano
        5. Aplicar filtro Mediana
        6. Aplicar filtro Moda
        7. Aplicar filtro Max
        8. Aplicar filtro Min
    Selecciona alguna de las opciones anteriores: '''))
        
    if(opc==1):
        val = input("\nIngresa el nivel de ruido (Ej: 0.0005): ")
        SalPimienta(nombre, float(val))
    elif(opc==2):
        Promedio(nombre)
    elif(opc==3):
        val = input("\nIngresa el nivel de suavizado (N>1): ")
        PromedioPesado(nombre, int(val))
    elif(opc==4):
        val = input("\nIngresa el nivel de suavizado (Ej: 1): ")
        FiltroGaussiano(nombre, int(val))
    elif(opc==5):
        val = input("\nIngresa el nivel de suavizado (Ej: 5): ")
        Mediana(nombre, int(val))
    elif(opc==6):
        Moda(nombre)
    elif(opc==7):
        Maximo(nombre)
    elif(opc==8):
        Minimo(nombre)
    #else:

