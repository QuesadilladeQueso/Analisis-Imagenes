
import tkinter

ventana = tkinter.Tk()
ventana.geometry("600x600")

etiqueta = tkinter.Label(ventana, text="Practica 1")
etiqueta.pack()

def saludo():
    print("Hola "+nombre)

boton1 = tkinter.Button(ventana, text="Aceptar", padx = 20, pady = 15, command = lambda: saludo)
boton1.pack()

ventana.mainloop()


