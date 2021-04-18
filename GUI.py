from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2

def Seleccion_Imagen(panel):
    
    ruta = filedialog.askopenfilename(filetypes= {
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")
    })
    
    if len(ruta) > 0:
        
        image = cv2.imread(ruta)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        
        image = ImageTk.PhotoImage(image)

        if panel is None:
            
            panel = Label(image=image)
            panel.image = image
            panel.pack(side = "top",padx=10, pady=10)
            
        else:
            
            panel.configure(image=image)
            panel.image = image


if __name__ == '__main__':
    
    raiz = Tk()
    panel = None
    
    
    btn = Button(raiz, text="Selecciona una imágen", command=Seleccion_Imagen(panel) )
    btn.pack(fill="both", expand="yes", padx="10", pady="10")
    
    raiz.mainloop()