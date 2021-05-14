#python 3.7.4,opencv4.1
#  https://blog.csdn.net/caimouse/article/details/51749579
#
import cv2
import numpy as np
from scipy import signal
 
def edgesMarrHildreth(img, sigma):
 
    """
            finds the edges using MarrHildreth edge detection method...
            :param im : input image
            :param sigma : sigma is the std-deviation and refers to the spread of gaussian
            :return:
            a binary edge image...
    """
 
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
 
    return zero_crossing
 
 #Picture path
imgname = "edge1.png"
 
 #Read picture
image = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
 
 #Image height and width
h,w = image.shape[:2]
print('imagesize={}-{}'.format(w,h))
 
 #Show original image
cv2.imshow("Image",image)
 
 #operator
MH = edgesMarrHildreth(image, 3)
MH = MH.astype(np.uint8)
cv2.imshow("MH",MH)
 
cv2.waitKey(0)
cv2.destroyAllWindows()