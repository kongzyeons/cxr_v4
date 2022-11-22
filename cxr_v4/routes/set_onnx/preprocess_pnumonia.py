from re import L
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps, ImageFilter



def tranfrom_image(w,h):
    return transforms.Compose([transforms.Resize((w, h)),
        transforms.Grayscale(1),
        ContrastBrightness(1.2,25),
        HistEqualization(),
        SmothImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.25])])


class HistEqualization(object):
    """Image pre-processing.

    Equalize the image historgram
    """
    
    def __call__(self,image):
        
        return ImageOps.equalize(image, mask = None) 
    
class ContrastBrightness(object):
    """Image pre-processing.

    alpha = 1.0 # Simple contrast control [1.0-3.0]
    beta = 0    # Simple brightness control [0-100]
    """
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self,image,):
        image = np.array(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                image[y,x] = np.clip(self.alpha*image[y,x] + self.beta, 0, 255)

                return Image.fromarray(np.uint8(image)*255)

class SmothImage(object):
    """Image pre-processing.

    Smooth the image
    """
    def __call__(self,image):
        
        return image.filter(ImageFilter.SMOOTH_MORE)
