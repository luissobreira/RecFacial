import hashlib
import io
import numpy as np
import cv2
from PIL import Image, ImageEnhance

def adjust_image(img_pil, brightness=1.0, contrast=1.0, sharpness=1.0, equalize=False, grayscale=False):
    """
    Aplica filtros de aprimoramento de imagem para análise forense/CFTV.
    """
    res = img_pil.copy()
    
    if grayscale:
        res = res.convert('L').convert('RGB')
        
    if equalize:
        # Converter para OpenCV para aplicar CLAHE (Equalização de Histograma Adaptativa)
        img_np = np.array(res)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        res = Image.fromarray(img_np)
        
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(res)
        res = enhancer.enhance(brightness)
        
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(res)
        res = enhancer.enhance(contrast)
        
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(res)
        res = enhancer.enhance(sharpness)
        
    return res

def blend_images(img1_pil, img2_pil, alpha=0.5, target_size=(300, 300)):
    """
    Funde duas imagens ajustadas/recortadas para verificação de sobreposição anatômica.
    alpha = 0.0 -> Apenas Imagem 1
    alpha = 1.0 -> Apenas Imagem 2
    """
    im1 = img1_pil.copy().convert('RGB').resize(target_size)
    im2 = img2_pil.copy().convert('RGB').resize(target_size)
    
    blended = Image.blend(im1, im2, alpha)
    return blended

def get_image_hash(img_pil):
    """Gera hash SHA-256 das imagens para cadeia de custódia pericial."""
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return hashlib.sha256(buf.getvalue()).hexdigest()
