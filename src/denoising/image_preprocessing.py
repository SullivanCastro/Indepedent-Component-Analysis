import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt

IMAGES_PATH = 'images/'


class ImagePreprocessing:
    def __init__(self):
        pass

    @staticmethod
    def gray(image):
        if type(image) == Image.Image:
            image = np.array(image)
        gray_filters = np.array([0.2989, 0.5870, 0.1140])
        return ImagePreprocessing.standardize(np.dot(image, gray_filters))

    @staticmethod
    def noising(image_name, noise_level, width=512, height=512):
        I = Image.open(os.path.join(IMAGES_PATH, image_name)).resize((width, height))
        I = ImagePreprocessing.standardize(np.array(I))
        # I = ImagePreprocessing.gray(I)
        noise = np.random.normal(loc=0, scale=noise_level, size=I.shape[0:2])
        return ImagePreprocessing.standardize(I + noise[:, :, None])
    
    @staticmethod
    def standardize(image):
        return ((image - np.min(image)) / (np.max(image) - np.min(image))).squeeze()
    
    @staticmethod
    def denoising_wt(image, wavelet='haar', threshold=0.1):
        coeffs = pywt.dwt2(image, wavelet=wavelet)

        coeffs_denoised = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
        coeffs = [coeffs[0]] + coeffs_denoised

        return pywt.idwt2(coeffs, wavelet=wavelet)