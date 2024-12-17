import numpy as np

class Metrics:

    @staticmethod
    def mse(image1, image2):
        return np.mean(np.square(image1 - image2))
    
    @staticmethod
    def psnr(image1, image2):
        mse = Metrics.mse(image1, image2)
        return 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-9))