# -*- coding: utf-8 -*-

# M1 GPHY Image. TP 02 : alignement ICIA

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageAlignerIcia(object):
    """Classe objet générique (abstraite) pour aligner deux images avec l'algorithme ICIA"""

    @staticmethod
    def _showImage(img, ax, title=''):
        ax.cla()
        ax.imshow(img, cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=255)
        ax.set_title(title)

    @classmethod
    def _showAlignmentResult(cls, img1, img2, ax, title='Superposition'):
        img_diff = np.zeros(img1.shape[:2]+(3,), dtype=np.uint8)
        img_diff[:,:,0] = img1
        img_diff[:,:,1] = img2
        img_diff[:,:,2] = img1/2+img2/2
        mse = np.sqrt(np.sum((img1-img2)**2)) / img1.size
        cls._showImage(img_diff, ax, title='{}\nMean-square error: {:.3f}'.format(title, mse))

    @staticmethod
    def _warpImageAffine(src, dst, matrix):
        return cv2.warpAffine(src=src, M=matrix[:2], dsize=None, dst=dst, flags=cv2.WARP_INVERSE_MAP)

    @staticmethod
    def _computeG(img):
        g = np.zeros(img.shape + (2,))
        g[:,:,0] = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3, scale=0.125)
        g[:,:,1] = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3, scale=0.125)
        return g

    @staticmethod    
    def _getDwawa(dp):
        raise NotImplementedError('You need to define a _getDwawa method!')
    
    @staticmethod    
    def _getK(x,y):
        raise NotImplementedError('You need to define a _getK method!')
    
    @staticmethod    
    def _getNbp():
        raise NotImplementedError('You need to define a _getNbp method!')
    
    @classmethod
    def alignImages(cls, img_target, img_template, num_iter_max=50, epsilon=1e-3):
        img_target = img_target.astype(np.float32)
        img_template = img_template.astype(np.float32)
        
        wawa = np.eye(3,3)
        dwawa = np.zeros((3,3))
        idwawa = np.zeros((3,3))
        nbp = cls._getNbp()
        cheucheu = np.zeros((nbp,nbp))
        icheucheu = np.zeros((nbp,nbp)) 
        b = np.zeros(nbp)
        dp = np.zeros(nbp)
        sdi = np.zeros(img_template.shape + (nbp,))

        img_warped = img_target.copy()
        g = cls._computeG(img_template)

        for y in range(img_template.shape[0]):
            for x in range(img_template.shape[1]):
                k = cls._getK(x,y)
                sdi[y,x] = np.dot(g[y,x], k)
                cheucheu += np.matmul(np.transpose([sdi[y,x]]), [sdi[y,x]]) #calcul of hessian matrix
        icheucheu = np.linalg.inv(cheucheu) #inversion de matrice
    
        fig, axarray = plt.subplots(2, 2, figsize=(12,12), dpi=80)
        plt.ion()
        fig.suptitle('ICIA running...', fontsize=14)
        cls._showAlignmentResult(img_template, img_target, axarray[1][0], title='Initialization')
        cls._showImage(img_template, axarray[0][0], title='Template image')
        cls._showImage(img_target, axarray[0][1], title='Target image')

        it=0
        while(it < num_iter_max):
            cls._warpImageAffine(src=img_target, dst=img_warped, matrix=wawa)
        
#            b = np.zeros(nbp)
#            for y in range(img_template.shape[0]):
#                for x in range(img_template.shape[1]):
#                    D = img_warped[y,x] - img_template[y,x]
#                    b += D * sdi[y,x]
            b[:] = np.sum([[(img_warped[y,x] - img_template[y,x]) * sdi[y,x] for y in range(img_template.shape[0])] for x in range(img_template.shape[1])], axis=(0,1))
            #step7, ICIA
            dp = np.matmul([b], icheucheu).reshape(-1) #delta p
            dwawa = cls._getDwawa(dp)
            idwawa = np.linalg.inv(dwawa)
            wawa = np.dot(wawa, idwawa)
            it += 1

            cls._showAlignmentResult(img_template, img_warped, axarray[1][1], title='Alignement. Iter:{:3d}'.format(it))
            plt.pause(0.01)
            plt.draw()
            
            if np.max(np.abs(dp)) < epsilon:
                break
            
        fig, axarray = plt.subplots(1, 3, figsize=(15,5), dpi=80)
        fig.suptitle('ICIA results', fontsize=14)
        cls._showImage(img_template, axarray[0], title='Template')
        cls._showImage(img_target, axarray[1], title='Target')
        cls._showImage(img_warped, axarray[2], title='Aligned target')

        return wawa

class ImageAlignerIciaTranslation(ImageAlignerIcia):
    """Classe objet pour aligner deux images avec l'algorithme ICIA et un modèle de translation. C'est moi qu'il faut utiliser !"""

    @staticmethod    
    def _getDwawa(dp):
        return np.array([[1.0, 0.0, dp[0]], [0.0, 1.0, dp[1]], [0.0, 0.0, 1.0]])
    
    @staticmethod    
    def _getK(x,y):
        return np.matrix([[1,0],[0,1]])
    
    @staticmethod       
    def _getNbp():
        return 2

class ImageAlignerIciaSimilarity(ImageAlignerIcia):
    """Classe objet pour aligner deux images avec l'algorithme ICIA et un modèle de similarité. C'est moi qu'il faut décommenter"""

    @staticmethod    
    def _getDwawa(dp):
        return np.array([[1.0+dp[0], dp[1], dp[2]], [-dp[1], 1.0+dp[0], dp[3]], [0.0, 0.0, 1.0]])
    
    @staticmethod    
    def _getK(x,y):
        return np.matrix([[x, y, 1, 0],[y, -x, 0, 1]])
    
    @staticmethod    
    def _getNbp():
        return 4

def main(argv):
    plt.close('all')
    np.set_printoptions(precision=3, suppress=True)

    img1 = cv2.imread('m1gphy_tp_01_alignement_irm1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('m1gphy_tp_01_alignement_irm2.png', cv2.IMREAD_GRAYSCALE)
    res = ImageAlignerIciaSimilarity.alignImages(img2, img1)
    print('Result:\n{}'.format(res))

    plt.show()

if __name__ == '__main__':
    main(sys.argv)





