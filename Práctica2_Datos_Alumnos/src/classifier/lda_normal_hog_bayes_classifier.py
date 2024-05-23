# @brief LdaNormalHogBayesClassifier

import cv2
import numpy as np
from classifier.lda_normal_bayes_classifier import LdaNormalBayesClassifier

class LdaNormalHogBayesClassifier(LdaNormalBayesClassifier):

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)

    def extract_features(self, img):
        """
        Extract HOG (Histogram of Oriented Gradients) features from the image.
        """
        winSize = (25, 25)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (5, 5)
        nbins = 9

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        h = hog.compute(img)
        return h.flatten()
