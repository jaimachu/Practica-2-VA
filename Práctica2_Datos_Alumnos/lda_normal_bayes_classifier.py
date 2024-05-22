# @brief LdaNormalBayesClassifier


# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla
import cv2
import numpy as np
from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        

    def training(self):
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(self.X_reduced.astype(np.float32), cv2.ml.ROW_SAMPLE, self.y.astype(np.int32))
    
    def predicting(self):
        _, predicted_label = self.classifier.predict(self.features_reduced.astype(np.float32))
        return int(predicted_label[0, 0])