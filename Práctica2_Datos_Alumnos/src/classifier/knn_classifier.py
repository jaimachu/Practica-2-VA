import cv2
from sklearn.neighbors import KNeighborsClassifier

from classifier.ocr_classifier import OCRClassifier

class KNNClassifier(OCRClassifier):
    def __init__(self, ocr_char_size=(25, 25), n_neighbors=3):
        super().__init__(ocr_char_size)
        self.classifier_name = "KNN"
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def training(self):
        self.classifier.fit(self.CR, self.E)

    def predicting(self):
        predicted_label = self.classifier.predict(self.features_reduced)
        return int(predicted_label[0])

