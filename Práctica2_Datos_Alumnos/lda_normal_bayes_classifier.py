# @brief LdaNormalBayesClassifier


# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Initialize lists to store features and labels
        X = []
        y = []

        # Extract features from each image and its corresponding label
        for char, images in images_dict.items():
            for img in images:
                # Preprocess the image (e.g., thresholding, resizing)
                processed_img = self.preprocess_image(img)

                # Extract features from the image
                features = self.extract_features(processed_img)

                # Append features and label to X and y
                X.append(features)
                y.append(ord(char[0]))

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Perform LDA training
        self.lda = LinearDiscriminantAnalysis()
        X_reduced = self.lda.fit_transform(X, y)

        # Train the classifier with the reduced features
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(X_reduced.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.int32))

        return X, y

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """
        # Preprocess the image
        processed_img = self.preprocess_image(img)

        # Extract features from the image
        features = self.extract_features(processed_img)

        # Reduce the features using LDA
        features_reduced = self.lda.transform([features])

        # Use the trained classifier to predict the label
        _, predicted_label = self.classifier.predict(features_reduced.astype(np.float32))

        return int(predicted_label[0, 0])

    def preprocess_image(self, img):
        """
        Preprocess the input image (e.g., thresholding, resizing).

        :img Input image
        """

        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours and get the bounding box
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            img = img[y:y+h, x:x+w]

        # Resize to a fixed size (25x25 pixels)
        img = cv2.resize(img, (25, 25), interpolation=cv2.INTER_AREA)

        return img

    def extract_features(self, img):
        """
        Extract features from the input image.

        :img Input image
        """

        # Flatten the image to create a feature vector
        return img.flatten()