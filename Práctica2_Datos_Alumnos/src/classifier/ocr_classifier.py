# @brief OCRClassifier

from abc import ABC, abstractmethod
import string
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class OCRClassifier(ABC):
    """
    Classifier for Optical Character Recognition
    """

    def __init__(self, ocr_char_size=(25, 25)):
        self.ocr_char_size = ocr_char_size
        self.classifier_name = None

        self.lda = None
        self.classifier = None
        
    def char2label(self, c):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars.find(c)+1


    def label2char(self, label):
        all_chars = '0123456789' + string.ascii_letters
        return all_chars[label-1]

    def get_labels_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.char2label(key))

        return responses    
    
    def extract_features(self, img):
        """
        Extract features from the input image.

        :img Input image
        """

        # Flatten the image to create a feature vector
        return img.flatten()

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

    def predict_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.predict(img))
        return responses

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
        self.features_reduced = self.lda.transform([features])

        # Use the trained classifier to predict the label
        return self.predicting()
    
    @abstractmethod
    def predicting(self):
        None

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Initialize lists to store features and labels
        self.X = []
        self.y = []

        # Extract features from each image and its corresponding label
        for char, images in images_dict.items():
            for img in images:
                # Preprocess the image (e.g., thresholding, resizing)
                processed_img = self.preprocess_image(img)

                # Extract features from the image
                features = self.extract_features(processed_img)

                # Append features and label to X and y
                self.X.append(features)
                self.y.append(ord(char[0]))

        # Convert lists to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # Perform LDA training
        self.lda = LinearDiscriminantAnalysis()
        self.X_reduced = self.lda.fit_transform(self.X, self.y)

        # Train the classifier with the reduced features)
        self.training()

    @abstractmethod
    def training(self):
        None
