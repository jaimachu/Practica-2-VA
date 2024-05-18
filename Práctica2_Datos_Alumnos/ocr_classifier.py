# @brief OCRClassifier


import string

class OCRClassifier:
    """
    Classifier for Optical Character Recognition
    """

    def __init__(self, ocr_char_size=(25, 25)):
        self.ocr_char_size = ocr_char_size
        self.classifier_name = None


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

    def predict_dict(self, images_dict):
        responses = []
        for key in images_dict:
            for img in images_dict[key]:
                responses.append(self.predict(img))
        return responses


