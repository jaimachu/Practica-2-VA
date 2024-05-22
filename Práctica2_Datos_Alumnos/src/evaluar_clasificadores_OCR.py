# Asignatura de Visión Artificial (URJC). Script de evaluación.
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import sklearn.metrics
import os
from classifier.lda_normal_hog_bayes_classifier import LdaNormalHogBayesClassifier
from classifier.lda_normal_bayes_classifier import LdaNormalBayesClassifier
from classifier.knn_classifier import KNNClassifier

def load_images_from_folder(folder):
    # Returns a dictionary where keys are class labels and values are lists of images.
    images_dict = {}
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            # Check if it is a folder of numbers
            if label.isdigit():
                images = []
                for filename in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                images_dict[label] = images
            # Check if it is a folder of letters (Mayusculas or minusculas)
            else:
                for sublabel in os.listdir(label_folder):
                    sublabel_folder = os.path.join(label_folder, sublabel)
                    if os.path.isdir(sublabel_folder):
                        images = []
                        for filename in os.listdir(sublabel_folder):
                            img_path = os.path.join(sublabel_folder, filename)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                images.append(img)
                        images_dict[sublabel] = images
    return images_dict

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

if __name__ == "__main__":

    while(True):
        print("\nMenu de clasificadores:")
        print("1.- LDA Normal Bayes")
        print("2.- LDA Normal Bayes extract features with Hog")
        print("3.- LDA KNN\n")

        option = input("Seleccione el numero de un clasificador: ")
        if option == '1': 
            clasiffier = "lda_normal_bayes"
            break
        if option == '2':
            clasiffier = "lda_normal_hog"
            break
        if option == '3':
            clasiffier = "lda_KNN"
            break
        else: 
            print("\nElija una de las opciones disponibles\n")

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str, default=clasiffier, help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./validation_ocr", help='Select the validation data dir')

    args = parser.parse_args()

    # Load training data
    print("Loading training data...")
    train_images_dict = load_images_from_folder(args.train_path)

    # Load validation data
    print("Loading validation data...")
    validation_images_dict = load_images_from_folder(args.validation_path)
    
    # Prepare validation data in the same format
    gt_labels = []
    predicted_labels = []

    # Initialize classifier
    ocr_char_size = 25 * 25  # Assuming fixed size 25x25 pixels
    if args.classifier == "lda_normal_bayes":
        classifier = LdaNormalBayesClassifier(ocr_char_size)
    elif args.classifier == "lda_normal_hog":
        classifier = LdaNormalHogBayesClassifier(ocr_char_size)
    elif args.classifier == "lda_KNN":
        classifier = KNNClassifier(ocr_char_size)
    else:
        raise ValueError("Unknown classifier type: {}".format(args.classifier))

    # Train classifier
    print("Training classifier...")
    classifier.train(train_images_dict)


    # Validate classifier
    print("Validating classifier...")
    total_images = sum(len(images) for images in validation_images_dict.values())
    with tqdm(total=total_images, desc="Procesando imágenes de validación") as pbar:
        for label, images in validation_images_dict.items():
            for img in images:
                gt_labels.append(ord(label[0]))
                predicted_label = classifier.predict(img)
                predicted_labels.append(predicted_label)
                pbar.update(1) 

    # Evaluate results
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    print("Accuracy = ", accuracy)

    # Plot confusion matrix
    cm = sklearn.metrics.confusion_matrix(gt_labels, predicted_labels)
    plot_confusion_matrix(cm)
    plt.show()

