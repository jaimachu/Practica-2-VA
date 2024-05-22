import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import random

class MainPanelsOCR:
    def __init__(self):
        None
    
    """
    Función principal que devolverá las regiones detectadas de los caracteres, que se utilizará posteriormente por el clasificador
    """
    def obtainRegionsDetected(self, pathImagen):
        image = cv2.imread(pathImagen, 0)
        imageContours = self.umbraliceImage(pathImagen)
        rectanglesDetected = self.mser(imageContours)
        centers = self.computeCenters(rectanglesDetected)
        rectanglesDetected, centers = self.eliminateDuplicatedRectangles(rectanglesDetected, centers)
        clusterCenters = self.groupCenters(centers) # Conjuntos de puntos que pertencen a cada linea
        lines = self.getLines(clusterCenters, image)
        self.drawDetection(pathImagen, clusterCenters, rectanglesDetected, lines)
        return rectanglesDetected, clusterCenters


    """
    Aplica un umbralizado y obtiene los bordes
    """
    def umbraliceImage(self, pathImagen):
        image = cv2.imread(pathImagen, 0)
        imageUmbralice = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
        contours, _ = cv2.findContours(imageUmbralice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = np.zeros_like(image) # Cargamos una imagen en negro donde vamos a dibujar los contornos
        cv2.drawContours(imageContours, contours, -1, (255, 255, 255), 1)
        return imageContours

    """
    Detectamos con MSER caracteres y devolvemos las coordenadas en forma de lista de los rectangulos
    """
    def mser(self, imageUmbralice):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        imageUmbralice = cv2.dilate(imageUmbralice, kernel)
        mser = cv2.MSER_create(delta=5, max_variation=0.8, min_area=50, max_area=200)
        polygons, _ = mser.detectRegions(imageUmbralice)
        rectangles = []
        for p in polygons:
            x, y, w, h = cv2.boundingRect(p)
            if (h / w > 0.9 and h / w <= 1.9) or (h / w > 2 and h / w <= 5):
                rectangles.append((x,y,w,h))
        return rectangles

    """
    Calcula los centros de cada cuadro detectado
    """
    def computeCenters(self, rectanglesDetected):
        centers = []
        for rectangle in rectanglesDetected:
            x, y, w, h = rectangle
            center = (x + (w//2), y + (h//2))
            centers.append(list(center))
        return centers

    """
    Calcula el valor de IoU entre dos cajas. Valor entre 0 y 1
    """
    def intersectionOverUnion(self, box1, box2):
        # Obtenemos los puntos x1, y1 y x2, y2 de la región primera
        xA1, yA1, wA, hA = box1
        xA2 = xA1 + wA
        yA2 = yA1 + hA
        # Obtenemos los puntos x1, y1 y x2, y2 de la región segunda
        xB1, yB1, wB, hB = box2
        xB2 = xB1 + wB
        yB2 = yB1 + hB
        # Coordenadas de la intersección
        xI1 = max(xA1, xB1)
        yI1 = max(yA1, yB1)
        xI2 = min(xA2, xB2)
        yI2 = min(yA2, yB2)
        # Área de la intersección
        interWidth = max(0, xI2 - xI1)
        interHeight = max(0, yI2 - yI1)
        interArea = interWidth * interHeight
        box1Area = wA * hA
        box2Area = wB * hB
        unionArea = box1Area + box2Area - interArea
        iou = interArea / unionArea
        return iou

    """
    Elimina rectangulos que están dentro de otros rectangulos. Si supera un umbral de IoU, se elimina
    """
    def eliminateDuplicatedRectangles(self, rectangulos, centros):
        eliminarIndices = set()
        for i in range(len(rectangulos)):
            for j in range(i+1, len(rectangulos)):
                x1, y1, w1, h1 = rectangulos[i]
                x2, y2, w2, h2 = rectangulos[j]
                area1 = w1 * h1
                area2 = w2 * h2
                # Comprobamos viendo los puntos si están contenidos
                iouValue = abs(self.intersectionOverUnion(rectangulos[i], rectangulos[j]))
                if (iouValue > 0.1):
                    if (area1 >= area2):
                        eliminarIndices.add(j)
                    else:
                        eliminarIndices.add(i)
        rectangulosFiltrados = []
        centrosFiltrados = []
        for i in range(len(rectangulos)):
            if i not in eliminarIndices:
                rectangulosFiltrados.append(rectangulos[i])
                centrosFiltrados.append(centros[i])
        return rectangulosFiltrados, centrosFiltrados

    """
    Agrupamos cada caracter con su conjunto de palabras de la linea
    """
    def groupCenters(self, points, min_samples=2, residual_threshold=10):
        lines = []
        remaining_points = points.copy()
        
        while len(remaining_points) >= min_samples:
            X = np.array([p[0] for p in remaining_points]).reshape(-1, 1)
            Y = np.array([p[1] for p in remaining_points])
            
            ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold)
            ransac.fit(X, Y)
            
            inliers_mask = ransac.inlier_mask_
            outliers_mask = np.logical_not(inliers_mask)
            
            # Store the inliers as a line
            inliers = [remaining_points[i] for i in range(len(remaining_points)) if inliers_mask[i]]
            lines.append(inliers)
            
            # Update remaining points to outliers only
            remaining_points = [remaining_points[i] for i in range(len(remaining_points)) if outliers_mask[i]]
        return lines
    
    """
    Traza las líneas de cada conjunto de puntos
    """
    def getLines(self, puntosAgrupados, imagen):
        alto, ancho = imagen.shape
        lines = []
        ransac = RANSACRegressor()
        for puntos in puntosAgrupados:
            centrosX = [sublista[0] for sublista in puntos]
            centrosY = [sublista[1] for sublista in puntos]
            centrosX = np.array(centrosX).reshape(-1, 1)
            centrosY = np.array(centrosY)

            ransac.fit(centrosX, centrosY)
            x0 = centrosX.min()
            x1 = centrosX.max()  # Usamos el rango de nuestros datos
            y0 = ransac.estimator_.intercept_
            y1 = x1 * ransac.estimator_.coef_[0] + ransac.estimator_.intercept_
            start_point = (0, int(y0))
            end_point = (ancho-1, int(y1))
            lines.append([start_point, end_point])
        return lines
    
    """
    Dibuja las líneas, los recuadros y puntos detectados
    """
    def drawDetection(self, pathImage, clusterPoints, rectangles, lines):
        image = cv2.imread(pathImage)
        for cluster in clusterPoints:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for center in cluster:
                cv2.circle(image, (center[0], center[1]), radius=1, color=color, thickness=2)
        for rectangle in rectangles:
            x, y, w, h = rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        for line in lines:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            start_point = line[0]
            end_point = line[1]
            cv2.line(image, start_point, end_point, color=color, thickness=2)
        cv2.imshow("Centros detectados", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

