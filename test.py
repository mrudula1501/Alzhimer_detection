import cv2
import numpy as np
import math
from PIL import Image, ImageOps
from cvzone.ClassificationModule import Classifier

classifier = Classifier("CNN Model/keras_model.h5", "CNN Model/labels.txt")

offset = 20
imgSize = 224


labels = ["Healthy", "Parkinson's Disease Detected"]


def DetectParkinson():

    img = cv2.imread('test1.jpg')
    size = (224, 224)
    #img = ImageOps.fit(img, size, Image.ANTIALIAS)
    prediction, index = classifier.getPrediction(img, draw=False)
    print(prediction,index)

DetectParkinson()