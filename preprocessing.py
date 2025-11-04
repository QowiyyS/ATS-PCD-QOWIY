import cv2
import numpy as np

def step1_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def step2_denoise(image):
    return cv2.medianBlur(image, 3)

def step3_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def step4_threshold(image):
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def step5_edges(image):
    return cv2.Canny(image, 100, 200)

def step6_morph(image):
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def step7_resize_normalize(image):
    resized = cv2.resize(image, (224,224))
    normalized = resized / 255.0
    return normalized
