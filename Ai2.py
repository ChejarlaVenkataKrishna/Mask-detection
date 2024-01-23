import cv2
import numpy as np

# Load the pre-trained face and mask detection cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mask_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mask.xml')  # You may need to train or find a pre-trained mask classifier

# Load an image
img = cv2.imread('My photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Loop over each face and try to detect a mask
for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]
    
    # Detect masks within the region of interest (ROI)
    masks = mask_cascade.detectMultiScale(face_roi)
    
    # Draw rectangles around faces and masks
    for (mx, my, mw, mh) in masks:
        cv2.rectangle(img, (x + mx, y + my), (x + mx + mw, y + my + mh), (0, 255, 0), 2)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Mask Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
