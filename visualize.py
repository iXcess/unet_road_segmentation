#!/usr/bin/env python3
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
from skimage.transform import resize

camerafile = sys.argv[1]
model = load_model('model.h5')

IMG_HEIGHT = 128
IMG_WIDTH = 128

cap = cv2.VideoCapture(camerafile)

while True:
  plt.clf()
  plt.title("Kommu Segnet")
  (ret, current_frame) = cap.read()
  if not ret:
       break

  frame = current_frame.copy()
  frame_ = resize(frame, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
  train = model.predict(frame_[np.newaxis, ...])
  img = resize(train[0], current_frame.shape, mode='constant', preserve_range=True)
  plt.imshow(img)

  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  


