import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import tensorflow as tf
import cv2
import numpy as np

# Suppress OpenCV logs
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# Load the model
model = tf.keras.models.load_model("save/model.ckpt")

# Load the steering wheel image
img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

# Open the webcam
cap = cv2.VideoCapture(0)

while cv2.waitKey(10) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    image = cv2.resize(frame, (200, 66)) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the steering angle
    degrees = model.predict(image)[0][0] * 180 / np.pi

    # Remove print statement (No console output)
    # print("Predicted steering angle: " + str(degrees) + " degrees")

    cv2.imshow('frame', frame)

    # Make smooth angle transitions
    if abs(degrees - smoothed_angle) > 1e-3:  # Avoid division by zero
        smoothed_angle += 0.2 * pow(abs(degrees - smoothed_angle), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

cap.release()
cv2.destroyAllWindows()
