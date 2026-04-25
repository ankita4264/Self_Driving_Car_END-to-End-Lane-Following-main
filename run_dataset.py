import tensorflow as tf
import cv2
import numpy as np
import math
import os

# Load the model with proper error handling
model_path = "save/model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model is trained and saved correctly.")

try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load the steering wheel image
img_path = 'steering_wheel_image.jpg'
if not os.path.exists(img_path):
    raise FileNotFoundError("Steering wheel image 'steering_wheel_image.jpg' not found.")

img = cv2.imread(img_path, 0)
rows, cols = img.shape
smoothed_angle = 0

# Read dataset information
data_path = "driving_dataset/data.txt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file '{data_path}' not found. Ensure the dataset is correctly placed.")

xs, ys = [], []
with open(data_path, "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 2:
            continue
        image_file = os.path.join("driving_dataset", parts[0])
        angle = float(parts[1]) * np.pi / 180  # Convert degrees to radians
        xs.append(image_file)
        ys.append(angle)

num_images = len(xs)
if num_images == 0:
    raise ValueError("No images found in dataset.")

# Start from 80% of dataset
i = math.ceil(num_images * 0.8)
print(f"Starting from frame {i} of {num_images}")

while i < num_images:
    image_path = xs[i]
    if not os.path.exists(image_path):
        print(f"Warning: Image '{image_path}' not found. Skipping...")
        i += 1
        continue

    full_image = cv2.imread(image_path)
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict steering angle
    try:
        degrees = model.predict(image, verbose=0)[0][0] * 180.0 / np.pi
    except Exception as e:
        print(f"Prediction error at frame {i}: {e}")
        i += 1
        continue

    print(f"Steering angle: {degrees:.2f}° (predicted) | {ys[i] * 180 / np.pi:.2f}° (actual)")

    # Add label showing steering angle (only text, no lines)
    cv2.putText(full_image, f"Angle: {degrees:.2f}°", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display video frame
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

    # Smooth steering wheel animation
    angle_diff = degrees - smoothed_angle
    smoothed_angle += 0.2 * pow(abs(angle_diff), 2.0 / 3.0) * np.sign(angle_diff)

    # Rotate steering wheel image
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    i += 1

cv2.destroyAllWindows()
