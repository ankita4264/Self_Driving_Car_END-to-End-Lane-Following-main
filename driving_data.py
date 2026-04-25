import cv2
import numpy as np
import random

xs = []
ys = []

# Points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# Read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        # The paper by Nvidia uses the inverse of the turning radius,
        # but steering wheel angle is proportional to the inverse of turning radius
        # so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * np.pi / 180)

# Get number of images
num_images = len(xs)

# Split data into training and validation sets
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Read and preprocess the image
        image = cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image[-150:], (200, 66)) / 255.0  # Crop and resize
        x_out.append(image)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return np.array(x_out), np.array(y_out)

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Read and preprocess the image
        image = cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image[-150:], (200, 66)) / 255.0  # Crop and resize
        x_out.append(image)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return np.array(x_out), np.array(y_out)
