# Self-Driving Car – End-to-End Lane Following

This project implements an **end-to-end self-driving car** using **NVIDIA’s CNN architecture** to map **raw camera images directly to steering angles**.  
The model is inspired by **Sully Chen’s behavioral cloning project** and updated to **TensorFlow 2.x / Keras**.

---

## Features

- **End-to-end CNN** for autonomous lane following (NVIDIA DAVE-2 style).
- **TensorFlow 2.x + Keras** implementation for modern compatibility.
- **Data preprocessing & augmentation**:
  - Cropping to focus on the road
  - Flipping and brightness adjustments for drift recovery
  - Normalization for stable training
- **Real-time inference pipeline** for driving simulation.

---

## Dataset

- Trained on **~70,000 labeled frames (~2.2 GB)** with steering angles.
- Dataset can be collected using:
  - **Udacity Self-Driving Car Simulator** (behavioral cloning mode)
  - Real-world driving with a front-facing camera
- **Augmentation** applied for:
  - Off-center drift
  - Brightness and shadow variations

---

## Training Details

- **Model Architecture**: 5 Convolutional layers + Fully Connected layers → 1 steering output  
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizer**: Adam  
- **Epochs**: 50  
- **Performance**:
  - ~**90% lane-following accuracy** in validation tests
  - **25–30 FPS inference** on GPU for real-time driving simulation

---

## Demo Video

https://github.com/user-attachments/assets/163f9151-75ee-413c-93ce-a74e772237ee
# Self_Driving_Car_END-to-End-Lane-Following
