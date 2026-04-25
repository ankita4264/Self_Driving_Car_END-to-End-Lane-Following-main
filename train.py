import os
import tensorflow as tf
from tensorflow.keras import optimizers
import driving_data
import model

# Define constants
LOGDIR = './save'
L2NormConst = 0.001
epochs = 30
batch_size = 100

# Load the model
from model import create_model
model = create_model()


# Define the loss function
def loss_fn(y_true, y_pred):
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables]) * L2NormConst
    return tf.reduce_mean(tf.square(y_true - y_pred)) + l2_loss

# Define the optimizer
optimizer = optimizers.Adam(learning_rate=1e-4)

# Training loop
for epoch in range(epochs):
    for i in range(int(driving_data.num_images / batch_size)):
        # Load training batch
        xs, ys = driving_data.LoadTrainBatch(batch_size)

        # Perform a training step
        with tf.GradientTape() as tape:
            y_pred = model(xs, training=True)
            loss_value = loss_fn(ys, y_pred)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Print training progress
        if i % 10 == 0:
            # Load validation batch
            xs_val, ys_val = driving_data.LoadValBatch(batch_size)
            val_loss = loss_fn(ys_val, model(xs_val, training=False))
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss_value.numpy()}, Val Loss: {val_loss.numpy()}")

        # Save the model checkpoint as .h5
        if i % batch_size == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "model.h5")
            model.save(checkpoint_path)
            print(f"Model saved in file: {checkpoint_path}")

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
