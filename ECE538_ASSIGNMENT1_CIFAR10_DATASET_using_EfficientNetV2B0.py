# Name: Henry Hoang
# Date: 9/14/25
# Course: ECE 538
# Assignment 1 - PART 4 - Train Image dataset - "CIFAR-10" - and evaluate the EfficientNetV2B0 using Tensorflow

# Disable oneDNN optimizations in TensorFlow (optional; may help with reproducibility or debugging)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize and resize images
x_train = tf.image.resize(x_train.astype('float32') / 255.0, (96, 96))
x_test = tf.image.resize(x_test.astype('float32') / 255.0, (96, 96))

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create datasets
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Build model
def build_model(num_classes):
    from tensorflow.keras.applications import EfficientNetV2B0
    base_model = EfficientNetV2B0(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate and train model
model_cifar10 = build_model(10)
history = model_cifar10.fit(train_data, validation_data=val_data, epochs=10)

# Predict on validation data
start_time = time.time()
y_pred = model_cifar10.predict(val_data)
end_time = time.time()

# Compute throughput and latency
total_time = end_time - start_time
num_samples = x_test.shape[0]
throughput = num_samples / total_time
latency = (total_time / num_samples) * 1000

# Print hardware metrics
print("\nHardware Metrics:")
print(f"Total Inference Time: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Average Latency: {latency:.2f} ms/sample")

# Convert predictions and labels to class indices
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
y_true_classes = tf.argmax(y_test, axis=1).numpy()

# Evaluation metrics
acc = accuracy_score(y_true_classes, y_pred_classes)
prec = precision_score(y_true_classes, y_pred_classes, average='weighted')
rec = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Print evaluation results
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Accuracy and loss plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Model summary
model_cifar10.summary()
