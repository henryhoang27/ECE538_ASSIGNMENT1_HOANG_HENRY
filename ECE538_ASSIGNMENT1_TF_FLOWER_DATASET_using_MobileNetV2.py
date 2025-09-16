# Name: Henry Hoang
# Date: 9/14/25
# Course: ECE 538
# Assignment 1 - PART 1 - Train Image dataset - "TF_Flower" - and evaluate the MobileNETV2 using Tensorflow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Load the TensorFlow Flowers dataset
flowers = tfds.load('tf_flowers', split='train', as_supervised=True)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (96, 96)) / 255.0
    label = tf.one_hot(label, depth=5)
    return image, label

# Shuffle and preprocess
flowers = flowers.shuffle(1000).map(preprocess)

# Split into training and validation sets
total_batches = tf.data.experimental.cardinality(flowers).numpy()
val_batches = int(0.2 * total_batches)
val_data = flowers.take(val_batches).batch(32)
train_data = flowers.skip(val_batches).batch(32)

# Build model
def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_flowers = build_model(5)

# Train model
history = model_flowers.fit(train_data, validation_data=val_data, epochs=10)

# Reload validation data for evaluation
flowers_for_eval = tfds.load('tf_flowers', split='train', as_supervised=True)
flowers_for_eval = flowers_for_eval.map(preprocess)
val_data_eval = flowers_for_eval.take(val_batches).batch(32)

# Generate predictions
y_pred_flowers = model_flowers.predict(val_data_eval)
y_pred_flowers_classes = tf.argmax(y_pred_flowers, axis=1).numpy()

# Measure latency and throughput
start_time = time.time()
_ = model_flowers.predict(val_data_eval)
end_time = time.time()
total_time = end_time - start_time
num_samples = val_batches * 32
throughput = num_samples / total_time
latency = (total_time / num_samples) * 1000



# Reload validation data again to extract true labels
flowers_for_eval = tfds.load('tf_flowers', split='train', as_supervised=True)
flowers_for_eval = flowers_for_eval.map(preprocess)
val_data_eval = flowers_for_eval.take(val_batches).batch(32)

# Extract true labels
y_true_flowers = []
for batch in val_data_eval:
    labels = tf.argmax(batch[1], axis=1)
    y_true_flowers.extend(labels.numpy())

# Evaluation metrics
acc_flowers = accuracy_score(y_true_flowers, y_pred_flowers_classes)
prec_flowers = precision_score(y_true_flowers, y_pred_flowers_classes, average='weighted')
rec_flowers = recall_score(y_true_flowers, y_pred_flowers_classes, average='weighted')
f1_flowers = f1_score(y_true_flowers, y_pred_flowers_classes, average='weighted')

# Print evaluation results
print("Accuracy:", acc_flowers)
print("Precision:", prec_flowers)
print("Recall:", rec_flowers)
print("F1 Score:", f1_flowers)

# Print hardware metrics
print("\nHardware Metrics:")
print(f"Total Inference Time: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Average Latency: {latency:.2f} ms/sample")

# Confusion matrix
cm = confusion_matrix(y_true_flowers, y_pred_flowers_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy and loss
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
model_flowers.summary()
