# Name: Henry Hoang
# Date: 9/14/25
# Course: ECE 538
# Assignment 1 - PART 3 - Train Image dataset - "TF Flowers" - and evaluate the EfficientNetV2B0 using Tensorflow

# Disable oneDNN optimizations in TensorFlow (optional; may help with reproducibility or debugging)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import core libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Load the TF-Flowers dataset and metadata
dataset, info = tfds.load('tf_flowers', as_supervised=True, with_info=True)
num_classes = info.features['label'].num_classes
total_examples = info.splits['train'].num_examples

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (96, 96))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label

# Split dataset
val_size = int(0.2 * total_examples)
train_size = total_examples - val_size
raw_train = dataset['train'].take(train_size)
raw_val = dataset['train'].skip(train_size)

# Prepare datasets
train_data = raw_train.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_data = raw_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

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
model_flowers = build_model(num_classes)
history = model_flowers.fit(train_data, validation_data=val_data, epochs=10)

# Evaluate model performance
y_true = []
y_pred = []

# Measure latency and throughput
start_time = time.time()

for images, labels in val_data:
    preds = model_flowers.predict(images)
    y_pred.extend(tf.argmax(preds, axis=1).numpy())
    y_true.extend(tf.argmax(labels, axis=1).numpy())

end_time = time.time()
total_time = end_time - start_time
num_samples = val_size
throughput = num_samples / total_time
latency = (total_time / num_samples) * 1000



# Compute evaluation metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Display evaluation results
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# Print hardware metrics
print("\nHardware Metrics:")
print(f"Total Inference Time: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Average Latency: {latency:.2f} ms/sample")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training and validation accuracy and loss
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
