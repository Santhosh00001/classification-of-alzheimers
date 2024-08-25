#!/usr/bin/env python
# coding: utf-8

# In[ ]:


```python


# In[ ]:


get_ipython().system('pip install tensorflow_addons')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# List the contents of the main directory
get_ipython().system('ls /content/drive/My\\ Drive/')


# In[ ]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, metrics
import tensorflow as tf
import tensorflow_addons as tfa


warnings.filterwarnings("ignore")


# Loading Data set

# In[ ]:


import zipfile
import os
import shutil

# Defining the path to your ZIP file in Google Drive
zip_path = '/content/drive/My Drive/archive (1).zip'
extract_path = '/content/mri_images/'

# Extracting the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Files extracted to: {extract_path}")

# Listing the contents of the extracted directory to verify
print("Contents of the extracted directory:")
get_ipython().system('ls /content/mri_images/')


# In[ ]:


# Defining the path to the 'Axial' directory
axial_path = os.path.join(extract_path, 'Axial')

# Listing the directories inside 'Axial'
subdirs = [name for name in os.listdir(axial_path) if os.path.isdir(os.path.join(axial_path, name))]
print("Subdirectories in 'Axial':", subdirs)


# In[ ]:


#loading the dataset
data = tf.keras.utils.image_dataset_from_directory(
    directory=axial_path,      # Using the path to the 'Axial' directory
    image_size=(128, 128),     # Resizing images to 128x128
    batch_size=10000,          # Adjusting the batch size as needed
    label_mode='int',          # Using integer labels (class indices)
    subset=None                # Loading the entire dataset
)


# In[ ]:


class_names=data.class_names #assigning class names names to a variable


# In[ ]:


#creating a dictionary mapping numerical labels to class names
label_map = {m : n for m, n in zip(np.arange(len(class_names)), class_names)}
#printing the dictionary to see the mapping
print(label_map)


# In[ ]:


#iterate over the data
for images, labels in data:
    # Extract the first batch of images and labels
    X = images.numpy().astype("uint8")
    #convert the labels tensor to a numpy array and cast the type to unint8(8-bit unsigned integer)
    y = labels.numpy().astype("uint8")

print(X.shape, y.shape)


# In[ ]:


# Initialize an array of zeros with the same height and width as X but with a single channel for grayscale
X_gray = np.zeros(shape = (*X.shape[:-1], 1))
# Iterate over the images in X
for idx, img in enumerate(X):
  # Convert each RGB image to grayscale using TensorFlow's function and assign it to the corresponding index in X_gray
    X_gray[idx] = tf.image.rgb_to_grayscale(img)
# Make a copy of the grayscale images to X
X = X_gray.copy()
print(X_gray.shape)


# data visualizaton

# In[ ]:


# Seting up a figure with a specified size
plt.figure(figsize=(10, 10))
# Looping to plot the first 16 images
for i in range(16):
  # Geting the i-th image from the grayscale image array X
    img = X[i]
    # Set up a subplot in a 4x4 grid for the i-th image
    plt.subplot(4, 4, i + 1)

    # Convert y[i] to an integer before using it as a key
    label_index = np.argmax(y[i])  # Get the index of the maximum value
    # Use the index to get the corresponding class name from label_map
    label = label_map[label_index]

    plt.title(label)
    plt.gray()
    plt.imshow(img)
    plt.tight_layout()

plt.show()


# In[ ]:


import numpy as np

# Print unique labels
unique_labels = np.unique(y)
print("Unique labels in the dataset:", unique_labels)


# In[ ]:


# Display one example image for each of the first three unique labels
visited = set()
for img, label in zip(X_gray, y):
    if label not in visited:
        visited.add(label)
        print(label_map[label])
        plt.gray()
        plt.imshow(img.squeeze())  # Remove the singleton dimension if present
        plt.show()
        if len(visited) == 3:
            break


# In[ ]:


# Count the occurrences of each label in y and print the counts
label_counts = pd.Series(y).value_counts()
print(label_counts)


# In[ ]:


# Convert numeric labels to class names
y_class_names = [label_map[label] for label in y]


# In[ ]:


# Create and display a bar plot showing the distribution of class labels in y_class_names
plt.figure(figsize=(10,6))
sns.countplot(x = y_class_names )
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# In[ ]:


# Create and display a pie chart showing the percentage distribution of class labels
plt.pie(label_counts, autopct = "%.2f%%", labels = list(map(lambda key : label_map[key], label_counts.keys())))
plt.show()


# data augmentation
# 

# In[ ]:


# Define a custom TensorFlow layer for Elastic Transformation augmentation, which warps images during training
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

# Define the ElasticTransformation layer
class ElasticTransformation(tf.keras.layers.Layer):
    def __init__(self, alpha, sigma, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # Magnitude of deformation
        self.sigma = sigma   # Standard deviation of the Gaussian distribution

    @tf.function
    def call(self, images, training=True):
        if training:
           # Generate random displacements for each pixel in the batch
            batch_size, height, width, channels = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]
            dx = tf.random.normal((batch_size, height, width), 0, self.sigma) * self.alpha
            dy = tf.random.normal((batch_size, height, width), 0, self.sigma) * self.alpha

            flow = tf.stack([dx, dy], axis=-1) # Combine displacements into flow vectors
            return tfa.image.dense_image_warp(images, flow)  # Apply dense image warp using TensorFlow Addons
        else:
            return images  # Return original images unchanged if not in training mode


# In[ ]:


# Define data augmentation pipeline with additional augmentations
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.2), # randomly rotate images up to 20% of the total degrees
    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2), # randomly zoom images up to 20% of the original size
    tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"), # randomly flip images horizontally and vertically
    tf.keras.layers.RandomContrast(factor=0.2), # randomly adjust the contrast of images
    tf.keras.layers.RandomBrightness(factor=0.2), # randomly adjust the brightness of images
    ElasticTransformation(alpha=34, sigma=4), # Apply Elastic Transformation augmentation
    tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 0.2)) # randomly adjust the hue of images
])


# In[ ]:


# Split the dataset using sklearn's train_test_split
# Assuming 'data' is your tf.data.Dataset

def split_dataset(data, test_size=0.15, val_size=0.15):
    data_size = data.cardinality().numpy()  # Get the total number of elements in the dataset
    test_size = int(test_size * data_size)  # Calculate the number of elements for test set
    val_size = int(val_size * (data_size - test_size))  # Calculate the number of elements for validation set
    train_size = data_size - test_size - val_size  # Calculate the number of elements for training set

    train_data = data.take(train_size)  # Take the first 'train_size' elements for training set
    val_data = data.skip(train_size).take(val_size)  # Skip the first 'train_size' elements and take the next 'val_size' elements for validation set
    test_data = data.skip(train_size + val_size)  # Skip the first 'train_size' and 'val_size' elements for test set

    return train_data, val_data, test_data  # Return the split datasets

train_data, val_data, test_data = split_dataset(data)  # Split the dataset

# Apply data augmentation only to the training set
train_data_aug = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)  # Apply data augmentation to the training set

# Print confirmation
print("Data augmentation applied to the training set!")


# In[ ]:


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode = "horizontal_and_vertical"), # randomly flip images horizontally and vertically
])


# In[ ]:


# Initialize empty lists to store augmented images and labels
X_new, y_new = [], []
max_total = 3000  # Maximum number of augmented images to generate
X_0, X_1, X_2 = X[y== 0], X[y == 1], X[y == 2]  # Separate images by class

for label, X_set in zip(label_map.keys(), [X_0, X_1, X_2]):  # Iterate over each class
  count = 0
  for img in X_set:
      if count > max_total - label_counts[label]:  # Check if augmentation limit is reached
          break
          break
      for i in range(2):
          augmented_img = data_augmentation(img)  # Apply data augmentation to the image
          X_new.append(augmented_img)  # Append the augmented image to the list
          y_new.append(label)
          count += 1

X_new, y_new = np.array(X_new), np.array(y_new)  # Convert lists to NumPy arrays
print(X_new.shape, y_new.shape)  # Print the shapes of the augmented images and labels


# In[ ]:


X = np.concatenate([X, X_new])  # Concatenate the original images with the augmented images
y = np.concatenate([y, y_new])  # Concatenate the original labels with the augmented labels

print(X.shape, y.shape)


# In[ ]:


print(pd.Series(y).value_counts())  # Print the distribution of augmented labels


# In[ ]:


# Min-Max Normalization

X = X / 255.0  # Normalize pixel values to [0, 1]

print(X.min(), X.max())


# Model Splittig

# In[ ]:


# Splitting the data into training, testing and validation sets

X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(X, y, test_size = 0.15,
                                                                            random_state = 3, stratify = y)  # Stratify the split based on the class labels

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, test_size = 0.15,
                                                                  random_state = 3, stratify = y_train_val)  # Stratify the split based on the class labels

print(X_train.shape, X_val.shape, X_test.shape)  # Print the shapes of the training, validation, and test sets

print(f"Total instances: {X.shape[0]}\nTrain instances: {X_train.shape[0]}           \nValidation instances: {X_val.shape[0]}\nTest instances: {X_test.shape[0]}")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_train, y_val, y_test are your labels for train, validation, and test sets
# Assuming label_map is a dictionary mapping numeric labels to class names

# Convert numeric labels to class names
y_train_names = [label_map[label] for label in y_train]
y_val_names = [label_map[label] for label in y_val]
y_test_names = [label_map[label] for label in y_test]

# Plot the class distribution
plt.figure(figsize=(18, 6))

# Train Set
plt.subplot(1, 3, 1)
plt.title("Train Set")
sns.countplot(x=y_train_names)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')  # Add count labels to each bar

# Validation Set
plt.subplot(1, 3, 2)
plt.title("Validation Set")
sns.countplot(x=y_val_names)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

# Test Set
plt.subplot(1, 3, 3)
plt.title("Test Set")
sns.countplot(x=y_test_names)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.tight_layout()
plt.show()


# In[ ]:


#implementing pie charts
plt.figure(figsize = (10, 10))

for i, labels, label_name in zip(range(3), [y_train, y_val, y_test], ["Train Set", "Validation Set", "Test Set"]):
    plt.subplot(3, 3, i + 1)
    plt.title(label_name)
    label_counts = pd.Series(labels).value_counts()
    plt.pie(label_counts, autopct = "%.2f%%", labels = label_counts.keys())
    plt.tight_layout()
plt.show()


# MODEL BULIDING

# In[ ]:


# Model building

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 200, kernel_size = (3, 3), input_shape = X_train.shape[1:], activation = "relu"),
    tf.keras.layers.MaxPooling2D(pool_size = (3, 3)),
    tf.keras.layers.Conv2D(filters = 100, kernel_size = (3, 3), input_shape = X_train.shape[1:], activation = "relu"),
    tf.keras.layers.MaxPooling2D(pool_size = (3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 100, activation = "relu"),
    tf.keras.layers.Dense(units = 50, activation = "relu"),
    tf.keras.layers.Dense(units = 3, activation = "softmax")
])


# In[ ]:


model.summary()  # Print a summary of the model architecture


# In[ ]:


for idx, layer in enumerate(model.layers):
    print(f"Layer {idx}:", layer.name, layer.output_shape, layer.count_params())  # Print layer details


# In[ ]:


#model architecture diagram
model_arch = tf.keras.utils.plot_model(model, show_shapes = True)
model_arch


# In[ ]:


# Model compilation

model.compile(optimizer = "adam", loss = "SparseCategoricalCrossentropy", metrics = ["accuracy"])  # Compile the model with the specified optimizer, loss function, and metrics


# In[ ]:


cb1 = tf.keras.callbacks.ModelCheckpoint("clf_model.h5", save_best_only = True)  # Save the best model based on validation accuracy
cb2 = tf.keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True)  # Stop training if validation loss does not improve for 5 epochs

history = model.fit(X_train, y_train, epochs = 100, callbacks = [cb1, cb2], validation_data = (X_val, y_val))  # Train the model with early stopping and model checkpointing


# In[ ]:


result_df = pd.DataFrame(history.history)  # Convert the history object to a DataFrame
result_df.head()


# In[ ]:


# @title Validation Loss and Accuracy

import matplotlib.pyplot as plt

plt.plot(result_df.index, result_df['val_loss'], label='val_loss')
plt.plot(result_df.index, result_df['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Validation Loss and Accuracy')
_ = plt.legend()


# In[ ]:


# @title Model Accuracy Over Time

import matplotlib.pyplot as plt

epochs = range(1, len(result_df['accuracy']) + 1)

plt.plot(epochs, result_df['accuracy'], label='Training Accuracy')
plt.plot(epochs, result_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
_ = plt.legend()


# In[ ]:


result_df.describe()  # Print summary statistics of the training results


# In[ ]:


model.summary()


# In[ ]:


import os
os.listdir()  # This will list all files in the current directory


# In[ ]:


# Learning curves

result_df.plot()
plt.show()


# In[ ]:


# Evaluating the model on train set

loss, acc = model.evaluate(X_train, y_train)


# In[ ]:


# Confusion Matrix

y_train_pred = model.predict(X_train)  # Make predictions on the training set
y_train_pred_labels = np.array([np.argmax(y_) for y_ in y_train_pred])  # Convert predictions to class labels

cm = tf.math.confusion_matrix(labels = y_train, predictions = y_train_pred_labels)  # Compute the confusion matrix
print(cm)


# In[ ]:


sns.heatmap(cm, annot = True)  # Plot the confusion matrix as a heatmap
plt.show()


# In[ ]:


# Calculating metrics for each class based on the confusion matrix (cm)
# for class 0
tp_0, tn_0 = cm[0][0], cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
fp_0, fn_0 = cm[1][0] + cm[2][0], cm[0][1] + cm[0][2]
# for class 1
tp_1, tn_1 = cm[1][1], cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
fp_1, fn_1 = cm[0][1] + cm[2][1], cm[1][0] + cm[1][2]
# for class 2
tp_2, tn_2 = cm[2][2], cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
fp_2, fn_2 = cm[0][2] + cm[1][2], cm[2][0] + cm[2][1]

def describeCM(tp, tn, fp, fn, i):
    print(f"\n\nClass {label_map[i]} : \n")
    print(f"True Positives : {tp}\nTrue Negatives : {tn}\nFalse Positives : {fp}\nFalse Negatives : {fn}")
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"\nPrecision : {precision}\nRecall (Sensitivity) : {recall}\nF1-Score : {f1}")

describeCM(tp_0, tn_0, fp_0, fn_0, 0)
describeCM(tp_1, tn_1, fp_1, fn_1, 1)
describeCM(tp_2, tn_2, fp_2, fn_2, 2)


# In[ ]:


# Classification Report

clf_report = metrics.classification_report(y_train, y_train_pred_labels)
print(clf_report)


# In[ ]:


model.save("clf_model_final.h5")  # Save the trained model to a file


# In[ ]:


# Evaluating the model on test set

loss, acc = model.evaluate(X_test, y_test)  # Evaluate the model on the test set
#print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")


# In[ ]:


# Confusion Matrix

y_test_pred = model.predict(X_test)
y_test_pred_labels = np.array([np.argmax(y_) for y_ in y_test_pred])

cm = tf.math.confusion_matrix(labels = y_test, predictions = y_test_pred_labels)

sns.heatmap(cm, annot = True)
plt.show()


# In[ ]:


# Calculating metrics for each class based on the confusion matrix (cm)
tp_0, tn_0 = cm[0][0], cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
fp_0, fn_0 = cm[1][0] + cm[2][0], cm[0][1] + cm[0][2]

tp_1, tn_1 = cm[1][1], cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
fp_1, fn_1 = cm[0][1] + cm[2][1], cm[1][0] + cm[1][2]

tp_2, tn_2 = cm[2][2], cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
fp_2, fn_2 = cm[0][2] + cm[1][2], cm[2][0] + cm[2][1]

def describeCM(tp, tn, fp, fn, i):
  print(f"\n\nClass {label_map[i]} : \n")
  print(f"True Positives : {tp}\nTrue Negatives : {tn}\nFalse Positives : {fp}\nFalse Negatives : {fn}")
  precision, recall = tp / (tp + fp), tp / (tp + fn)
  f1 = 2 * precision * recall / (precision + recall)
  print(f"\nPrecision : {precision}\nRecall (Sensitivity) : {recall}\nF1-Score : {f1}")

describeCM(tp_0, tn_0, fp_0, fn_0, 0)
describeCM(tp_1, tn_1, fp_1, fn_1, 1)
describeCM(tp_2, tn_2, fp_2, fn_2, 2)


# In[ ]:


# Classification Report

clf_report = metrics.classification_report(y_test, y_test_pred_labels)
print(clf_report)


# generating cam
# 

# In[ ]:


# List the contents of the extracted directory
extracted_files = os.listdir(extract_path)
print(extracted_files)


# In[ ]:


# Traverse the directory structure to find image files
image_files = []
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Adjust file extensions as needed
            image_files.append(os.path.join(root, file))

print("Image files found:")
for img_file in image_files:
    print(img_file)


# In[ ]:


import os

# Directory to search for images
search_dir = "/content/mri_images"

# List the contents of the directory
for root, dirs, files in os.walk(search_dir):
    print(f"Root: {root}")
    for file in files:
        print(f"File: {file}")


# In[ ]:


# Check if the model file exists
model_path = "clf_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No file or directory found at {model_path}")


# In[ ]:


# Selecting the first image path for CAM visualization
image_path = image_files[0]  # Selecting the first image path in the list

# Printing the selected image path
print("Selected image for CAM visualization:")
print(image_path)


# In[ ]:


# Load your trained model
model = tf.keras.models.load_model("clf_model.h5")

# Print model summary to see the layer names and types
print(model.summary())


# In[ ]:


# Update the image path based on the verified directory structure
image_path = "/content/mri_images/Axial/CI/CI068_S_0476a118.png"  # Example based on your provided structure

if not os.path.exists(image_path):
    raise FileNotFoundError(f"No file or directory found at {image_path}")


# In[ ]:


print(model.input_shape)


# In[ ]:


model.trainable_variables


# In[ ]:


def get_last_conv_layer_name(model):
  """
  Gets the name of the last convolutional layer in a TensorFlow model.

  Args:
      model: The TensorFlow model to inspect.

  Returns:
      str: The name of the last convolutional layer, or None if no convolutional layers are found.
  """

  last_conv_layer_name = None
  for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
      last_conv_layer_name = layer.name
  return last_conv_layer_name

# Example usage
last_conv_layer_name = get_last_conv_layer_name(model)

if last_conv_layer_name:
  print(f"Last convolutional layer name: {last_conv_layer_name}")
else:
  print("No convolutional layers found in the model.")


# In[ ]:


def is_layer_trainable(model, layer_name):
  """
  Checks if a specific layer in a TensorFlow model is trainable.

  Args:
      model: The TensorFlow model to inspect.
      layer_name: The name of the layer to check.

  Returns:
      bool: True if the layer is trainable, False otherwise.
  """

  try:
    layer = model.get_layer(layer_name)
    return layer.trainable
  except ValueError:
    print(f"Error: Layer '{layer_name}' not found in the model.")
    return False

# Example usage
layer_name = 'conv2d_3'
is_trainable = is_layer_trainable(model, layer_name)

if is_trainable:
  print(f"Layer '{layer_name}' is trainable.")
else:
  print(f"Layer '{layer_name}' is not trainable.")


# In[ ]:


get_ipython().system('pip install opencv-python')


# In[ ]:


# Example usage
import cv2

image_path = '/content/mri_images/Axial/CI/CI068_S_0476a118.png'  # Replace with your image path
img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
img_array = cv2.resize(img_array, (128, 128))  # Resize the image to 128x128
img_array = np.reshape(img_array, (1, 128, 128, 1))  # Reshape the image for the model input
img_array = img_array.astype('float32') / 255.0  # Normalize the pixel values


# In[ ]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_cam(model, img_array):
    """
    Generate a class activation map (CAM) for a given image and model.

    Args:
        model: The trained model.
        img_array: Input image array.

    Returns:
        cam_array: The class activation map for the given image.
    """
    # Ensure img_array has the shape (1, 128, 128, 1)
    if img_array.shape != (1, 128, 128, 1):
        img_array = np.reshape(img_array, (1, 128, 128, 1))

    print("Input image array shape:", img_array.shape)

    # Get the last convolutional layer
    last_conv_layer_name = 'conv2d_1'  # Replace with the actual last conv layer name if different
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError as e:
        print(f"Error: {e}. Check the last convolutional layer name.")
        return None  # Return None if layer not found

    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)

    # Create a model that maps the activations of the last conv layer to the final predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)

    # Compute the gradient of the top predicted class score with respect to the output feature map of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs = last_conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    print("conv_outputs shape:", conv_outputs.shape)
    print("predictions shape:", predictions.shape)
    print("class_idx:", class_idx.numpy())
    print("class_output shape:", class_output.shape)

    grads = tape.gradient(class_output, conv_outputs)

    if grads is None:
        print("Warning: Gradients are None. Check model and inputs.")
        return None

    print("Gradients shape:", grads.shape)

    # Compute the guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    print("Pooled gradients shape:", pooled_grads.shape)

    # Compute the weighted sum of the conv layer output channels
    conv_outputs = conv_outputs[0]
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(pooled_grads):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # Normalize the heatmap
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))

    print(f"CAM shape: {cam.shape}")

    return cam  # Return the calculated CAM array

# Define a dummy model and img_array for testing
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), name='conv2d_1'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
img_array = np.random.rand(1, 128, 128, 1)

# Generate CAM
cam_array = generate_cam(model, img_array)
if cam_array is None:
    print("Error: generate_cam function returned None. Check the implementation of generate_cam.")
else:
    # Plot original image and CAM overlay
    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.imshow(cam_array, cmap='jet', alpha=0.5)
    plt.show()


# In[ ]:


def plot_cam(image_path, cam_array, alpha=0.4):
    # Read the original image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Debugging: Check if the image is loaded
    if cam_array is None:
        print("Error: CAM generation failed. Check generate_cam function for errors.")
        return

    # Resize CAM to match the original image size
    cam_array_resized = cv2.resize(cam_array, (img.shape[1], img.shape[0]))
    # Apply colormap to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_array_resized), cv2.COLORMAP_JET)
    # Convert grayscale image to RGB for overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Overlay CAM on the original grayscale image
    superimposed_img = heatmap * alpha + img_rgb * (1 - alpha)
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)
    # Plot original image and CAM overlay
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('CAM Overlay')
    plt.axis('off')
    plt.show()


# In[ ]:


print(img_array.shape)


# In[ ]:


# Generate CAM
cam_array = generate_cam(model, img_array)


# In[ ]:


# Check if cam_array is None
if cam_array is None:
    print("Error: generate_cam function returned None. Check the implementation of generate_cam.")
else:
    # Plot original image and CAM overlay
    plot_cam(image_path, cam_array)


# In[ ]:


print("cam_array shape:", cam_array.shape)


# In[ ]:


model.summary()  # Print model summary to see the layer names and types


# In[ ]:


```

