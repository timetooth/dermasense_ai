import numpy as np
import pandas as pd
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import seaborn as sns

path_1 = 'data/short'
# path_2 = 'data/short'
meta_path = 'data/metaData.csv'
classes = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
encoder = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}
no_classes = len(classes)
meta = pd.read_csv(meta_path)
meta.head()

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    tf.keras.layers.RandomContrast(0.8),
])


images_data = []
labels = []
missed = []

for i in range(len(meta)):
    row = meta.iloc[i]
    image_name = row['image_id']
    image_class = row['dx']
    
    image_path = os.path.join(path_1, image_name + '.jpg')
    
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image) / 255.0
    except Exception as e:
        missed.append((image_name, str(e)))
        continue
    
    images_data.append(image)
    labels.append(encoder[image_class])

    # Data augmentation
    if image_class != 'nv':
        for _ in range(3):
            augmented_image = data_augmentation(np.expand_dims(image, 0))
            images_data.append(augmented_image[0])
            labels.append(encoder[image_class])

# Convert lists to NumPy arrays
images_data = np.array(images_data)
labels = np.array(labels)

print(f"Processed {len(images_data)} images with {len(missed)} misses.")

# Optionally, split into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.2, random_state=42)

# Create tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(no_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

y_train = tf.keras.utils.to_categorical(y_train, no_classes)
y_test = tf.keras.utils.to_categorical(y_test, no_classes)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test),
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

model.save(f'diff_weights/custom_model2_{test_accuracy:.3f}_.h5')