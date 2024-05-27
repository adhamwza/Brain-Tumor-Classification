import tensorflow as tf
import keras
from tensorflow import *
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications import MobileNetV2
from keras.utils import to_categorical
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.image as mpimg


#2 CLASSES DATASET



#Loads image path, extracts labels,
#creates dataframe.
def load_data(data_path):
    image_data_path = list(data_path.glob("**/*.jpg"))
    label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
    final_data = pd.DataFrame({"image_data": image_data_path, "label": label_path}).astype("str")
    final_data = final_data.sample(frac=1).reset_index(drop=True)
    return final_data


# Load and preprocess images
# Resize all images and changes their format
#+views some images from each folder to make sure
#data is loaded properly
def view_preprocess(directory, num_samples=25):
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        file = random.choice(os.listdir(directory))
        image_path = os.path.join(directory, file)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        i_value = int(i.numpy())
        ax = plt.subplot(5, 5, i_value + 1)
        plt.imshow(img_array.astype("uint8"))
        plt.axis("off")
    plt.show()


# Paths
#"C:\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\Brain Tumor\Training\yes"
#"C:\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\Brain Tumor\Training\no"
yes_tumor = r'\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\Brain Tumor\Training\yes'
no_tumor = r'\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\Brain Tumor\Training\no'


# Load and preprocess images for each class
view_preprocess(yes_tumor)
view_preprocess(no_tumor)

# Data paths
image_path = r'\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 2 Classes(1)\\Brain Tumor\\Training\\'
train_data_path = Path(image_path)
valid_data_path = Path(r'\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 2 Classes(1)\\Brain Tumor\\Validation\\')
test_data_path = Path(r'\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 2 Classes(1)\\Brain Tumor\\Testing\\')


# Load and preprocess data
final_train_data = load_data(train_data_path)
final_valid_data = load_data(valid_data_path)
final_test_data = load_data(test_data_path)


#Batch
batch_size = 30

#Train Data Augmentation Generator
traindata_generator = ImageDataGenerator(
    rescale=1./ 255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)

# Validation Data Generator
validdata_generator = ImageDataGenerator(rescale=1. / 255)
testdata_generator = ImageDataGenerator(rescale=1. / 255)

# Model parameters
train_data_generator = traindata_generator.flow_from_dataframe(
    dataframe=final_train_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True
)

valid_data_generator = validdata_generator.flow_from_dataframe(
    dataframe=final_valid_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True
)

test_data_generator = testdata_generator.flow_from_dataframe(
    dataframe=final_test_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = keras.models.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(input_shape = (224,224,3)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)


# Training
#And training times (Epoch amount)
model_history = model.fit(train_data_generator, epochs=50, validation_data=valid_data_generator)

model.save("MobileNet_Detection_Model.h5")

# Evaluation
test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test Accuracy: {test_accuracy}")

# Predictions and visualization of only 25 images with ground truth
prediction= model.predict(test_data_generator)
prediction=np.argmax(prediction,axis=1)
map_label=dict((m,n) for n,m in test_data_generator.class_indices.items())
final_predict=pd.Series(prediction).map(map_label).values
y_test=list(final_test_data.label)


plt.figure(figsize=(15, 15))
plt.style.use("classic")
number_images = (5, 5)
for i in range(1, (number_images[0] * number_images[1]) + 1):
    plt.subplot(number_images[0], number_images[1], int(i))  # Convert i to integer
    plt.axis("off")

    i = i.numpy() if isinstance(i, tf.Tensor) else i  # Convert to NumPy array if it's a TensorFlow tensor

    color = "green"
    if final_test_data.label.iloc[int(i)] != final_predict[int(i)]:
        color = "red"
    plt.title(f"True:{final_test_data.label.iloc[int(i)]}\nPredicted:{final_predict[int(i)]}", color=color)
    plt.imshow(plt.imread(final_test_data['image_data'].iloc[int(i)]))

plt.show()

#Notes
#99.4 training acc, 0.016 loss, 98.8 val accuracy, 0.050 val loss, and 99.5% test acc