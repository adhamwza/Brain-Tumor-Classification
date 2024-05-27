#Imports
import tensorflow as tf
from tensorflow import keras
import random
#import cv2 as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout


Yes_Tumor = '\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\\Brain Tumor\\Training\\yes'

for i in range(25):
    file=random.choice(os.listdir(Yes_Tumor))
    food_image_path=os.path.join(Yes_Tumor,file)
    img=mpimg.imread(food_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()

No_Tumor = '\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\\Brain Tumor\\Training\\no'


for i in range(25):
    file=random.choice(os.listdir(No_Tumor))
    building_image_path=os.path.join(No_Tumor,file)
    img=mpimg.imread(building_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()



image_path='\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\\Brain Tumor\\Training\\'


#Label names
label_name= ['yes', 'no']



image_size=(224,224)



train_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\\Brain Tumor\\Training\\")
image_data_path=list(train_data_path.glob(r"**/*.jpg"))
train_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_train_data=pd.DataFrame({"image_data":image_data_path,"label":train_label_path}).astype("str")
final_train_data=final_train_data.sample(frac=1).reset_index(drop=True)
print(final_train_data['image_data'])


valid_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\\Brain Tumor\\Validation\\")
image_data_path=list(valid_data_path.glob(r"**/*.jpg"))
valid_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_valid_data=pd.DataFrame({"image_data":image_data_path,"label":valid_label_path}).astype("str")
final_valid_data=final_valid_data.sample(frac=1).reset_index(drop=True)
print(final_valid_data['image_data'])



test_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 2 Classes(1)\\Brain Tumor\\Testing\\")
image_data_path=list(test_data_path.glob(r"**/*.jpg"))
test_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_test_data=pd.DataFrame({"image_data":image_data_path,"label":test_label_path}).astype("str")
final_test_data=final_test_data.sample(frac=1).reset_index(drop=True)
print(final_test_data['image_data'])

def rgb_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)

batch_size=64

traindata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                    #rotation_range = 40,
                                                                    #zoom_range=0.4,
                                                                    #width_shift_range=0.2,
                                                                    #height_shift_range=0.2)
                                                                    #shear_range=0.2,
                                                                    #horizontal_flip=True,
                                                                    validation_split=0.2)
                                                                    #fill_mode='nearest')



validdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



train_data_generator=traindata_generator.flow_from_dataframe(dataframe=final_train_data,
                                                             x_col="image_data",
                                                             y_col="label",
                                                             batch_size=batch_size,
                                                             class_mode="categorical",
                                                             target_size=(224,224),
                                                             color_mode="rgb",
                                                             shuffle=True)


valid_data_generator=validdata_generator.flow_from_dataframe(dataframe=final_valid_data,
                                                             x_col="image_data",
                                                             y_col="label",
                                                             batch_size=batch_size,
                                                             class_mode="categorical",
                                                             target_size=(224,224),
                                                             color_mode="rgb",
                                                             shuffle=True )


test_data_generator=testdata_generator.flow_from_dataframe(dataframe=final_test_data,
                                                           x_col="image_data",
                                                           y_col="label",
                                                           batch_size=batch_size,
                                                           class_mode="categorical",
                                                           target_size=(224,224),
                                                           color_mode="rgb",
                                                           shuffle=False )

class_dict = train_data_generator.class_indices
class_list = list(class_dict.keys())
print(class_list)



model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[224, 224, 3]),
    keras.layers.Dense(200, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(170, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(140, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(110, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(80, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(20, activation='relu'),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()


def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00006),
              metrics=['accuracy']
              )


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


model_history = model.fit(
    train_data_generator,
    epochs=50,
    validation_data=valid_data_generator,
    callbacks=[early_stopping]
)

model.save("CNN_Adam_Detection_Model.h5")

test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test Accuracy: {test_accuracy}")


prediction= model.predict(test_data_generator)
prediction=np.argmax(prediction,axis=1)
map_label=dict((m,n) for n,m in (test_data_generator.class_indices).items())
final_predict=pd.Series(prediction).map(map_label).values
y_test=list(final_test_data.label)

plt.figure(figsize=(15, 15))
plt.style.use("classic")
number_images = (5, 5)
for i in range(1, (number_images[0] * number_images[1]) + 1):
    plt.subplot(number_images[0], number_images[1], i)
    plt.axis("off")

    color = "green"
    if final_test_data.label.iloc[i] != final_predict[i]:
        color = "red"
    plt.title(f"True:{final_test_data.label.iloc[i]}\nPredicted:{final_predict[i]}", color=color)
    plt.imshow(plt.imread(final_test_data['image_data'].iloc[i]))

plt.show()



predictions = model.predict(test_data_generator)
predicted_labels = np.argmax(predictions, axis=1)


true_labels = test_data_generator.classes


conf_matrix = confusion_matrix(true_labels, predicted_labels)


class_names = list(test_data_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Notes:
#Bad accuracy 53% w data augmentation,
#much higher without data augmentation
#94.4% acc test w no aug except rescaling, 99% train acc, 98.6% val acc.(Good confusion matrix)