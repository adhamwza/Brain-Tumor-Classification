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


#Path for the glioma training data
##'\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 4 Classes\Training\glioma'
Glioma_Images = '\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Training\\glioma'

#Visiualization of 25 images of the glioma
for i in range(25):
    file=random.choice(os.listdir(Glioma_Images))
    food_image_path=os.path.join(Glioma_Images,file)
    img=mpimg.imread(food_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()

#Path for the meningioma training data
#'\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 4 Classes\Training\meningioma'
Meningioma_Images = '\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Training\\meningioma'

#Visiualization of 25 images of the meningioma
for i in range(25):
    file=random.choice(os.listdir(Meningioma_Images))
    building_image_path=os.path.join(Meningioma_Images,file)
    img=mpimg.imread(building_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()

#Path for the no tumor training data
#'\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 4 Classes\Training\notumor'
No_Tumor = '\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Training\\notumor'

#Visiualization of 25 images of the no tumor images
for i in range(25):
    file=random.choice(os.listdir(No_Tumor))
    landscape_image_path=os.path.join(No_Tumor,file)
    img=mpimg.imread(landscape_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()


#Path for the pituitary data images
#'\Users\adham\OneDrive\Desktop\Grad Proj Datasets\Datasets (Normal)\Brain Tumor 4 Classes\Training\pituitary'
Pituitary_Images = '\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Training\\pituitary'

#Visiualization of 25 images of the pituitary tumor images
for i in range(25):
    file=random.choice(os.listdir(Pituitary_Images))
    people_image_path=os.path.join(Pituitary_Images,file)
    img=mpimg.imread(people_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()


#Path for the full training folder
image_path='\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Training\\'


#Labels
label_name= ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


#determined image size
image_size=(224,224)

#Debugging (make sure the amount of classes are correct)
class_names = os.listdir(image_path)
print(class_names)
print("Number of classes : {}".format(len(class_names)))

numberof_images={}
for class_name in class_names:
    numberof_images[class_name]=len(os.listdir(image_path+"/"+class_name))
images_each_class=pd.DataFrame(numberof_images.values(),index=numberof_images.keys(),columns=["Number of images"])
print(images_each_class)



#Determining the training, validation, and testing datasets and their shuffling
train_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Training\\")
image_data_path=list(train_data_path.glob(r"**/*.jpg"))
train_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_train_data=pd.DataFrame({"image_data":image_data_path,"label":train_label_path}).astype("str")
final_train_data=final_train_data.sample(frac=1).reset_index(drop=True)
print(final_train_data['image_data'])

#Training should have majority of data: 70%
#Validation should have 15%, 20% validation images
#Testing  should have remaining 15% or 10%


valid_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Validation\\")
image_data_path=list(valid_data_path.glob(r"**/*.jpg"))
valid_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_valid_data=pd.DataFrame({"image_data":image_data_path,"label":valid_label_path}).astype("str")
final_valid_data=final_valid_data.sample(frac=1).reset_index(drop=True)
print(final_valid_data['image_data'])


#'\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Testing\\'
test_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Grad Proj Datasets\\Datasets (Normal)\\Brain Tumor 4 Classes\\Testing\\")
image_data_path=list(test_data_path.glob(r"**/*.jpg"))
test_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_test_data=pd.DataFrame({"image_data":image_data_path,"label":test_label_path}).astype("str")
final_test_data=final_test_data.sample(frac=1).reset_index(drop=True)
print(final_test_data['image_data'])



#Batch size
batch_size=64
#Training data generator that generates augmented images for the model
#to continue training on
#No augmentation necessary
traindata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                    #rotation_range = 40,
                                                                    #zoom_range=0.4,
                                                                    #width_shift_range=0.2,
                                                                    #height_shift_range=0.2,
                                                                    #shear_range=0.2,
                                                                    #horizontal_flip=True,
                                                                    #validation_split=0.2)
                                                                    #fill_mode='nearest')
)


#Data generator for both validation and testing
validdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



train_data_generator=traindata_generator.flow_from_dataframe(dataframe=final_train_data,
                                                             x_col="image_data",
                                                             y_col="label",
                                                             batch_size=batch_size,
                                                             class_mode="categorical",
                                                             target_size=(224,224),
                                                             color_mode="rgb",
                                                             shuffle=True )


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


#Model
#Contains 8 layers
model = keras.models.Sequential([
    #first layer:
    keras.layers.Flatten(input_shape=[224, 224, 3]),

    keras.layers.Dense(200, activation = 'relu'),

    keras.layers.Dense(170, activation = 'relu'),

    keras.layers.Dense(140, activation='relu'),

    keras.layers.Dense(110, activation = 'relu') ,

    keras.layers.Dense(80, activation = 'relu') ,

    keras.layers.Dense(50, activation = 'relu'),

    keras.layers.Dense(20, activation='relu'),

    keras.layers.Dense(4, activation = 'softmax')
])


model.summary()

# Define your learning rate scheduler
#Learning rate changes after 5 epochs by a certain rate
#Method found online
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Create a learning rate scheduler
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

#Compilation with initial learning rate
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00006),
              metrics = ['accuracy']
              )

#45, val loss 0.2
#45+, val loss 0.2 --> 0.6
#55 epochs from current lowest best vall loss epoch, val loss kept deteriorating, stop training, 
#epochs is how many times the model will train. in this case, 5 times, so he gets more accurate and loses less as the epochs go.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
model_history = model.fit(train_data_generator, epochs=1, validation_data=valid_data_generator, callbacks = [early_stopping])

#saving model
#model.save("CNN_Adam_Classification_Model.h5")

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


# Generate predictions on the test set
predictions = model.predict(test_data_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels
true_labels = test_data_generator.classes

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualize confusion matrix using seaborn
class_names = list(test_data_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Notes
#with data augmentation
#accuracy 72%, loss 0.66, train
#accuracy 72%, loss 0.59, validation
#accuracy 64%, loss 1.07, testing
#mediocre speed, accuracy improves over epochs, but needs
#an immense number of epochs and an immense amount of time to train
#surprisingly decent confusion matrix
#without data augmentation
#so far reaching 100% train acc with 0.0032 loss and validation reached 1.00 and loss of 0.0041 at epoch 28
#so far reaching 100% train acc with 0.000- loss and validation reached 1.00 and loss of 0.000-- epoch 46
#accuracy 100%, loss 0.000-, train
#accuracy 100%, loss 0.000-, validation
#accuracy 92%, loss 1.2, testing
#fast
#confusion matrix very good
#Based on the provided information, it's less likely that your model is severely overfitting. Here's why:

#Reasonable Testing Performance: The fact that your model achieved an accuracy of 92% on the testing data suggests that it's able to generalize reasonably well to unseen examples. Overfitting often leads to a significant drop in performance on unseen data compared to the training/validation data.

#Validation Performance: While perfect scores on the validation set can sometimes be indicative of overfitting, it's not necessarily the case here. If your model was heavily overfitting, it would likely perform much worse on the testing data. The fact that the performance drop from validation to testing is reasonable further supports this.

#Consistent Performance: The performance metrics (accuracy and loss) across training, validation, and testing data seem to be consistent and within a reasonable range, indicating that your model is not overly biased towards the training data.