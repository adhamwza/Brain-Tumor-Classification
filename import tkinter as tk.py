import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

#2.15.0 tensorflow
#2.15.0 keras
# Placeholder function to simulate running a script on the selected image
def run_script_on_image(script_name, image_path):
    if(script_name == "detection_script"):
        preprocess_image(image_path)
        class_names = ['No Tumor', 'Tumor']
        predicted_class, prediction = predict_image_detection(image_path)
        predicted_class_name = class_names[predicted_class]
        prediction_str = ''.join(map(str, prediction))
        print(predict_image_detection(image_path))
        messagebox.showinfo("Detection", ("Predicted Class:", predicted_class_name, '\n', "Prediction", prediction_str))

    elif(script_name == "classification_script"):
        preprocess_image(image_path)
        class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        predicted_class, prediction = predict_image_classification(image_path)
        predicted_class_name = class_names[predicted_class]
        prediction_str = f'Class: {predicted_class_name}\nProbability: {prediction}'
        print(predict_image_classification(image_path))
        messagebox.showinfo("Classification", (prediction_str))

    elif(script_name == "benign_vs_malignant_script"):
        preprocess_image(image_path)
        class_names = ['Benign Tumor', 'Malignant Tumor']
        predicted_class, prediction = predict_image_benignVSmalignant(image_path)
        predicted_class_name = class_names[predicted_class]
        prediction_str = f'Class: {predicted_class_name}\nProbability: {prediction}'
        print(predict_image_benignVSmalignant(image_path))
        messagebox.showinfo("BenignVSMalignant", (prediction_str))
        print(f"Executing {script_name} on image: {image_path}")

    elif(script_name == "chest_cancer_script"):
        preprocess_image(image_path)
        class_names = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 'normal', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']
        predicted_class, prediction = predict_image_chestcancer(image_path)
        predicted_class_name = class_names[predicted_class]
        prediction_str = f'Class: {predicted_class_name}\nProbability: {prediction}'
        print(predict_image_chestcancer(image_path))
        messagebox.showinfo("Chest Cancer   ", (prediction_str))
        print(f"Executing {script_name} on image: {image_path}")

    elif(script_name == "ALL_cancer_script"):
        preprocess_image(image_path)
        class_names = ['Benign', 'Early', 'Pre-B', 'Pro-B']
        predicted_class, prediction = predict_image_ALL(image_path)
        predicted_class_name = class_names[predicted_class]
        prediction_str = f'Class: {predicted_class_name}\nProbability: {prediction}'
        print(predict_image_ALL(image_path))
        messagebox.showinfo("ALL Cancer   ", (prediction_str))
    print(f"Executing {script_name} on image: {image_path}")

# Function to handle button click events
def handle_button_click(script_name):
    # Prompt user to select an image file
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        run_script_on_image(script_name, image_path)

# Create the main application window
root = tk.Tk()
root.title("Tumor Analysis")

# Create buttons for each analysis type
detection_button = tk.Button(root, text="Tumor Detection", command=lambda: handle_button_click("detection_script"))
detection_button.pack(pady=5)

classification_button = tk.Button(root, text="Tumor Classification", command=lambda: handle_button_click("classification_script"))
classification_button.pack(pady=5)

benign_malignant_button = tk.Button(root, text="Benign vs Malignant", command=lambda: handle_button_click("benign_vs_malignant_script"))
benign_malignant_button.pack(pady=5)

Chest_Cancer_button = tk.Button(root, text="Chest Cancer", command=lambda: handle_button_click("chest_cancer_script"))
Chest_Cancer_button.pack(pady=5)

ALL_button = tk.Button(root, text="ALL Cancer", command=lambda: handle_button_click("ALL_cancer_script"))
ALL_button.pack(pady=5)


#To use model:

# Load the trained model
model = tf.keras.models.load_model("VGG_Detection_Model.h5")
model1 = tf.keras.models.load_model("CNN_Adam_Classification_Model.h5")
model2 = tf.keras.models.load_model("MobileNet_BenignVSMalignant_Model.h5")
model3 = tf.keras.models.load_model("Chest_Cancer_MobileNet_Classification.h5")
model4 = tf.keras.models.load_model("ALL_MobileNet_Classification_Model.h5")

# Function to preprocess a single image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions on a single image
def predict_image_detection(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

def predict_image_classification(image_path):
    img_array = preprocess_image(image_path)
    prediction = model1.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

def predict_image_benignVSmalignant(image_path):
    img_array = preprocess_image(image_path)
    prediction = model2.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

def predict_image_chestcancer(image_path):
    img_array = preprocess_image(image_path)
    prediction = model3.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

def predict_image_ALL(image_path):
    img_array = preprocess_image(image_path)
    prediction = model4.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

# Start the Tkinter event loop
root.mainloop()
