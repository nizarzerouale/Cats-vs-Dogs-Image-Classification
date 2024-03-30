import cv2
import numpy as np
from tensorflow.keras.models import load_model

def loadImages(image_paths):
    images = []
    for image_path in image_paths:
        # Load the image
        img = cv2.imread(image_path)
        # Convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize it to 150x150
        img = cv2.resize(img, (150, 150))
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        images.append(img)
    # Convert the list to a numpy array and add an extra dimension
    images = np.array(images)
    return images

# Path to the saved model
model_path = 'Models_cats_dogs_small_dataset_pretrained.h5'
# Load the pre-trained model
model = load_model(model_path)

# Image paths
image_paths = ['test1.jpg', 'test2.jpg']
# Load and preprocess the images
images = loadImages(image_paths)

# Predict the categories of the images
predictions = model.predict(images)

# The model outputs a probability close to 1 for dogs and close to 0 for cats
for i, prediction in enumerate(predictions):
    if prediction < 0.5:
        print(f"{image_paths[i]} is a cat with probability {1 - prediction[0]}")
    else:
        print(f"{image_paths[i]} is a dog with probability {prediction[0]}")
