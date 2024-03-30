import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, Xception, InceptionV3, ResNet50, MobileNet
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam

# Define a dictionary of pre-trained models to evaluate
pretrained_models = {
    'VGG16': VGG16,
    'Xception': Xception,  # Requires input images to be 299x299
    'InceptionV3': InceptionV3,  # Requires input images to be 299x299
    'ResNet50': ResNet50,
    'MobileNet': MobileNet
}

# Assuming you have your data directories set up
train_dir = 'path/to/train_dir'
validation_dir = 'path/to/validation_dir'

# Image dimensions expected by Xception and InceptionV3
xception_inception_size = (299, 299)

# Standard size for other models
standard_size = (150, 150)

# Define a function to create and train a model
def create_and_train_model(model_name, base_model_function, input_size):
    # Load the pre-trained base model
    base_model = base_model_function(weights='imagenet', include_top=False, input_shape=input_size + (3,))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create a new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Create data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=20,
        class_mode='binary')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=input_size,
        batch_size=20,
        class_mode='binary')
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # Adjust based on your dataset
        epochs=10,  # Adjust based on your needs
        validation_data=validation_generator,
        validation_steps=50)  # Adjust based on your dataset
    
    # Return the trained model and its history
    return model, history

# Iterate over each pre-trained model and train
for model_name, model_function in pretrained_models.items():
    print(f"Training with {model_name}...")
    if model_name in ['Xception', 'InceptionV3']:
        model, history = create_and_train_model(model_name, model_function, xception_inception_size)
    else:
        model, history = create_and_train_model(model_name, model_function, standard_size)
    
    # Evaluate model performance (customize evaluation based on your needs)
    accuracy = history.history['val_accuracy'][-1]  # Last validation accuracy
    print(f"{model_name} Validation Accuracy: {accuracy:.4f}\n")
