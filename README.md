# Cats vs Dogs Image Classification

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify images as either cats or dogs. It explores the effectiveness of various pre-trained models, including VGG16, Xception, InceptionV3, ResNet50, and MobileNet, leveraging the transfer learning technique to fine-tune these models on a dataset of cat and dog images.

## Project Structure

project/
│
├── data/ # Directory for storing the dataset
│ ├── train/ # Training data (cats and dogs images)
│ ├── validation/ # Validation data
│ └── test/ # Test data
│
├── models/ # Saved models and weights
│
├── src/ # Source code for the project
│ ├── prepareDatabase.py # Script to prepare the dataset
│ ├── imagePreprocessing.py # Script for image data preprocessing
│ ├── defineCNNModelFromScratch.py # Script to define a CNN model from scratch
│ ├── defineCNNModelVGGPretrained.py # Script to define and train models using pre-trained networks
│ └── main.py # Main script to run the project
│
├── README.md # Project overview and setup instructions
└── requirements.txt # List of project dependencies

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

Ensure you have Python 3.6+ installed on your machine. You will also need to install the necessary Python libraries:

```bash
pip install -r requirements.txt ```

## Dataset
The dataset used in this project consists of images of cats and dogs. Structure your dataset as described in the project structure section.You can download a suitable dataset from [Kaggle's Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats/data).
## Models
This project evaluates several pre-trained models for the task of image classification:

VGG16
Xception
InceptionV3
ResNet50
MobileNet
Each model is fine-tuned on the cats vs dogs dataset to compare their performance.

## Results
The accuracy and loss during training and validation are visualized and saved in the project directory. These results help in comparing the effectiveness of each pre-trained model for the classification task.
