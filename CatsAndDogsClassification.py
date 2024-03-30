import tensorflow 
from tensorflow.keras import layers
from tensorflow.keras import models
#from tensorflow.keras.optimizers import rmsprop_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,ResNet50,InceptionV3,Xception,MobileNet
from tensorflow.keras.optimizers import RMSprop
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16

import os
import shutil
from glob import glob

#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def visualizeTheTrainingPerformances(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    pyplot.title('Training and validation accuracy')
    pyplot.plot(epochs, acc, 'bo', label = 'Training accuracy')
    pyplot.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
    pyplot.legend()
    pyplot.savefig('Accuracy '+model_name+'.png')
    pyplot.figure()
    pyplot.title('Training and validation loss')
    pyplot.plot(epochs, loss, 'bo', label = 'Training loss')
    pyplot.plot(epochs, val_loss, 'b', label = 'Validation loss')
    pyplot.legend
    pyplot.savefig('Loss '+model_name+'.png')
    
    pyplot.show()
    
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def prepareDatabase(original_directory, base_directory):

    #If the folder already exist remove everything
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)

    #Recreate the basefolder
    os.mkdir(base_directory)
    
    #TODO - Application 1 - Step 1a - Create the training folder in the base directory
    train_directory = os.path.join(base_directory, 'train')
    os.mkdir(train_directory)
    
    #TODO - Application 1 - Step 1b - Create the validation folder in the base directory
    validation_directory = os.path.join(base_directory, 'validation')
    os.mkdir(validation_directory)

    #TODO - Application 1 - Step 1c - Create the test folder in the base directory
    test_directory = os.path.join(base_directory, 'test')
    os.mkdir(test_directory)
    
    #TODO - Application 1 - Step 1d - Create the cat/dog training/validation/testing directories - See figure 4

    # create the train_cats_directory
    train_cats_directory = os.path.join(train_directory, 'cats')
    os.mkdir(train_cats_directory)
    
    # create the train_dogs_directory
    train_dogs_directory = os.path.join(train_directory, 'dogs')
    os.mkdir(train_dogs_directory)

    # create the validation_cats_directory
    validation_cats_directory = os.path.join(validation_directory, 'cats')
    os.mkdir(validation_cats_directory)

    # create the validation_dogs_directory
    validation_dogs_directory = os.path.join(validation_directory, 'dogs')
    os.mkdir(validation_dogs_directory)

    # create the test_cats_directory
    test_cats_directory = os.path.join(test_directory, 'cats')
    os.mkdir(test_cats_directory)

    # create the test_dogs_directory
    test_dogs_directory = os.path.join(test_directory, 'dogs')
    os.mkdir(test_dogs_directory)

    #TODO - Application 1 - Step 1e - Copy the first 1000 cat images into the training directory (train_cats_directory)
    original_directory_cats = str(original_directory + '/cats/') 
    fnames = ['{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_directory_cats, fname)
        dst = os.path.join(train_cats_directory, fname)
        shutil.copyfile(src, dst)

    #TODO - Application 1 - Step 1f - Copy the next 500 cat images into the validation directory (validation_cats_directory)
    original_directory_cats = str(original_directory + '/cats/') 
    fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_directory_cats, fname)
        dst = os.path.join(validation_cats_directory, fname)
        shutil.copyfile(src, dst)

    #TODO - Application 1 - Step 1g  - Copy the next 500 cat images in to the test directory (test_cats_directory)
    original_directory_cats = str(original_directory + '/cats/') 
    fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_directory_cats, fname)
        dst = os.path.join(test_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Step 1h - Copy the first 1000 dogs images into the training directory (train_dogs_directory)
    original_directory_dogs = str(original_directory + '/dogs/') 
    fnames = ['{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_directory_dogs, fname)
        dst = os.path.join(train_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Step 1i - Copy the next 500 dogs images into the validation directory (validation_dogs_directory)
    original_directory_dogs = str(original_directory + '/dogs/') 
    fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_directory_dogs, fname)
        dst = os.path.join(validation_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Step 1j  - Copy the next 500 dogs images in to the test directory (test_dogs_directory)
    original_directory_dogs = str(original_directory + '/dogs/') 
    fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_directory_dogs, fname)
        dst = os.path.join(test_dogs_directory, fname)
        shutil.copyfile(src, dst)

    #TODO - Application 1 - Step 1k - As a sanitary check verify how many pictures are in each directory
    print('Total number of CATS used for training = {}'.format(len(os.listdir(train_cats_directory)))) 
    print('Total number of CATS used for validation = {}'.format(len(os.listdir(validation_cats_directory)))) 
    print('Total number of CATS used for testing = {}'.format(len(os.listdir(test_cats_directory))))
    print('Total number of DOGS used for training = {}'.format(len(os.listdir(train_dogs_directory)))) 
    print('Total number of DOGS used for validation = {}'.format(len(os.listdir(validation_dogs_directory)))) 
    print('Total number of DOGS used for testing = {}'.format(len(os.listdir(test_dogs_directory))))

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineCNNModelFromScratch():

    #Application 1 - Step 3a - Initialize the sequential model
    model = models.Sequential()

    #TODO - Application 1 - Step 3b - Create the first hidden layer as a convolutional layer
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))

    #TODO - Application 1 - Step 3c - Define a maxpooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 1 - Step 3d - Create the third hidden layer as a convolutional layer
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    
    #TODO - Application 1 - Step 3e - Define a pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 1 - Step 3f - Create another convolutional layer
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

    #TODO - Application 1 - Step 3g - Define a pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 1 - Step 3h - Create another convolutional layer
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

    #TODO - Application 1 - Step 3i - Define a pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 1 - Step 3j - Define the flatten layer
    model.add(layers.Flatten())
    
    # Define the dropout layer
    model.add(layers.Dropout(rate=0.5))

    #TODO - Application 1 - Step 3k - Define a dense layer of size 512
    model.add(layers.Dense(512, activation='relu'))

    #TODO - Application 1 - Step 3l - Define the output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    #TODO - Application 1 - Step 3m - Visualize the network arhitecture (list of layers)
    model.summary()

    #TODO - Application 1 - Step 3n - Compile the model
    model.compile(optimizer=tensorflow.keras.optimizers.legacy.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineCNNModelVGGPretrained():

    #TODO - Application 2 - Step 1 - Load the pretrained VGG16 network in a variable called baseModel
    #The top layers will be omitted; The input_shape will be kept to (150, 150, 3)


    #TODO - Application 2 - Step 2 -  Visualize the network arhitecture (list of layers)


    #TODO - Application 2 - Step 3 -  Freeze the baseModel convolutional layers in order not to allow training


    #TODO - Application 2 - Step 4 - Create the final model and add the layers from the baseModel
    VGG_model = models.Sequential()
    #VGG_model.add(baseModel)            #Uncomment this


    # TODO - Application 2 - Step 4a - Add the flatten layer


    # TODO - Application 2 - Step 4b - Add the dropout layer


    # TODO Application 2 - Step 4c - Add a dense layer of size 512


    # TODO - Application 2 - Step 4d - Add the output layer


    # TODO - Application 2 - Step 4e - Compile the model


    return VGG_model
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
def imagePreprocessing(base_directory):
    train_directory = base_directory + '/train'
    validation_directory = base_directory + '/validation'
    test_directory = base_directory + '/test'

    #TODO - Application 1 - Step 2 - Create the image data generators for train, validation and test
    train_datagen = ImageDataGenerator(rescale=1./255)
    #train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(train_directory, target_size = (150, 150), batch_size = 20, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_directory,target_size = (150, 150), batch_size = 20, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_directory, target_size = (150, 150), batch_size = 20, class_mode='binary')

    #TODO - Application 1 - Step 2 - Analyze the output of the train and validation generators
    for data_batch, labels_batch in train_generator:
        print('Data batch shape in train: ', data_batch.shape) 
        print('Labels batch shape in train: ', labels_batch.shape) 
        break
    for data_batch, labels_batch in validation_generator:
        print('Data batch shape in validation: ', data_batch.shape) 
        print('Labels batch shape in validation: ', labels_batch.shape) 
        break
    for data_batch, labels_batch in test_generator:
        print('Data batch shape in test: ', data_batch.shape) 
        print('Labels batch shape in test: ', labels_batch.shape) 
        break
    return train_generator, validation_generator, test_generator
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
def main():
    original_directory = "./Kaggle_Cats_And_Dogs_Dataset"
    base_directory = "./Kaggle_Cats_And_Dogs_Dataset_Small"

    #TODO - Application 1 - Step 1 - Prepare the dataset
    #prepareDatabase(original_directory, base_directory)

    #TODO - Application 1 - Step 2 - Call the imagePreprocessing method
    train_generator, validation_generator, test_generator=imagePreprocessing(base_directory)

    #TODO - Application 1 - Step 3 - Call the method that creates the CNN model
    model=defineCNNModelFromScratch()
    #model = defineCNNModelVGGPretained()

    #TODO - Application 1 - Step 4 - Train the model
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
    
    #TODO - Application 1 - Step 5 - Visualize the system performance using the diagnostic curves
    visualizeTheTrainingPerformances(history, 'modelfromscratch')

    model.evaluate(test_generator)
    model.save('Models_cats_dogs_small_dataset_pretrained.h5')
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
