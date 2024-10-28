# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
~~~
The goal of this project is to develop a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. Handwritten digit classification is a fundamental task in image processing and machine learning, with various applications such as postal code recognition, bank check processing, and optical character recognition systems.

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), totaling 60,000 training images and 10,000 test images. The challenge is to train a deep learning model that accurately classifies the images into the corresponding digits.
~~~
## Neural Network Model

![image](https://github.com/user-attachments/assets/3d5d5090-039f-4cdc-8bf8-f9bc3f1dc872)


## DESIGN STEPS

### Step 1:
Load the MNIST dataset, which contains images of handwritten digits and corresponding labels.

### Step 2:
Reshape and normalize the images by scaling the pixel values between 0 and 1.

### Step 3:
Define a CNN model using layers such as convolution, pooling, and fully connected layers. Add a softmax output layer for classification.

### Step 4:
Compile the model using an appropriate optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and evaluation metric (accuracy).

### Step 5:
Train the model on the training dataset for a fixed number of epochs (e.g., 10 epochs) while monitoring the validation accuracy using early stopping to halt training once a desired accuracy is achieved.

### Step 6:
Stop training if the model reaches 98% accuracy on the training set to prevent overfitting and save computation time.

## PROGRAM

### Name:S DHANUSH PRABOO
### Register Number:212221230019
~~~
Importing Libraries
import os
import base64
import numpy as np
import tensorflow as tf
Loading and Inspecting the MNIST dataset

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "/content/mnist.npz.zip")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
Resizing and Reshaping Function
FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = np.expand_dims(images, axis=-1)

    # Normalize pixel values
    images = images/255.0

    ### END CODE HERE ###

    return images
# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name: S Dhanush Praboo            RegisterNumber: 212221230019        \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")
CallBack Function

# GRADED CLASS: EarlyStoppingCallback

### START CODE HERE ###

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self,epoch, logs={}):

        # Check if the accuracy is greater or equal to 0.98
        if (logs.get('accuracy')>=0.995):

            # Stop training once the above condition is met
            self.model.stop_training=True
            print("\nReached 98% accuracy so cancelling training!")
print('Name: S DHANUSH PRABOO          Register Number: 212221230019         \n')
### END CODE HERE ###
GRADED FUNCTION: convolutional_model
Network Model
def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([

    # Add convolutions and max pooling
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
    return model
Model Summary:
model.summary()
### Model compiling and Training

# Define your compiled (but untrained) model
model = convolutional_model()

# Train your model (this can take up to 5 minutes)
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])
~~~

## OUTPUT

### Reshape and Normalize output

![Screenshot 2024-10-28 105429](https://github.com/user-attachments/assets/480d89d8-688f-4510-ba79-937af697a743)


### Training the model output

![Screenshot 2024-10-28 105446](https://github.com/user-attachments/assets/b91e5133-3b2c-433e-bc50-29346b13abc3)

![Screenshot 2024-10-28 110051](https://github.com/user-attachments/assets/067ce851-feb8-44e3-8619-4a6314a9e8a7)



## RESULT

Thus the program to create a Convolution Neural Network to classify images is successfully implemented.
Include your result here.
