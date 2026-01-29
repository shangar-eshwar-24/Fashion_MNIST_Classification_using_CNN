# Fashion_MNIST_Classification_using_CNN

This project demonstrates how to classify images from the Fashion MNIST dataset
using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

----------------
Overview
----------------
The notebook covers the complete workflow for image classification:
- Loading the Fashion MNIST dataset
- Preprocessing and normalizing image data
- Visualizing sample images
- Building a CNN model
- Training the model
- Evaluating performance on test data
- Visualizing training accuracy and loss
- Making predictions on new samples

----------------
Dataset
----------------
Fashion MNIST is a dataset of grayscale clothing images (28x28 pixels) with 10 categories:
0 - T-shirt/top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle boot

The dataset contains:
- 60,000 training images
- 10,000 test images

----------------
Requirements
----------------
Python libraries used:
- numpy
- pandas
- matplotlib
- tensorflow (keras)

----------------
How to Run
----------------
1) Install dependencies:
   pip install numpy pandas matplotlib tensorflow

2) Open and run the notebook:
   jupyter notebook fashion-mnist-classification-using-cnn.ipynb

----------------
Model
----------------
A CNN model is built using layers such as:
- Conv2D
- MaxPooling2D
- Flatten
- Dense

The model is compiled using:
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy

----------------
Training
----------------
The model is trained for multiple epochs with a validation split to monitor performance during training.
Training history is used to visualize accuracy and loss trends.

----------------
Evaluation
----------------
After training, the model is evaluated on the Fashion MNIST test dataset to report final test accuracy and loss.

----------------
Output
----------------
The notebook displays:
- Sample dataset images
- Training and validation accuracy plot
- Training and validation loss plot
- Model evaluation metrics
- Example prediction on a test image

----------------
Notes
----------------
CNN models generally perform better than traditional dense networks for image classification tasks.
Performance can be improved by tuning hyperparameters such as learning rate, number of filters,
batch size, and number of epochs.
