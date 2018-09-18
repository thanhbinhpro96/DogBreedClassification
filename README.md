# DogBreedClassification

Binh Thanh Nguyen - CS2014 - Vietnamese-German University - Frankfurt University of Applied Sciences

This repository stores the code for my thesis project (submitted on September 17, 2018). For this repository, I split the instructions into 2 different sections: training and predicting.

First off, we have to install Python and get the necessary libraries: TensorFlow, Keras, WxPython.

## Training

We can initalize training by running:

```
python /fine_tune.py --images /dataset/stanford_dogs_cropped --model /output/dog_weights.hdf5 --classes 120
```

Where: 

- **fine_tune.py** is the training file 

- **/dataset/stanford_dogs_cropped** is the folder that contains the training images (must be in hierarchical order. For example: */dataset/stanford_dogs_cropped/German_shepherd/image1.jpg*) 

- **/output/dog_weights.hdf5** is the file the network will produce after the training process, this file will later be used for predicting on unseen data.

- **classes** is the number of dog breeds, in this case, it should be 120, but if you want to reduce the number of classes, feel free to delete the training data and adjust this parameter.

More settings can be adjusted in the *fine_tune.py* file, such as the training/validating split, data augmentation, the number of layers frozen,...

## Predicting

We will make predictions with a graphical user interface (GUI). To start off, please download the weights file from this link: [https://drive.google.com/open?id=1ek1o21jQq-WnKEaePpHi0BEbtAyFgOf_]. You can also use the weights file produced during the aforementioned training process for this task. Put the weights file into the same folder as the *gui.py* file. Then simply run:

```
python gui.py
```
To begin, choose *Browse* and select an image of a dog in your local machine. Afterwards, the results should appear after about 15 seconds.



The *Browse* button is now disabled. You can make another prediction by choosing *Reset* and choose *Browse* again.
