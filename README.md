# Using Keras to tackle deep learning for small-data: augmenting organic chemistry structural images to train a convolutional neural network (CNN)
## Carly Wolfbrandt

### Table of Contents
1. [Introduction](#Introduction)
    1. [Small-data problem and image augmentation using Keras](#small-data) 
2. [Question](#Question)
3. [Model](#model)
    1. [Hyperparameters](#hp)
    2. [Training](#train)
    3. [Performance](#performance)
4. [Future Work](#future_work)

## Introduction <a name="Introduction"></a>

The skeletal formula of a chemical species is a type of molecular structural formula that serves as a shorthand representation of a molecule's bonding and contains some information about its molecular geometry. It is represented in two dimensions, and is used as a shorthand representation for sketching species or reactions. This shorthand representation is particularly useful in that carbons and hydrogens, the two most common atoms in organic chemistry, don't need to be explicitly drawn.

Each structure conveys unique information about elements and bonding orientation in a chemical species. Since the structures are unique, this means that there is only one correct way to represent every chemical species. This presents an interesting problem when trying to train a neural network to predict the name of a structure - by convention the datasets are going to be sparse. The [hydrocarbon dataset](https://github.com/cwolfbrandt/csk_database/edit/master/README.md) has 2,135 rows, each with a unique name and 300 x 300 pixel structural image.

### Small-data problem and image augmentation using Keras <a name="small-data"></a>

There has been a recent explosion in research of modeling methods geared towards "big-data." Certainly, data science as a discipline has an obsession with big-data, as specialty methods are required to effectively analyze large datasets. However, an often overlooked problem in data science is small-data. It is generally (and perhaps incorrectly) believed that deep-learning is only applicable to big-data. 

It is true that deep-learning does usually require large amounts of training data in order to learn high-dimensional features of input samples. However, convolutional neural networks are one of the best models available for image classification, even when they have very little data from which to learn.

In order to make the most of the small dataset, more images must be generated. In Keras this can be done via the `keras.preprocessing.image.ImageDataGenerator` class. This method is used to augment each image, generating a a new image that has been randomly transformed. This ensures that the model should never see the exact same picture twice, which helps prevent overfitting and helps the model generalize better.

**Table 1**: 

| Common Name      | IUPAC Name |Molecular Formula | Skeletal Formula | 
| :-----------: | :-----------:| :-----------: | :----------:| 
| coronene      |  coronene | C<sub>24</sub>H<sub>12</sub> | ![](images/494155.png) |
| biphenylene  | biphenylene | C<sub>12</sub>H<sub>8</sub> |![](images/497397.png)|
|1-Phenylpropene | [(E)-prop-1-enyl]benzene | C<sub>9</sub>H<sub>10</sub>| ![](images/478708.png)  |

## Question <a name="Question"></a>

Can I build a model that can correctly classify organic chemistry molecules given that my current dataset has only one image per target?

## Model <a name="model"></a>

### Hyperparameters  <a name="hp"></a>

### Training <a name="train"></a>

### Performance <a name="performance"></a>

## Future Work <a name="future_work"></a>
