# Using Keras to tackle deep learning for small-data: augmenting organic chemistry structural images to train a convolutional neural network (CNN)
## Carly Wolfbrandt

### Table of Contents
1. [Introduction](#Introduction)
    1. [Convolutional Neural Networks (CNNs) using TensorFlow](#CNN) 
    2. [Small-data problem and image augmentation using Keras](#small-data) 
2. [Question](#Question)
3. [Model](#model)
    1. [Hyperparameters](#hp)
    2. [Training](#train)
    3. [Performance](#performance)
4. [Future Work](#future_work)

## Introduction <a name="Introduction"></a>

### Convolutional Neural Networks (CNNs) using TensorFlow <a name="CNN"></a>

### Small-data problem and image augmentation using Keras <a name="small-data"></a>

**Table 1**: 

| Common Name      | IUPAC Name |Molecular Formula | Skeletal Formula | 
| :-----------: | :-----------:| :-----------: | :----------:| 
| ethanol      |  ethanol | CH<sub>3</sub>CH<sub>2</sub>OH | ![](images/ethanol.png) |
| acetic acid   | ethanoic acid | CH<sub>3</sub>COOH  |![](images/acetic_acid.png)|
|cyclohexane | cyclohexane | C<sub>6</sub>H<sub>12</sub>| ![](images/cyclohexane.png)  |
| diphenylmethane | 1,1'-methylenedibenzene | (C<sub>6</sub>H<sub>5</sub>)<sub>2</sub>CH<sub>2</sub>|![](images/diphenylmethane.png)|

## Question <a name="Question"></a>

Can I build a model that can correctly classify organic chemistry molecules given the limitations of my current dataset that there is only one image per target?

## Model <a name="model"></a>

### Hyperparameters  <a name="hp"></a>

### Training <a name="train"></a>

### Performance <a name="performance"></a>

## Future Work <a name="future_work"></a>
