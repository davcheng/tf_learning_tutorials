# CNN Overview
Source:
https://www.tensorflow.org/versions/master/tutorials/layers
http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

Convolutional Neural Networks (CNNs) are
despite the unreasonably intimidating name, CNNs are one of the easier architectures to understand. The Convolution part of the name comes from partial differential equations CNNs work like this:



state of the art for image recognition tasks - learn higher level features

Three core components:
Convolutional layer:
labeling input signals (parts of an input) based on what it has seen in the past.
For example, if you show a picture of a cat, and the model sub-patterns within the greater context of a cat (e.g., whiskers, pointy ears, cute upside-down triangular nose, etc.). It doesn't matter where the whiskers (or whatever pattern it recognizes) are in the image, it just identifies that there are whiskers present.

Pooling layers:


Fully (dense) Connected layer:
