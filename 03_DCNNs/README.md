# Deep Neural Networks

## Core Examples

### CAB420_DCNNs_Example_1_Classification_with_Deep_Learning.ipynb

In this example we will:

*  Train a deep learning classifier to classify images
*  Explore different layers, include fully connected and convoluational layers, and different activation functions

Using this example as a starting point, try the following:

*  Change the network. Start by making it simpler (remove some layers, reduce the number of filters, etc) and see what happens. Then make it more complex (add layers, increase the number of filters, etc) and see what happens. Consider the change in accuracy, the number of parameters, and the time taken to train the network.
*  Change the training parameters. Try varying the batch size and the number of epochs. What happens when you increase the batch size, yet leave the number of epochs constant? What about decreasing the batch size?
*  Try training the network with much less data (I mean much less, throw away 90% of the data or more). Consider what you need to do with respect to batch size and the number of epochs when the amount of data is reduced.
*  Add layers such as batch normalisation or dropout to the network. How do these impact training speed and overfitting?

### CAB420_DCNNs_Example_2_Regression_with_Deep_Learning.ipynb

In this example we will:

*  Train a deep network for regression
*  Look at the difference between a regression task and classification task in terms of network structure
*  Use a different activation function (swish) because we can

Using this example as a starting point, try all those things that were suggested to try with the first example (change the network, add batch normalisation, etc), just to see what happens.

### CAB420_DCNNs_Example_3_ResNet.ipynb

Neural networks can be put together in lots of different ways. The topology of the network controls how information flows through the networks - and how gradients flow backwards. This example will look at ResNet style networks, which use residual blocks and allows for deeper, yet smaller (in terms of number of parameters), networks.

### CAB420_DCNNs_Example_4_Fine_Tuning_and_Data_Augmentation.ipynb

Deep networks need a lot of data to train. What can you do when you don't have much? This example will look at two approaches:

*  Fine Tuning, where we adapt one network to some new data
*  Data Augmentation, where we create addition data by subtling varying what we have

Using this example as a starting point, try the following:

*  Fine-tune a different network. Load another one of the pre-trained networks (pick a different type of network) and fine-tune this network. Experiment with freezing different numbers of layers and see what impact this has on training time and final performance.
*  Explore the extremes of data augmentation. How much data augmentation is too much? How little is too little? Train a network with no augmentation as a baseline and then gradually change the type and amount of augmentation (visualising as you go) and observe the impact.

You're also encouraged to try playing with training parameters. Try changing optimisers, changing learning rates, adding weight decay, and see what happens. You can do this either with fine-tuned networks, or when training from scratch, but experimenting with these options is highly recommended.

***

## Additional Examples

### CAB420_DCNNs_Additional_Example_1_Images_Introduction.ipynb

This example concerns images, and manipulating these in python. If you're already familar with how images are stored and can be manipulated, you can probably skip this. This example will look at some different image manipulation details such as:

*  Loading and displaying images
*  Cropping images
*  Resizing images
*  Reshaping images

All these things are used in other examples, and may be useful various bits of assessment.

It's important to note that within python, there are lots of different packages that can be used to manipulate images. In this example I'll use opencv, however skimage and PIL are also good choices, and have very similar functions to what is demonstrated below. If you are familar with one of those, feel free to use it throughout the semseter.

### CAB420_DCNNs_Additional_Example_2_Convolutions.ipynb

In this example we will:

*  Have a look at the convolution operator, which is the building block of convolutional neural networks

### CAB420_DCNNs_Example_3_What_Does_the_Network_Learn.ipynb

This example is all about exploring what is being learned by a DCNN. Please note I'm not worried about performance in here. The networks that we're using a very simple, because simple networks are easier to visualise. Performance of models is not the focus here. We're going to train a few networks, and after training there will be a confusion matrix, but this is really just here as a sanity check to make sure that something actually trained.

This example really is very visual, it's all about looking at the pretty pictures to try and get a sense of what is actually being learnt.

### CAB420_DCNNs_Additional_Example_4_Training_Parameters.ipynb

There are a lot of options to set when training networks, let's look at some.

### CAB420_DCNNs_Additional_Example_5_Layer_Order_and_Overfitting.ipynb

There are two big problems with deep neural networks:

*  they take a long time to train
*  they are easy to overfit

This example will look at a couple of things to improve that, but also at what can happen when you try to reduce things too much.

### CAB420_DCNNs_Additional_Example_6_Variation.ipynb

Neural networks contain a lot of parameters, typically in the order of hundreds of thousands, or millions. Typically, these paraemters are randomly initialised, and then data is fed to the model is a random order. This means that each time a network is trained, you will get a slightly different network. Generally, if you're network is appropriate and you have sufficient data, results over multiple trials will be fairly similar, but they will be different. This example explores this, and looks at how we can combine multiple models to get a small increase in accuracy.

It should be noted, there is a lot of research on model ensembles. This example is a very superficial look at this idea - but it is an idea that has a lot of interest and possibilities.

### CAB420_DCNNs_Example_7_Breaking_VGG.ipynb

The initial DCNNs we've looked at, and those in the literature, simply relied on stacking layers one on top of the other, with up to a few convoluations followed by a max-pooling operation to reduce the resolution, and then a repeat of this until a sufficient number of layers was reached. More layers was seen as better, as it allowed for richer and richer representations. However, after a while performance stops improving, and can in fact go backwards. This example simply explores this.

This example is pretty light in terms of explanation of the networks, etc. If you're unclear what's happening in places, go back and revise the previous DCNN examples.


### CAB420_DCNNs_Additional_Example_8_1x1_Convolutions.ipynb

This example has a look at 1x1 convolutions, and visualises what these do within a network.

### CAB420_DCNNs_Additional_Example_9_Runtimes.ipynb

In this example we will look at how fast (or slow) DCNNs are compared to:

*  CKNNs
*  SVMs
*  Random Forests

Why are we doing this? We've spent a bit of time saying how slow and demanding DCNNs are, so let's put that in perspective using a large number of samples of moderately high dimensional data.

***

## Bonus Examples

### CAB420_DCNNs_Bonus_Example_DCNNs_and_Audio_Data.ipynb

This example looks at using audio data with a deep convolutional neural network. We're using a dataset broadly similar in size to our image examples. We'll look at two approaches to use audio data:

*  Converting audio to an image, and then using a regular 2D CNN
*  Leaving the audio as a 1D signal, and using a 1D CNN

As we'll see, the 1D CNN looks much like a 2D CNN, with just a change in name. There are a number of other things that can be done with audio (and other 1D signal data such as bio-medical signals) that are beyond the scope here, but further improve performance. Interested students are encouraged to look into SincNet layers as a replacement for the first 1D CNN layer.

This example is heavily based on [this tensorflow example](https://www.tensorflow.org/tutorials/audio/simple_audio).

### CAB420_DCNNs_Bonus_Example_Explainability_for_DCNNs.ipynb

Neural networks have lots of parameters. Where as with simpler models, like regression, or random forests, we can inspect parameters and attach meaning to them, this is harder with a neural network. In recent years a variety of techniques have sprung up to help understand the decisions neural networks make (and some such like methods, like Shapley values, work on non-deep models as well). This example will have a look at two methods (the links below were used as a basis for what's in this example):

*  SHAP (SHapley Additive exPlanations)
*  Grad-CAM

There are way more methods than these out there, but this will serve as a starting point if this is something that's of interest.

### CAB420_DCNNs_Bonus_Example_Model_Calibration.ipynb

If they were a person, a typical DCNN would probably be characterised as an arrogant bastard. They are the professional athletets of the machine learning world: highly tuned, high performing beasts, that are incredibly confident in their own ability.

What I'm getting at with this ill-conceived analogy is that neural networks are often very confident in their decisions. A typical model trained for classification will output some softmax scores. If we get say a score of 0.8 for a given class, we might assume that this indicates that there is an 80% chance that the true value will be this class, i.e. 20% of the time our network will be wrong. In practice, this is rarely the case and networks are often massivley over-confident, i.e. they are wrong more often than they should be if the softmax scores were actually probabilities.

One way to address this is to calibrate the model. Essentially, this amounts to applying a weight to the softmax scores after the network has been trained, such that soft-max scores are now better calibrated.

### CAB420_DCNNs_Bonus_Example_Tensorflow_Dataset_API.ipynb

This is, more of less, my quick start guide to tensorflow datasets. If you're here, it's also going to be worth having a look at the [documentation on all this too](https://www.tensorflow.org/datasets).

Before we dive in, why TFDS? Datasets can be big. In CAB420, generally we use small datasets that fit into memory easily, however often this is not possible. When dealing with large datasets, a popular approach is to stream the data from disk, only storing in memory what's needed for the next batch (or few batches). There are lots of ways to do this, and the tensorflow datasets API is one such way.