# Classification

## Core Examples

### CAB420_Classification_Example_1_Classification_Three_Ways.ipynb

In this first example we'll explore:

*  Three types of classification models, SVMs, K-Nearest Neighbours, and Random Forests.
*  Introduce confusion matricies, which are a good way to visualise performance.
*  Look at how we select hyper-parameters, and ways to help automate this process

When you've had an explore of this first example, try the following:

*  For each type of model, adjust the hyper-parameters and observe the change in the shape and complexity of the decision boundary. You may also wish to change the two dimensions being selected to use for classification to help visualise the behaviour.

### CAB420_Classification_Example_2_Multi_Class_Classification.ipynb

In this example we'll explore:

*  Extending classification to multiple classes for CKNN, random forests, and SVMs
*  Different encoding schemes to adapt a binary classifier (i.e. an SVM) to a multi-class problem
*  How class balance challenges can be (somewhat) mittigated

This example will also cover classification of text data, however we will, to an extent, brush over this.

As an activity, using the second example as a starting point (and the first and third as a guide as needed), try the following:

*  Implement a grid search to select the best hyper-parameters for the SVM, Random Forest, and CKNN in the second example.
*  Compute Precision, Recall and F1 score for each of your final SVM, Random Forest and CKNN. Note the performance of common and rare classes for the different metrics.

### CAB420_Classification_Example_3_Classification_Metrics.ipynb

To this point, we've simply used raw accuacy as our performance measure. This can be expressed as:

$$A=\frac{TP+TN}{N}$$
 
where A is the overall accuracy, TP is the number of true positives (i.e. points that are true that we classify as true), TN is the number of true negatives (points that we correctly label as false) and N is the total number of samples. This is a very intuitive measure, but it also lacks nuance and doesn't tell us much about the type of errors our model makes.

In a binary classification case, we have two possible types of errors:

*  False Positive: classifying something that belongs to class 0 as class 1 (also referred to as a false detection)
*  False Negative: classifying something that belongs to class 1 as class 0 (also referred to as a missed detection) Extending this to a multi-class case, we also have confusion between classes.

All of these errors are well captured by a confusion matrix (which is why confusion matrices are so good), but when evaluating lots of models we'd also like to be able to obtain one or more numbers to capture performance, as when simply looking at confusion matrices en-masse, it may be easy to miss key details.

This example will re-use the data from our second classification example (mulit-class beer classification) and explore some alternate performance metrics that better capture performance.

***

## Additional Examples

CAB420_Classification_Additional_Example_Classifier_Parameters_and_Decision_Boundaries.ipynbLinks to an external site.

This example illustrates the impact of classifier parameters on the learned decision bounaries of our three classifiers:

*  Support Vector Machines
*  K-Nearest Neighbours Classifier
*  Random Forest

This example is not a detailed run through of these algorithms, and supports the other examples from this week. Think of this really as a supplementary example with lots of pictures. It mainly exists to generate plots and figures for use in the slides.

***

## Bonus Examples

### CAB420_Classification_Bonus_Example_HOG.ipynb

To allay any concerns, yes, this example is vegan. We are not talking about the source of pork, but rather the Histogram of Orientated Gradients, no pulled pork reciepies here (though in brief, I recommend letting the pork soak for a day or two in apple cider vinegar; and then using a rub based around salt, smoked paprika, cumin, chipotle, onion power, garlic power and corriander seed; and then smoke low and slow).

So, why do I want histograms of gradients? Let's think about our classification methods so far. With tabular data, we might have a dataset with 10 columns, the first 9 are different properties of the thing that we're interested in, and the last column is our class. We take the first 9, use them to decide on the class. One critical thing here is that the first 9 columns are always the same, and always in the same order. If we suddenly change the order of those columns, any model we've trained is invalid and won't perform as we expect.

Now consider an image. An image is a collection of pixels. While we can essentially look at an image as a matrix (2D if it's grayscale, 3D if it's colour), the classification methods we've looked at so far are all about processing 1D data, so to classify an image we'd vectorise it. This is shown below:

```
    1 4 1
    4 4 4    ->   1 4 1 4 4 4 1 4 1
    1 4 1
```

We go from a 2D matrix to a vector by taking each column and just stacking them, one on top of the other.

Much like our tabular data, now each dimension has a meaning. In our case, the first dimension is the top left corner of the image, and the last dimension is the bottom right corner. Other dimensions also have a fixed "meaning", i.e. they correspond to a location in the image.

At first glance this might seem fine, but there are a couple of issues:

*  We get big feature vectors quickly. Our  image only gives us  dimensions (also known as features), but what about a  image (i.e. 200 pixels wide, 200 pixels high, and in colour)? This gives us  dimensions. A lot of classification methods will start to grind a bit with representations of that size.
*  What happens if we take an image, and shift it one pixel to the left? Or one pixel up, or down, or right? None of these operations change the underlying meaning of the image. If it was an image of a narwhal, it is still an image of a narwhal. But the order of the data has now been changed.
*  What happens if the lighting changes? This will change the contrast, the brightness, etc. Every pixel value will change, some by a lot - but the underlying thing in the image is the same.

What we want is a way to make our data representation more compact, while also making it less sensitive to small changes in the image, such as the image being translated (i.e. shifted) slightly, and ideally invariant to other factors such as lighting. There are a lot of ways to do this, and the method we'll consider here, the Histogram of Orientated Gradients (HOG), is one such as approach.

HOG will take an image and divide it into a grid. Within each grid cell, it will look at the gradients of the pixels. The gradient of each pixel is characterised by a magnitude, and a direction. The magnitude tells us how strong the gradient is, the direction simply tells us in which direction the gradient is moving. Strong gradients correspond to edges, and with the direction we can determine if the edge is a vertical, horizontal, or something else. All up, this gradient information tells us about the local texture in the patch. When we aggregate patches across the image, we get an idea of the gradient/texture across the image, and we can use this as the feature to represent the image. Compared to raw pixels, we get:

*  A more compact feature representation (unless we have tiny images)
*  Something that is invariant to small translations and shifts
*  Something that has at least some invariance to other external factors (lighting, etc).

Enough preamble, let's crack(ling) on and have a look at HOG.

### CAB420_Classification_Bonus_Example_BOW.ipynb

A common characteristic of the majority of machine learning methods (and certainly classifiers) is that they like a fixed size input. By this, I mean that our input data is always the same size. When dealing with tabular data, this isn't an issue. We have a fixed number of columns and every sample has the same set of columns. Even with data like images, this can be managed by resizing images. Text however, is a bit harder.

If we're trying to classify articles into their underlying topic, it's very unlikely that we're going to see every article being the same length. We saw this issue with the beer names example, and we solved this by padding the sequences so that they were all the same length. But this approach has other issues:

It can greatly increase the size of our features, which can make our classifiers harder to train.
The feature representation makes the model dependent on the order of the data, i.e. it means that the system needs to have the same words in the same places.

Bag of Words is a feature transform that allows us to go from a variable length sample to a fixed length representation. To do this, we transform our data into a histogram, that measure how many instances of each word we have. There are a few steps involved in this:

*  Clean the text. This means usually converting to lower case, removing punctuation, removing stop words, performing lemmatisation
*  Build a dictionary. This contains all possible words. We may at this point choose to exlude words that are very rare (if they're very rare, it's harder for our model to understand their meaning), very common (if they're very common, they're probably not much use for discriminating between classes), or just take the top N words to avoid having histograms that are too big.
*  Convert our text into a Bow of Words via the dictionary. This will take us from a variable length text sample to a fixed length histogram. Each word in the dictionary will have an entry in the histogram. The value in the histogram will be how many times that word occurs in the sample.

Bag of words solves our issues of variable length sequences, however it's not without problems:

*  It can result in very large feature representations. If we have a dictionary of 10,000 words, we'll transform each sample into a length 10,000 vector; even if the original sample is only a few words of text.
*  We destroy order. The histogram contains no information on the order of words, so information is lost.
*  The histogram representation is not invariant to the length of the document. We get the same sized representation, but a longer document will have much bigger values in the histogram, as there are more words to count.

There are various solutions (or partial solutions) to each of the above issues, which this example explores.