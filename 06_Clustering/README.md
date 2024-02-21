# Clustering

## Core Examples

### CAB420_Clustering_Example_1_KMeans_Clustering.ipynb

In this example we'll look at k-means clustering. This will break a dataset into N clusters, where we have to specify N.

### CAB420_Clustering_Example_2_Gaussian_Mixture_Models.ipynb

Ih this example we'll look at a GMM. This extends k-means, such that rather than having hard cluster boundary, we now have a set of disrtibutions, and we can capture uncertainty. For example, we'll be able to tell when a point is close to being part of two different classes.

### CAB420_Clustering_Example_3_How_Many_Clusters.ipynb

Let's compare K-Means and a GMM, and look at what happens as we increase the number of clusters. To do this, we're going to use some real data. We'll use a segment of data that shows NY taxi usage. We'll only use one day's data to keep the example from being too slow, and we'll focus only on trips that end at JFK airport, again to reduce the data and make it easy to visualise.

Using this example as a starting point, try the following: 

*  Pick a different patch of New York. Change the latitude and longitude values to pull out some other patch of the city (use google maps or similar to pull out a region). Using this modified data set repeat what the example does, taking note of: 
*  What value of K is recommended by the various methods? How does it differ from what you saw before, and how does it vary between K-means and the GMM? If you make the area bigger or smaller does the suggested K change? 

Now go and look at example 4, and based on that try the following:

*  Can you see patterns of use in the different data based on cluster occurrences? 
*  Can you find abnormalities using a GMM? 

### CAB420_Clustering_Example_4_Clustering_Applications.ipynb


In this example we'll look at two example clustering applications:

*  Segmentation / Knowledge Discovery
*  Anomaly Detection

In this example we're using some data from the New York Citibike service (more data can be found here). We'll use data from three months:

*  July, 2019
*  December, 2019
*  July, 2020

Note that with clustering, we don't always have the same training, validation and testing splits that we do for supervised models. This is due to the fact that we don't have labels, and so evaluating overfitting is much more challenging (if at all possible).

#### Segmentation / Knowledge Discovery
This is a common task when we have a large dataset with no labels. Often in such situations we seek to understand what is present in the data by grouping related items. This also aids visualisation, and can also help simplify the data by reducing a large dataset to a few representitive examples (i.e. cluster centres).

In this task, we'll use the first two sets (July and December '19) to train our clustering models. We'll then use these models to see if there is a difference in bike usage between July '19, Decmeber '19 and July '20.

#### Anomaly Detection
This task involves finding points that are unusual, or abnormal. A typical process for this is:

*  Train the model on a batch of normal data
*  For a new point, determine if it's fit the trained model. If it doesn't, then the point is abnormal

Again we'll use the first two sets (July and December '19) to train our clustering models. We'll then look for abnormalities in the July '20 data. Note that this therefore assumes that the July and December '19 data does not contain any abnormal points, which is probably not the case, though at least we can reasonably expect them to be very scarce.

#### Runtime Warning

Finally, a quick warning that due to the size of these datasets the two blocks of code that compute reconsturction errors, approximate BIC and BIC for different values of N are very slow. The rest of code runs quite quickly.
  
  
***

## Additional Examples



***

## Bonus Examples

### CAB420_Clustering_Classification_Bonus_Example_BOVW.ipynb

A while back, we mentioned this thing called "Bag of Words". This is a text processing method, and the idea is that rather than look at some piece of text as a sequence of words and consider it "in order" (which brings all sorts of constraints, like requiring us to have sequences all be the same length, and meaning that the position of words becomes potentially restrictive), we simply count how many times each word occurs and use the resulting histogram as our feature. This means that:

*  Any sized input will get transformed to the same length, based on the size of our dictionary (i.e. how many words we have)
*  Any order information is lost

In this example we're going to apply the same idea to image data. The trick is to get some visual words, which will allow us to replicate the text pipeline, including adding all the bells and whistles such as TF-IDF transforms and the like.


### CAB420_Clustering_Bonus_Example_HAC.ipynb

In this example we'll look at HAC. HAC does not require us to specify the initial number of clusters, and instead will build a tree that captured the distance between all points. We can then 'truncate' that tree at a level of our choosing, to get the number of clusters.

### CAB420_Clustering_Bonus_Example_DBScan.ipynb

Our final clustering method is DBScan. Like HAC, this one doesn't need us to specify the number of clusters. Instead we specifya minimum cluster size, and a proximity threshold, and it goes ahead and builds a set of clusters by finding points that are near each other. Nicely, it also considers noise, and so will attempt to filter out isolated points that are like nothing else, rather than trying to fit those into a cluster where they don't really go.

### CAB420_Clustering_Bonus_Example_Evaluating_Clustering_Methods.ipynb

How do we know when our clustering is correct? Or when one solution is better than another? To do this we need a way to evaluate the accuracy of our clustering methods. Before we go too far, it's important to state that it's not always possible to do this. To evaluate our clustering we need to know what it should look like, i.e. we need to have some ground truth labels for what the clusters actually are. Often you won't have this, or if you do other techniques (such as classification) may be more appropriate for your problem.

Assuming that we have ground truth, there are a lot of metrics that have been proposed to evaluate clustering accuracy incuding:

*  Rand index
*  F-measure
*  Jaccard index
*  Dice Index
*  Fowlkes-Mallows Index
*  Mutual Information (and Normalised Mutual Information)
*  and many more.... ([wikipedia](https://en.wikipedia.org/wiki/Cluster_analysis#External_evaluation) has a reasonable overview of these for those who are interested)

We're going to look at some metrics not on that list:

*  Purity
*  Completeness
*  V-Measure

These have been chosen becuase they are quite useful, and quite easy to interperet and understand what is being measured.

### CAB420_Clustering_Bonus_Example_Applying_Clustering_to_Diarisation.ipynb

In this example we'll look at an example application of clustering: diarisation. Diarisation is the task of grouping instances by their identity, and is usually applied to either faces/people or segments of speech. In a video/image context, the task is determining which instances of a face belong to the same person, for speech it's determining who spoke when.

We'll focus on grouping face images in this example as:

*  it's easier to display the images than listen to audio
*  there is a very easy to use python face detection and recognition library that we can use to enable this, meaning we can get straight to our application much quicker

We're going to use this [face-recognition package](https://pypi.org/project/face-recognition/) which makes use of the excellent [DLib](http://dlib.net/) toolkit.

To achieve our task, we have a few steps to follow. We need to:

*  Locate faces
*  Encode each of the faces to capture identify information, obtaining a set of embeddings (like we saw out of our Siamese and Triplet networks)
*  Cluster the embeddings

We'll rely on the above mentioned face-recognition and DLib packages to help with the first two tasks, and then explore our different clustering methods for the third.

### CAB420_Clustering_Bonus_Example_Comparing_Clustering_Methods.ipynb

Let's now look at the four clustering methods side by side. We'll use the NY taxi data, and constrain ourselves to JFK airport. We'll also look at two applications:

*  Visualisation/Knoweldge Discovery; and
*  anomaly detection. And see how well each method works for each.

This is, to a large extent, a revist of the fourth example from last week that explored the new york citibike data, but we're now adding HAC and DBScan into the mix.