# Metric Learning

## Core Examples

### CAB420_Metric_Learning_Example_1_Siamese_Networks.ipynb

In this example we will look at Siamese Networks. These are very useful when we have things that we want to compare. A classic example is a verification problem, where we want to tell if two things are the same. In our example, we're going to try and work out if two items of clothing are the same type of thing.

### CAB420_Metric_Learning_Example_2_Contrastive_Loss.ipynb

In this example we will continue our exploration of Siamese networks. Our first Siamese Network used binary cross entropy. This worked ok for telling us if things were the same, or different, but didn't really say how similar or how different things are. Often in machine learning, we'd like to rank things by similarity. For example, consider a biometrics scenario. We might want to find a suspect in CCTV, but rather than check if each and every person is the one we're after, it makes more sense to return the best  
  matches. However to do this, we need a way to quantify how similar things are. This is where contrastive loss comes in.

This example is going to repeat a fair bit of code from the CAB420_Metric_Learning_Ex1_SiameseNetworks demo, so if you haven't looked at that example, please do so before diving into this one.

For some further exploration, using this example as a starting point try the following: 

*  Change the backbone network. You could use a pre-trained network here, or just make it more complex (or less complex). Experiment with different backbones with various levels of complexity. 
*  Change the size of the embedding. This is also worth playing with as you change the backbone. 
*  Add data augmentation to the pair/triplet generation.  


### CAB420_Metric_Learning_Example_3_Triplet_Loss.ipynb

What's better than a pair of images? Three images! With this realisation, the triplet loss was born.

The triplet loss really just extends the siamese networks by making them bigger. Now we have three streams, three images, and a loss that looks all of those together. Cricially, the triplet is made of two image from the same class, and one from a second, different, class. So when the network evalautes a triplet of images, it evaluates a positive pair, and a negative pair simultaenously.

This example is going to build fairly heavily on the CAB420_Metric_Learning_Example_2_Contrastive_Loss example.

Like the previous example, again you can explore a bunch of things to further your understanding. For example, you may wish to try:

*  Change the backbone network. You could use a pre-trained network here, or just make it more complex (or less complex). Experiment with different backbones with various levels of complexity. 
*  Change the size of the embedding. This is also worth playing with as you change the backbone. 
*  Add data augmentation to the pair/triplet generation.  

***

## Additional Examples

### CAB420_Metric_Learning_Additional_Example_Embedding_Size.ipynb

In the previous examples, we've always used an embedding size of 32. Why? Is this is a good size, or just a number that I picked? This example has a little exploration of that.

In this example, we'll take our triplet network and run that with different embedding sizes.

***

## Bonus Examples

### CAB420_Metric_Learning_Bonus_Example_TFDS_and_Triplet_Mining.ipynb

If you're here, I am expecting that you've gone through the metric learning content, and have a pretty good idea of what is hapenning when we implelement a triplet loss. Hopefully, you'll also have worked through the prac (if not, stop reading this and go and do that). Having gone through the triplet network faff a few times, you might be a bit over it, and be of the opinion that the whole process of setting up your generator, building triplets, passing them through multiple branches, etc, is too much bother.

However, in a dramatic plot twist, it turns out that there is an easier way. Before you get all upset that we went through the other way first, please put your pitchfork down and keep in mind that:

*  The other way is not that bad. You should see what it looks like in MATLAB for example. Really, it's pretty easy to set up a triplet loss.
*  It's really important to understand what's going on with these metric learning methods, and what's actually being minimised by the training process. The approach we're going to explore here really hides of all the important stuff where we get distances between triplets, and seek to minmise that. So having an understanding of what's actually going on is important, as the code we're going to use here doesn't capture the underlying mechanics at all.

With that out the way, there are a few limitations of our previous triplet approaches that are worth highlighting:

*  We have to generate all of our triplets in advance.
*  Not all triplets are good learning examples. By that, I mean a lot of them are possibly very easy. They contain a positive pair that is very similar, and a negative image that is clearly very different. Ideally, we'd like triplets where things are a bit less obvious (if you're not certain about what I mean here, have a look at the "semi-hard and hard triplets" section below.
*  It's not super efficient. We have to pass three images through the network to generate a single training example. In a batch of 128 triplets, 384 images go through the backbone. There's simply a lot of underutilisation there.
*  The prior approach doesn't scale cleanly to large (i.e. cannot fit in memory) datasets. How would you integrate it with tfds for each to stream data into the model?

The solution to the above woes is to not create triplets - or at least to not create triplets at the input to the network. To explain this, consider a typical network like we've done (over and over again) for classification. Think about what a batch of that data looks like. We have $N$ samples, each of those samples has it's data, and a class label. Within that one batch of data, there may exist dozens, or hundreds, of potential triplets. So rather than pass triplets into the network, why not pass a batch of data through, get the embeddings, and then create a huge set of triplets? What's more, having created these triplets we could just keep the ones that are interesting for learning. This process is referred to as online triplet mining, and if you'd like a really good breakdown of how this is achieved in code, [this blog post](https://omoindrot.github.io/triplet-loss). is well worth a read. Here, we'll just the loss function in the tensorflow addons package to achieve our aims.