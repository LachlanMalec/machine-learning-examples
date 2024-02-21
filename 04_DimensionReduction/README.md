# Dimension Reduction

## Core Examples

### CAB420_Dimension_Reduction_Example_1_Principal_Component_Analysis.ipynb

In this example we'll explore PCA using some simple 2D data that we create. This will allow us to see how PCA operates.

### CAB420_Dimension_Reduction_Example_2_PCA_and_Dimension_Reduction.ipynb

PCAs main claim to fame is dimension reduction. Often in machine learning (and data science in general) we have huge data sets, with potentially thousands of dimensions. For an example, consider an image. A small image (lets say 256x256) has 65,536 pixels, and we can consider each pixel as a dimension.

In most cases however, not all dimensions are equal. If we think back to our regression problems, usually there were a small number of variables that were really important, and then a bunch of others that were just along for the ride. PCA, and dimension reduction techniques in general, give us way to find those dimensions that are most important, and get rid of the others.

Using this example as a basis, apply PCA to a regression dataset (maybe one of our cycling datasets from the first couple of weeks) and try the following: 

*  Transform the data using PCA (don’t transform the response, just the predictors), and use the transformed data in regression. 
*  Compare the performance of the regression with the original and PCA transformed data. 
*  Look at the correlation between predictors in the PCA transformed data. 

### CAB420_Dimension_Reduction_Example_3_Linear_Discriminant_Analysis.ipynb 

Having seen how great PCA is, let's break it. And then make everything better again through Linear Discriminant Analysis.

### CAB420_Dimension_Reduction_Example_4_Linear_Discriminant_Analysis_II_Action_Time.ipynb 

We'll apply LDA to some real data now. And then we'll break it too.

Once you've broken and fixed things, take this example as a starting point, try the following: 

* Change the data. Grab another dataset from an earlier week (maybe the wine classification data from the week 3 prac) and see how that performs with PCA and LDA. 

### CAB420_Dimension_Reduction_Example_5_Eigenfaces.ipynb

The "Eigenfaces" method for face recognition is an enduring pattern recognition method, that has also been applied to most (all?) other biometrics. While it's been totally superceeded by more recent methods, it's still a great demonstration of dimension reduction.

