# Linear Regression
These examples cover:

*  Linear Regression
*  Ridge Regression (a regularised form of regression)
*  LASSO Regression (another regularised form of regression)

This README will outline some activities you may wish to complete with respect to the core examples to help understand the methods, and provide an overview of the bonus and other examples.

## Core Examples

### CAB420_Regression_Example_1_Linear_Regression.ipynb

This is our first regression example and is a good starting point to explore regression a bit. Using this example as a starting point, try the following:

*  Change the response variable from the overall count (cnt) to either the casual or paid user counts. Re-run the model and note any changes.
*  Add additional predictors. Explore adding additional categorical terms to the model. Re-run the model and observe what happens as you add more predictors. Focus on:
  *  Changes in the p-values of terms;
  *  Changes in the R2of the model;
  *  Changes in the RMSE of the model on the training and testing sets.
  
### CAB420_Regression_Example_2_Regularised_Regression.ipynb

This is our first regularised regression example. Again, there's lots to explore. Using this example as a starting point, try the following:

*  Change the predictors and response of the model to focus on some different sites in Brisbane’s road network. Start by using many fewer predictors than we do by default.
*  With the changed predictors and response, re-run the example. Note the changes that are observed with respect to the selected value of lambda.
*  Increase the number of predictors now, either by simply adding more columns, or by adding higher order terms (or both). Re-run the example and look at how the selected values of lambda change. Take particular note of what happens with LASSO regularisation, and how many terms are non-zero at the optimal value of lambda.
*  Try turning standardisation on and off. Are the same terms supressed (Ridge) or eliminated (LASSO) in both settings?
*  Using the second example as a starting point, try the following:

### CAB420_Regression_Example_3_Regression_with_Less_Data.ipynb

This is our final *core* regression example. It's a bit bare bones. To further practice your skills, for the final three models (linear, ridge and lasso regression), compute the and draw qq-plots to further compare the models.

***
## Additional Examples

### CAB420_Regression_Additional_Example_Regularisation_Impact.ipynb

This example further explores regularisation impact, in particular how increasing shrinks the values of coefficients, and makes the predictions from the model closer to a constant. 

***

## Bonus Examples

### CAB420_Regression_Additional_Example_Regression_Diagnostics.ipynb

Residuals, and the various plots of them, often cause some confusion. In this additional example, we're going to look at some toy scenarious and see what the regression diagnostics plots can look like for the various toy scenarios.

We will stress that this is all just made up toy models and data, but should hopefully this should illustrate what these plots can look like, and how that can relate to the what's in the actual data.

Three of the main assumptions that we are looking for with an OLS model are
1. The residuals are normally distributed
2. There is actually a linear relationship between predictors and our response
3. there is constant variance in our residuals

There are other assumptions that go into linear regression models, but these are probably the main ones, and the ones that we've consider here.

Lets see how different plots can help us identify if these assumptions are being violated.

Again, the data is totally made up. This example really is just all about highlighting some ways to use the diagnostic plots to identify some of this stuff.

### CAB420_Regression_Bonus_Example_Standardisation.ipynb

This is simply a little extra look at standardisation and why that removes the intercept term. There's even a little video to go with it (go over to canvas for that if you're interested).

### CAB420_Regression_Bonus_Example_GLMs.ipynb

In linear regression, we have a *linear model* where:

*  The response variable (AKA, our output), $y$, is expressed as a linear combination of our predictors (or inputs), $X$;
*  The underlying relationship between $y$ and $X$ is linear;
*  Our response variable is unbounded, i.e. our output range is $[-\infty..\infty]$ and;
*  The distribution of the residuals (errors), is Gaussian.

What happens when the above isn't true though? A good example might be trying to predict a count. Perhaps predicting the number of cyclists given some weather conditions. Here, an unbounded output, $[-\infty..\infty]$, is clearly not appropriate, and $-\infty$ cyclists is not going to be a good estimation in any circumstances. Ideally, we'd like our model to not even be able to come up with such an answer.

Generalised Linear Models, as the name implies, generalise aspsects of our linear regression so they can be more correctly applied to other scenarios. This means that we can end up with a model that better fits our data, though it does mean that we need to make some additional decisions when setting out model up. Conveniently, GLMs also look a lot like linear regression, which means that they're not that hard to get a handle on from where we're at currently.

### CAB420_Regression_Bonus_Example_StatsModels_vs_SKLearn.ipynb

One of the great things about python is that there are lots of ways to do any one thing. At times, this can also be a bit shit, as you may not know quite which approach to take.

For linear regression, there are really two main packages that you will see discussed on the internets:

*  StatsModels, which we've used in most of the linear regression examples
*  SKLearn, which we're going to start using for classification

Which is better? Well, that kind of comes down to what you want to do. StatsModels is probably richer in terms of giving you lots of the stats goodness that underpins the models (and which we're totally glossing over in CAB420), but SKLearn is a more mature package with the same API for a whole heap of other stuff. For more general ML type stuff, SKLearn is probably better. For more stats heavy stuff, StatsModels is probably better. The reason we've used statsmodels so far is that we do at least have a passing interest in things like p-values, which statsmodels gives us, and sklearn doesn't provide quite as easily (given the ability to misuse p-values, this is perhaps not the worst choice on sklearns part). Really though, the bottom line is you can use either.

This example is simply going to have a look at the two side by side. What you'll see is:

*  In terms of use, they are very similar. Fitting models, making predictions, a finding an optimal value of $\lambda$, all look very much the same in both.
*  Results from the packages are similar, but not identical. The different results is interesting, but is not actually cause for alarm, as we'll discuss further down in the example.