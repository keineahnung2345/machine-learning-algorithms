# machine-learning-algorithms

## Logistic Regression
We all know that we should use cross-entropy loss with logistic regression.

But what if we use mean square loss?

Following are pictures of the curve MeanSquareCost(weight).

These pictures show that there will be  multiple local minima if we use MeanSquareCost.

![alt text](https://github.com/keineahnung2345/machine-learning-algorithms/blob/master/Logistic%20Regression/images/mean_square_cost_function_of_w_1.PNG)

![alt text](https://github.com/keineahnung2345/machine-learning-algorithms/blob/master/Logistic%20Regression/images/mean_square_cost_function_of_w_2.PNG)

![alt text](https://github.com/keineahnung2345/machine-learning-algorithms/blob/master/Logistic%20Regression/images/mean_square_cost_function_of_w_3.PNG)

## Neural Network
We all know that we should random initialize neural network before starting to train.

But what if we initialize neural network with 0?

The demo shows that the weight of neurons in the same layer will be the same, which make them perform the same function, and this symmetry won't be broken during training.
