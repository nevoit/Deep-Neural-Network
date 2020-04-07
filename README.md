# Deep-Neural-Network
In this assignment we were tasked with creating a Deep Neural Network that can classify images from the MNIST database.
The data was loaded through the keras API for Python. As was specified, 20% of the training data was used as a validation set in order to track our progress between training iterations.
The input layer receives a single 1d array, so the images had to be slightly manipulated. Since each image is 28*28 pixels, a total of 738 pixels, we reshaped each image into a 1d array. This array serves as the input to the network. Additionally, this is a multi-class problem with 10 possible classes, so the output layer has 10 neurons.
Therefore, the architecture of the network is [784, 20, 7, 5, 10] neurons (the number of neurons in the hidden layers is taken from the assignment guidelines). The learning rate is 0.009, and we used batch sizes of 512 and up to 100,000 iterations (around 1066 epochs). The size of the training set is 48,000 samples (i.e. 94 iterations per epoch). We found success with these parameters.

# Additional Functions:
We added several useful functions to our code, mostly for code clarity.
- **Image_classification_experiments**:
A function for loading the mnist dataset used in this assignment. Additionally, here we set the various parameters used for the DNN and it wraps the function that trains and runs the experiment. Here we experimented with various batch sizes and iterations.
- **Load_mnist_data**:
The image_classification_experiments function calls this function to load the dataset through the keras API. In addition, this function normalizes the training data by dividing each cell by 255 since the pixels are grayscale. This function also reshapes the input, which is 2d arrays of size 28*28 to a 1d array of 784. 
-	**Run_experiments**:
This function loops through different possible hyperparameter values and calls run_comb() with the appropriate parameters. Finally, it saves a csv file with the results of each run.
-	**run_comb**:
This function receives the parameters and trains the DNN with the different possible combinations. It also times the time it takes to train the DNN and predicts on the test set. 

Lastly, it appends the results to a csv file that we can check later to see the performance of the DNN with different parameters.

-	**Check_stop_critera**:
This function is called after each iteration and checks the current accuracy of the model on the validation set and the current cost (i.e. the result of the loss function) on the
training data. We do this so we can stop the training if there is no significant improvement.
-	**Compute_accuracy**:
Returns the accuracy score of the prediction compared to the true classes.
- **add_to_results:**:
Appends to a file containing all the results of the previous runs the new run just completed. Each run is a complete training iteration that was halted because there was no improvement in the accuracy. We do this so we could run hundreds of experiments and find good values for batch sizes and iterations.


# Dropout
We chose to implement the dropout bonus. We did this by creating a new data frame with the same shape as the output of each layer, after the activation function (during the forward part of the network). Each cell has a certain probability to be initialized (randomly) to False. This probability determines the dropout rate (r). Then we compare each neuron’s output to this new dropout data frame, and if the cell there is ‘False’ then that particular output is ignored. This way we ignore some neuron’s output. Then, in the next layer, we treat the output of a dropped as 0. The results of this feature are detailed below.
Lastly, we take the neuron’s output and scale it up by dividing it by 1-r. For example, if we have u units (u neurons) in the current hidden layer (l), the output of the activation function will be reduced by 100* r percent, and on average we end up with u*r ignored neurons.

This means the value of the input of the next hidden layer is:
$$ Z^{(l+1)}=w^{(i+1)}*a^{l}+b^{(i+1)}$$
By dividing $$a^l$$ by $$1-r$$ will bump up it back up the roughly $r$ percent, so it will not change the expected value of $$a^l$$. This technique called the inverted dropout and this effect is that no matter what we define as the dropout rate, this inverted dropout ensures that the expected value of $$a^l$$ remains the same. This technique should help in the testing stage since we have less of scaling problem.
Please note that we do not use the dropout during the testing and also we do not use the dropout on the output layer.


# Experimental Results:
After implementing our DNN, we needed to not only see how batch normalization and dropout affected the results, but to find good values for batch size and number of iterations. In order to do this, we ran the network many times with varying hyperparameters and saved the results. In the end, we found that a batch size of 512 coupled with up to 100,000 iterations worked well (although it would converge much before that typically).

We performed four different experiments:

#### Expierment 1: Basic DNN, without batch normalization or dropout: 

| Final Accuracy      | Results |
| --------- | -----:|
| Training  | 91.6% |
| Validation     |   90.4% |
| Testing      |   91.26% |

**Training Time :**  71.68 seconds / 10,600 iterations

#### Expierment 2: With batch normalization and without dropout:
We found that a beta of 0.1 and Gamma of 1 leaded to good results.

| Final Accuracy      | Results |
| --------- | -----:|
| Training  | 94.53% |
| Validation     |   93.47% |
| Testing      |   93.25% |

**Training Time :**  116 seconds / 18,300 iterations
It is evident that batch normalization helped the results, albeit slightly. However, the run time was quite longer.

#### Expierment 3: Without batch and with dropout
We used a dropout rate of 10% in this experiment:

| Final Accuracy      | Results |
| --------- | -----:|
| Training  | 33.6% |
| Validation     |   33.33% |
| Testing      |  32.79% |

**Training Time :**  14.7 seconds / 2,200 iterations
Dropout by itself reduced the performance significantly, although the neural network converged quickly (although this doesn’t hold much weight when the results are so lackluster)


#### Expierment 4: With both batch normalization and dropout:
We used the same parameters as before (beta 0.1, gamma 1, dropout rate of 10%)

| Final Accuracy      | Results |
| --------- | -----:|
| Training  | 61.81% |
| Validation     |   62.58% |
| Testing      |  62.78% |

**Training Time :**  18.9 seconds / 2,700 iterations

**Conclusion**: Although the results were much lower than without dropout, they were significantly better than using solely dropout without batch normalization. The neural network converged quite quickly as well. Batch normalization improved the results whenever it was used.
