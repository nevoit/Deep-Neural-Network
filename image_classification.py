import time
import numpy as np
import pandas as pd
from keras.datasets import mnist
from deep_neural_network import L_layer_model
from deep_neural_network import predict
import deep_neural_network

"""
In this assignment we were tasked with creating a Deep Neural Network that can classify images from the MNIST database.
The data was loaded through the keras API for Python.
As was specified, 20% of the training data was used as a validation set in order to track our progress between training iterations.
The input layer receives a single 1d array, so the images had to be slightly manipulated.
Since each image is 28*28 pixels, a total of 738 pixels, we reshaped each image into a 1d array.
This array serves as the input to the network.
Additionally, this is a multi-class problem with 10 possible classes, so the output layer has 10 neurons.
Therefore, the architecture of the network is [784, 20, 7, 5, 10] neurons (the number of neurons in the hidden layers is taken from the assignment guidelines).
The learning rate is 0.009, and we used batch sizes of 512 and up to 100,000 iterations (around 1066 epochs).
The size of the training set is 48,000 samples (i.e. 94 iterations per epoch). We found success with these parameters.
"""

def image_classification():
    """
    This function loads the mnist dataset (a large database of handwritten digits) to validate our DNN.
    Note that there is a predefined division between the train and test set.
    We Use 20% of the training set as a validation set (samples need to be randomly chosen).
    We use 4 layers (aside from the input layer), with the following sizes: 20,7,5,10
    The input at each iteration flattened to a matrix of [m,784], where m is the number of samples
    We use a learning rate of 0.009
    The stopping criterion is when there is no improvement on the validation set (or the improvement is very small)
    for 100 training steps (iterations).
    """
    # ----------- Load The Data and Prepossessing -----------
    layers_dims, x_test, x_train, y_test, y_train = load_mnist_data()
    # ----------- / Load The Data and Prepossessing -----------

    # ----------- Parameters -----------
    verbose = 1
    learning_rate = 0.009
    batch_size = 512
    num_of_iterations = 10 ** 5
    beta = 0.1
    gamma = 1
    keep_prob = 0.9
    seed = 2
    use_batchnorm = False
    use_dropout = False
    validation_size = 0.2
    improvement_threshold = 0.0001
    iteration_criteria = 100
    # ----------- / Parameters -----------

    deep_neural_network.keep_prob = keep_prob
    deep_neural_network.beta = beta
    deep_neural_network.gamma = gamma
    deep_neural_network.seed = seed

    start_time = time.time()
    parameters, costs, train_accuracy, val_accuracy, train_step = L_layer_model(X=x_train,
                                                                                Y=y_train,
                                                                                layers_dims=layers_dims,
                                                                                learning_rate=learning_rate,
                                                                                num_iterations=num_of_iterations,
                                                                                batch_size=batch_size,
                                                                                use_batchnorm=use_batchnorm,
                                                                                use_dropout=use_dropout,
                                                                                verbose=verbose,
                                                                                validation_size=validation_size,
                                                                                improvement_threshold=improvement_threshold,
                                                                                iteration_criteria=iteration_criteria)

    training_time = time.time() - start_time
    start_time = time.time()
    testing_accuracy = predict(x_test.T, y_test, parameters, use_batchnorm)
    testing_time = time.time() - start_time
    print(f'Training Accuracy: {train_accuracy:.3f}')
    print(f'Training Time: {training_time:.3f}')
    print(f'Training Steps: {train_step:.3f}')
    print(f'Costs: {costs}')
    print(f'Validation Accuracy: {val_accuracy:.3f}')
    print(f'Testing Accuracy: {testing_accuracy:.3f}')
    print(f'Testing Time: {testing_time:.3f}')


def image_classification_experiments():
    """
    A function for loading the mnist dataset used in this assignment.
    Additionally, here we set the various parameters used for the DNN and it wraps the function that trains and runs the experiment.
    Here we experimented with various batch sizes and iterations.
    This function loads the mnist dataset (a large database of handwritten digits) to validate our DNN.
    Note that there is a predefined division between the train and test set.
    We Use 20% of the training set as a validation set (samples need to be randomly chosen).
    We use 4 layers (aside from the input layer), with the following sizes: 20,7,5,10
    The input at each iteration flattened to a matrix of [m,784], where m is the number of samples
    We use a learning rate of 0.009
    The stopping criterion is when there is no improvement on the validation set (or the improvement is very small)
    for 100 training steps (iterations).
    """
    # ----------- Load The Data and Prepossessing -----------
    layers_dims, x_test, x_train, y_test, y_train = load_mnist_data()
    # ----------- / Load The Data and Prepossessing -----------

    # ----------- Parameters -----------
    output_file_name = 'results.csv'
    verbose = 1
    learning_rate = 0.009
    batch_sizes_list = [512]
    num_of_iterations = [10 ** 5]
    beta = [0.1]
    gamma = [1]
    keep_prob = [0.5, 0.7, 0.8, 0.9]
    seeds_list = range(10)
    # use_batchnorm = [False, True]
    use_batchnorm = [False, True]
    use_dropout = [False, True]
    validation_size = 0.2
    improvement_threshold = 0.0001
    iteration_criteria = 100
    # ----------- / Parameters -----------

    run_experiments(batch_sizes_list, layers_dims, learning_rate, num_of_iterations, output_file_name, seeds_list,
                    use_batchnorm, use_dropout, verbose, x_test, x_train, y_test, y_train, validation_size,
                    improvement_threshold, iteration_criteria, beta, gamma, keep_prob)


def load_mnist_data() -> (list, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    The image_classification_experiments function calls this function to load the dataset through the keras API.
     In addition, this function normalizes the training data by dividing each cell by 255 since the pixels are grayscale.
      This function also reshapes the input, which is 2d arrays of size 28*28 to a 1d array of 784.
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_num = 784
    output_num = 10
    norm_colors = 255
    layers_dims = np.array([input_num, 20, 7, 5, output_num])
    x_train = x_train.reshape(x_train.shape[0], input_num)  # building the input vector from the 28x28 pixels
    x_test = x_test.reshape(x_test.shape[0], input_num)
    x_train = x_train.astype('float')
    x_test = x_test.astype('float')
    x_train /= norm_colors  # for grayscale
    x_test /= norm_colors  # for grayscale
    y_train = np.array(pd.get_dummies(y_train))
    y_train = y_train.transpose()
    y_test = np.array(pd.get_dummies(y_test))
    y_test = y_test.transpose()
    return layers_dims, x_test, x_train, y_test, y_train


def run_experiments(batch_sizes, layers_dims, learning_rate, num_of_iterations, output_file_name, seeds_list,
                    use_batchnorm, use_dropout, verbose, x_test, x_train, y_test, y_train, validation_size,
                    improvement_threshold, iteration_criteria, beta, gamma, keep_prob):
    """
    This function loops through different possible hyperparameter values and calls run_comb() with the appropriate parameters.
     Finally, it saves a csv file with the results of each run.
    :param beta:
    :param gamma:
    :param keep_prob:
    :param batch_sizes:
    :param layers_dims:
    :param learning_rate:
    :param num_of_iterations:
    :param output_file_name:
    :param seeds_list:
    :param use_batchnorm:
    :param use_dropout:
    :param verbose:
    :param x_test:
    :param x_train:
    :param y_test:
    :param y_train:
    :param validation_size:
    :param improvement_threshold:
    :param iteration_criteria:
    :return:
    """
    header = True
    for batchnorm in use_batchnorm:
        for dropout in use_dropout:
            for size_of_batch in batch_sizes:
                for iteration_size in num_of_iterations:
                    for s in seeds_list:
                        results = {
                            'Batch Normalization': [],
                            'Dropout': [],
                            'Batch Size': [],
                            'Number Of Iterations': [],
                            'Training Accuracy': [],
                            'Validation Accuracy': [],
                            'Testing Accuracy': [],
                            'Training Time': [],
                            'Testing Time': [],
                            'Costs': [],
                            'Training Iterations': [],
                            'Seed': [],
                            'beta': [],
                            'gamma': [],
                            'keep_prob': []
                        }
                        if batchnorm and dropout:
                            run_comb(batchnorm, beta, dropout, gamma, improvement_threshold, iteration_criteria,
                                     iteration_size, keep_prob, layers_dims, learning_rate, results, s, size_of_batch,
                                     validation_size, verbose, x_test, x_train, y_test, y_train)
                        elif batchnorm:
                            run_comb(batchnorm, beta, dropout, gamma, improvement_threshold, iteration_criteria,
                                     iteration_size, [None], layers_dims, learning_rate, results, s, size_of_batch,
                                     validation_size, verbose, x_test, x_train, y_test, y_train)
                        elif dropout:
                            run_comb(batchnorm, [None], dropout, [None], improvement_threshold, iteration_criteria,
                                     iteration_size, keep_prob, layers_dims, learning_rate, results, s, size_of_batch,
                                     validation_size, verbose, x_test, x_train, y_test, y_train)
                        else:
                            run_comb(batchnorm, [None], dropout, [None], improvement_threshold, iteration_criteria,
                                     iteration_size, [None], layers_dims, learning_rate, results, s, size_of_batch,
                                     validation_size, verbose, x_test, x_train, y_test, y_train)
                        df = pd.DataFrame.from_dict(results)
                        df.to_csv(output_file_name, mode='a', header=header)
                        header = False


def run_comb(batchnorm, beta, dropout, gamma, improvement_threshold, iteration_criteria, iteration_size, keep_prob,
             layers_dims, learning_rate, results, s, size_of_batch, validation_size, verbose, x_test, x_train, y_test,
             y_train):
    """
    This function receives the parameters and trains the DNN with the different possible combinations.
    It also times the time it takes to train the DNN and predicts on the test set.
    Lastly, it appends the results to a csv file that we can check later to see the performance of the DNN with different parameters.
    :param batchnorm:
    :param beta:
    :param dropout:
    :param gamma:
    :param improvement_threshold:
    :param iteration_criteria:
    :param iteration_size:
    :param keep_prob:
    :param layers_dims:
    :param learning_rate:
    :param results:
    :param s:
    :param size_of_batch:
    :param validation_size:
    :param verbose:
    :param x_test:
    :param x_train:
    :param y_test:
    :param y_train:
    :return:
    """
    for b in beta:
        for g in gamma:
            for p in keep_prob:
                deep_neural_network.keep_prob = p
                deep_neural_network.beta = b
                deep_neural_network.gamma = g
                deep_neural_network.seed = s
                start_time = time.time()
                parameters, costs, train_accuracy, val_accuracy, train_step = L_layer_model(X=x_train,
                                                                                            Y=y_train,
                                                                                            layers_dims=layers_dims,
                                                                                            learning_rate=learning_rate,
                                                                                            num_iterations=iteration_size,
                                                                                            batch_size=size_of_batch,
                                                                                            use_batchnorm=batchnorm,
                                                                                            use_dropout=dropout,
                                                                                            verbose=verbose,
                                                                                            validation_size=validation_size,
                                                                                            improvement_threshold=improvement_threshold,
                                                                                            iteration_criteria=iteration_criteria)

                training_time = time.time() - start_time
                start_time = time.time()
                testing_accuracy = predict(x_test.T, y_test, parameters, batchnorm)
                testing_time = time.time() - start_time
                add_to_results(size_of_batch, costs, s, iteration_size, results, testing_accuracy, testing_time,
                               train_accuracy, train_step,
                               batchnorm, dropout, val_accuracy, training_time, p, b, g)


def add_to_results(size_of_batch, costs, i, itr, results, testing_accuracy, testing_time, train_accuracy, train_step, use_batchnorm,
                   use_dropout, val_accuracy, training_time, p, b, g):
    """
    Appends to a file containing all the results of the previous runs the new run just completed.
    Each run is a complete training iteration that was halted because there was no improvement in the accuracy.
    We do this so we could run hundreds of experiments and find good values for batch sizes and iterations.
    :param b:
    :param costs:
    :param i:
    :param itr:
    :param results:
    :param testing_accuracy:
    :param testing_time:
    :param train_accuracy:
    :param train_step:
    :param use_batchnorm:
    :param use_dropout:
    :param val_accuracy:
    :param training_time:
    :return:
    """
    results['Batch Normalization'].append(use_batchnorm)
    results['Dropout'].append(use_dropout)
    results['Batch Size'].append(size_of_batch)
    results['Number Of Iterations'].append(itr)
    results['Training Accuracy'].append(train_accuracy)
    results['Validation Accuracy'].append(val_accuracy)
    results['Testing Accuracy'].append(testing_accuracy)
    results['Training Time'].append(training_time)
    results['Testing Time'].append(testing_time)
    results['Costs'].append(costs)
    results['Training Iterations'].append(train_step)
    results['Seed'].append(i)
    results['beta'].append(b)
    results['gamma'].append(g)
    results['keep_prob'].append(p)


if __name__ == '__main__':
    # image_classification_experiments()
    image_classification()
