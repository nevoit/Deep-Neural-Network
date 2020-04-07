import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Parameters
seed = 0
eps = 1e-5
beta = 0.5
gamma = 0.5
keep_prob = 0.8


# ------------ SECTION 1: FORWARD PROPAGATION ------------
def initialize_parameters(layer_dims) -> dict:
    """
    :param layer_dims: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened
     input, layer L is the output softmax)
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    np.random.seed(seed)
    weights_name = 'W'
    bias_name = 'b'
    parameters = {
        weights_name: {},
        bias_name: {}
    }
    for layer_index, current_layer_dim in enumerate(layer_dims[:-1], start=1):
        layer_size = layer_dims[layer_index]
        prev_layer_size = layer_dims[layer_index - 1]
        w_layer = np.random.randn(layer_size, prev_layer_size) / np.sqrt(prev_layer_size)
        b_layer = np.zeros((layer_size, 1))
        assert (w_layer.shape == (layer_size, prev_layer_size))
        assert (b_layer.shape == (layer_size, 1))
        parameters[weights_name][layer_index] = w_layer
        parameters[bias_name][layer_index] = b_layer
    return parameters


def linear_forward(A, W, b) -> (np.ndarray, dict):
    """
    Implement the linear part of a layer's forward propagation.
    :param A: the activations of the previous layer
    :param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    :param b: the bias vector of the current layer (of shape [size of current layer, 1])
    :return: Z – the linear component of the activation function
    (i.e., the value before applying the non-linear function)
    linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)
    """
    linear_cache = {
        'A': A,
        'W': W,
        'b': b
    }
    current_layer_size = W.shape[0]
    batch_size = A.shape[1]
    Z = np.dot(W, A) + b
    assert (Z.shape == (current_layer_size, batch_size))
    return Z, linear_cache


def softmax(Z) -> (np.ndarray, np.ndarray):
    """
    Softmax can be thought of as a sigmoid for multi-class problems. The formula for softmax for each node in the
    output layer is as follows: Softmax(z_i) = exp(z_i) / sum(exp(z_j))
    :param Z: the linear component of the activation function
    :return: A – the activations of the layer
    activation_cache – returns Z, which will be useful for the back propagation
    """
    activation_cache = Z
    batch_size = Z.shape[1]
    last_layer_size = Z.shape[0]
    Z = Z - Z.max(axis=0, keepdims=True)
    y = np.exp(Z)
    A = y / y.sum(axis=0, keepdims=True)
    assert (A.shape == (last_layer_size, batch_size))
    return A, activation_cache


def relu(Z) -> (np.ndarray, np.ndarray):
    """
    :param Z: the linear component of the activation function
    :return: A – the activations of the layer
    activation_cache – returns Z, which will be useful for the back propagation
    """
    activation_cache = Z
    batch_size = Z.shape[1]
    current_layer_size = Z.shape[0]
    A = Z * (Z > 0)
    assert (A.shape == (current_layer_size, batch_size))
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation) -> (np.ndarray, dict):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations of the previous layer
    :param W: the weights matrix of the current layer
    :param B: the bias vector of the current layer
    :param activation: the activation function to be used (a string, either “softmax” or “relu”)
    :return: A – the activations of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache
    """
    Z, linear_cache = linear_forward(A=A_prev, W=W, b=B)
    if activation == "relu":
        A, activation_cache = relu(Z=Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z=Z)
    else:
        raise Exception("Wrong activation function")
    current_layer_size = W.shape[0]
    batch_size = A_prev.shape[1]
    assert (A.shape == (current_layer_size, batch_size))
    cache = {'linear_cache': linear_cache, 'activation_cache': activation_cache}
    return A, cache


def L_model_forward(X, parameters, use_batchnorm, use_dropout) -> (np.ndarray, list):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
    :param X: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation (note that
     this option needs to be set to “false” in Section 3 and “true” in Section 4).
    :return: AL – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function
    """
    AL = X
    weights = parameters['W']
    bias = parameters['b']
    caches = []
    layers_size = len(bias)
    for layer in range(1, layers_size + 1):
        activation = "relu" if not layer == layers_size else "softmax"
        AL, cache = linear_activation_forward(A_prev=AL, W=weights[layer], B=bias[layer], activation=activation)
        if layer != layers_size:
            if use_batchnorm:
                AL = apply_batchnorm(A=AL)
            if use_dropout:
                AL = apply_dropout(A=AL)
        caches.append(cache)
    batch_size = X.shape[1]
    output_layer_size = weights[layers_size].shape[0]
    assert (AL.shape == (output_layer_size, batch_size))
    return AL, caches


def compute_cost(AL, Y) -> float:
    """
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss.
    :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return: cost – the cross-entropy cost
    """
    numerator = - 1 * (np.dot(Y, np.log(AL.T + eps)))
    numerator = np.trace(numerator)
    denominator = (Y.shape[1])
    cost = numerator / denominator
    return cost


def apply_batchnorm(A) -> np.ndarray:
    """
    performs batchnorm on the received activation values of a given layer.
    :param A: the activation values of a given layer
    :return: NA - the normalized activation values, based on the formula learned in class
    """
    A_mean_per_sample = A.mean(axis=0)
    A_var_per_sample = A.var(axis=0)
    A_std_per_sample = np.sqrt(A_var_per_sample + eps)
    A_norm = A - A_mean_per_sample / A_std_per_sample
    NA = gamma * A_norm + beta
    assert (NA.shape == A.shape)
    return NA


def apply_dropout(A) -> np.ndarray:
    """
    This function activates the dropout probability
    :param A:
    :return:
    """
    np.random.seed(seed=seed)
    random_df = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
    A_dropout = np.multiply(A, random_df)
    A_dropout /= keep_prob  # To not change the expected value of Z
    assert (A_dropout.shape == A.shape)
    return A_dropout


# ------------ SECTION 2: BACKWARD PROPAGATION ------------

def linear_backward(dZ, cache) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Implements the linear part of the backward propagation process for a single layer
    :param dZ: the gradient of the cost with respect to the linear output of the current layer (layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return: dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache.values()
    batch_size = A_prev.shape[1]

    dW = (1 / batch_size) * np.dot(dZ, A_prev.T)
    db = np.array(1 / batch_size * (np.sum(dZ, axis=1)))
    dA_prev = np.dot(W.T, dZ)

    last_layer_size = W.shape[0]
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == (last_layer_size,))

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then
    applies the linear_backward function.
    :param dA: post activation gradient of the current layer
    :param cache: contains both the linear cache and the activations cache
    :param activation:
    :return: dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    dA_prev = None
    dW = None
    db = None
    dZ = None
    if activation == "relu":
        linear_cache, activation_cache = cache.values()
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        linear_cache, activation_cache, Y = cache.values()
        dZ = softmax_backward(dA, Y)
    else:
        raise Exception("Wrong activation function")
    assert (dZ.shape == dA.shape)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def relu_backward(dA, activation_cache) -> np.ndarray:
    """
    Implements backward propagation for a ReLU unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == dA.shape)
    return dZ


def softmax_backward(dA, activation_cache) -> np.ndarray:
    """
    Implements backward propagation for a softmax unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: gradient of the cost with respect to Z
    """
    Y = activation_cache
    dZ = dA - Y
    assert (dZ.shape == dA.shape)
    return dZ


def L_model_backward(AL, Y, caches) -> dict:
    """
    Implement the backward propagation process for the entire network.
    the backpropagation for the softmax function should be done only once as only the output layers uses it and the
    RELU should be done iteratively over all the remaining layers of the network.
    :param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y: the true labels vector (the "ground truth" - true classifications)
    :param caches: list of caches containing for each layer: a) the linear cache; b) the activation cache
    :return: Grads - a dictionary with the gradients
    """
    grads = {}
    layers_num = len(caches)  # the number of layers
    last_layer = layers_num - 1
    current_cache = caches[layers_num - 1]

    # Initializing the backpropagation
    dA = AL
    grads["dA"] = {}
    grads["dW"] = {}
    grads["db"] = {}

    current_cache['Y'] = Y
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA=dA,
                                                                cache=current_cache,
                                                                activation="softmax")
    grads["dA"][last_layer] = dA_prev_temp
    grads["dW"][last_layer] = dW_temp
    grads["db"][last_layer] = db_temp
    for l in reversed(range(layers_num - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA=dA_prev_temp,
                                                                    cache=current_cache,
                                                                    activation="relu")
        grads["dA"][l] = dA_prev_temp
        grads["dW"][l] = dW_temp
        grads["db"][l] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate) -> dict:
    """
    Updates parameters using gradient descent
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :param grads: a python dictionary containing the gradients (generated by L_model_backward)
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :return: parameters – the updated values of the parameters object provided as input
    """
    L = len(parameters['W'])
    for l in range(L):  # Update rule for each parameter. Use a for loop.
        parameters["W"][l + 1] = parameters["W"][l + 1] - learning_rate * grads["dW"][l]
        parameters["b"][l + 1] = parameters["b"][l + 1] - learning_rate * grads["db"][l].reshape(
            grads["db"][l].shape[0], 1)

    return parameters


# ------------ SECTION 3: TRAINING THE NETWORK ------------
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm, use_dropout=False,
                  verbose=1, validation_size=0.2, improvement_threshold=0.0001, iteration_criteria=100) -> (
        dict, list, float, float, float):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function, and the
    final layer will apply the softmax activation function. The size of the output layer should be equal to the number
    of labels in the data. Please select a batch size that enables your code to run well (i.e. no memory overflows
    while still running relatively fast).
    Hint: the function should use the earlier functions in the following order:
    initialize -> L_model_forward -> compute_cost -> L_model_backward -> update parameters
    :param iteration_criteria:
    :param improvement_threshold:
    :param validation_size:
    (validation_size * 100) % of the training set as a validation set (samples need to be randomly chosen).
    :param verbose: 0, means silent value greater than 1 means one line per 100 iterations
    :param use_dropout: bool - TRUE activates the dropout functionality
    :param use_batchnorm: bool - TRUE performs batchnorm on the received activation values of a given layer
    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    Comment: since the input is in grayscale we only have height and width, otherwise it would have been height*width*3
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: the number of examples in a single training batch.
    :param num_iterations:
    :param batch_size:
    :return: parameters – the parameters learnt by the system during the training (the same parameters that were
    updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved after
    each 100 training iterations (e.g. 3000 iterations -> 30 values).
    """
    np.random.seed(seed)
    # Create random validation data
    x_train, x_val, y_train, y_val = train_test_split(X, Y.T, test_size=validation_size, random_state=seed)
    x_train, x_val, y_train, y_val = x_train.T, x_val.T, y_train.T, y_val.T

    costs = []
    row_num = x_train.shape[1]
    parameters = initialize_parameters(layer_dims=layers_dims)
    batch_loops = int(row_num / batch_size)
    epochs = int(num_iterations / batch_loops)
    stop_training = False
    prev_val_cost = 9999
    train_step = 0

    for ep in range(epochs):
        for i in range(batch_loops):
            # TODO: To shuffle the x_train and y_train. Should we random the array in the same order each epoch?
            starting_index_batch = i * batch_size
            ending_index_batch = (i + 1) * batch_size
            x_batch = x_train[:, starting_index_batch: ending_index_batch]
            y_batch = y_train[:, starting_index_batch: ending_index_batch]
            train_step += 1
            AL_batch, caches_batch = L_model_forward(X=x_batch,
                                                     parameters=parameters,
                                                     use_batchnorm=use_batchnorm,
                                                     use_dropout=use_dropout)
            grads_batch = L_model_backward(AL=AL_batch, Y=y_batch, caches=caches_batch)
            parameters = update_parameters(parameters=parameters, grads=grads_batch, learning_rate=learning_rate)
            if train_step % iteration_criteria == 0:
                accuracy_val, cost_val, diff_val_cost, training_cost = check_stop_criteria(AL_batch, costs, parameters,
                                                                                           prev_val_cost, use_batchnorm,
                                                                                           use_dropout, x_val, y_batch,
                                                                                           y_val)
                if verbose > 0:
                    print(f'Validation accuracy: {accuracy_val:.3f},'
                          f' Validation Cost: {cost_val:.3f} (Diff: {diff_val_cost:.3f}),'
                          f' Training Steps {train_step},'
                          f' Training Cost: {training_cost:.3f}')
                if not improvement_threshold <= diff_val_cost:
                    stop_training = True
                    break
                prev_val_cost = cost_val
        if stop_training:
            break
    if verbose > 0:
        print('Training is done.')
    train_accuracy = predict(x_train, y_train, parameters, use_batchnorm)
    val_accuracy = predict(x_val, y_val, parameters, use_batchnorm)
    return parameters, costs, train_accuracy, val_accuracy, train_step


def check_stop_criteria(AL_batch, costs, parameters, prev_val_cost, use_batchnorm, use_dropout, x_val, y_batch, y_val):
    """
    This function is called after each iteration and checks the current accuracy of the model on the validation set and
    the current cost (i.e. the result of the loss function) on the
    training data. We do this so we can stop the training if there is no significant improvement.
    :param AL_batch:
    :param costs:
    :param parameters:
    :param prev_val_cost:
    :param use_batchnorm:
    :param use_dropout:
    :param x_val:
    :param y_batch:
    :param y_val:
    :return:
    """
    training_cost = compute_cost(AL_batch, y_batch)
    costs.append(training_cost)
    accuracy_val = predict(x_val, y_val, parameters, use_batchnorm)
    AL_val, caches_val = L_model_forward(X=x_val, parameters=parameters, use_batchnorm=use_batchnorm,
                                         use_dropout=use_dropout)
    cost_val = compute_cost(AL_val, y_val)
    diff_val_cost = prev_val_cost - cost_val
    return accuracy_val, cost_val, diff_val_cost, training_cost


def predict(X, Y, parameters, use_batchnorm=False):
    """
    The function receives an input data and the true labels and calculates the accuracy of the trained neural network
    on the data.
    :param use_batchnorm: bool - TRUE performs batchnorm on the received activation values of a given layer
    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return: accuracy – the accuracy measure of the neural net on the provided data (i.e. the percentage of the samples
    for which the correct label receives the hughest confidence score). Use the softmax function to normalize the
    output values.
    """
    pred_score, cache = L_model_forward(X=X, parameters=parameters, use_batchnorm=use_batchnorm, use_dropout=None)
    # We do not want to use dropout while the prediction
    Y = np.argmax(Y, axis=0)
    return compute_accuracy(Y, pred_score)


def compute_accuracy(Y, pred_score) -> float:
    """
    Returns the accuracy score of the prediction compared to the true classes.
    :param Y:
    :param pred_score:
    :return:
    """
    pred_class = pred_score.argmax(axis=0)
    accuracy = np.mean(pred_class == Y)
    return accuracy