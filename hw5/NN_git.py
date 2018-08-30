import numpy as np

LEARNING_RATE = 0.1  # Default learning rate
DEFAULT_ACTIVATION = 'logistic'
ITERATION = 1000
HIDDEN_LAYER_SIZES = 100
THRESHOLD = 0.5  # Threshold for logistic activation function.
TOLERANCE = 1e-6  # Threshold of output delta for neural network converge.
CONSECUTIVE_TIMES = 10  # How many consecutive times the output delta less than tolerance to stop the training.


def read_pgm(filepath):
    with open(filepath, 'rb') as pgmf:
        P5 = pgmf.readline()
        comment = pgmf.readline()
        width, height = [int(value) for value in pgmf.readline().split()]
        scale = int(pgmf.readline())  # Must be less than 65536, and more than zero
        image = list()
        for _ in range(width * height):
            image.append(ord(pgmf.read(1)) / scale)
        return image


def readData(filepath):
    images = list()
    labels = list()
    with open(filepath) as f:
        for image_filepath in f.readlines():
            # image_filepath = 'Data/' + image_filepath.strip()
            image_filepath = image_filepath.strip()
            images.append(read_pgm(image_filepath))  # returns image data
            labels.append(1 if 'down' in image_filepath else 0)  # label is 1 if 'down' in file name
    return np.array(images), np.array(labels)


class NeuralNetwork(object):
    '''
    classdocs

    A NeuralNetwork class is implemented and provide a lot of value sets for reference.
    function lists:
    0. __init__(self, hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=DEFAULT_ACTIVATION, iteration=ITERATION, learning_rate=LEARNING_RATE, training_data=None, training_data_label=None, weight_low=0, weight_high=1, enable_binary_classification=True):
        NeuralNetwork class constructor.
        weight matrix initialization range will be [weight_low, weight_high), default is [0,1)
        enable_binary_classification will transform the output to binary classification if the value set to True. Default is True.
    1. logistic(self, x):
        Set logistic function as activation function.
    2. logistic_derivation(self, logistic_x):
        The derivation of logistic function. The input is the result of logistic(x).
    3. tanh(self, x):
        Set hyperbolic tangent function as activation function.
    4. tanh_derivation(self,tanh_x):
        The derivation of hyperbolic tangent function. The input is the result of tanh(x)
    5. initial_weights(self, network_layer_sizes):
        Set up the whole network layer. layer_sizes = [input_dimensions, hidden_layer1_sizes, ..., output_layer_sizes].
    6. set_layer_sizes(self, training_data, training_data_label):
        Construct the whole neural network structure, include [input layer sizes, hidden layer 1 sizes, ...hidden layer L sizes, output layer sizes]
        The input number of nodes/dimension and output number of nodes / dimensions will automatically define by training_data and training_data_label respectively.
    7. feed_forward(self, input_data):
        Neural Network feed forward propagation. It will return the calculation result of last/output layer which support multiple dimension.
        The output dimension will automatically adjust the dimension to fit with the dimensions of training data label.
    8. back_propagate(self, output, label_data):
        According to output data to perform back propagation on delta and weight array/matrix.
    9. predict(self, x):
        Return predict array to support multiple dimension results. The function also support output data transform to binary classification if the feature sets to True.

    '''

    def __init__(self, hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=DEFAULT_ACTIVATION, iteration=ITERATION,
                 learning_rate=LEARNING_RATE,
                 training_data=None, training_data_label=None, weight_low=0, weight_high=1,
                 enable_binary_classification=True, tol=TOLERANCE):
        '''
        Constructor
        '''

        self.hidden_layer_sizes = np.array(hidden_layer_sizes)

        self.weights = np.array
        self.max_iteration = iteration
        self.learning_rate = learning_rate
        self.training_data = np.array(training_data)
        self.input_numbers = 0
        self.input_dimensions = 0
        self.training_data_label = np.array(training_data_label)
        self.out_numbers = 0
        self.output_dimensions = 0
        self.X = []
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.enable_binary_classification = enable_binary_classification
        self.tol = tol

        self.network_layer_sizes = np.array

        self.activation = self.logistic
        self.activation_derivation = self.logistic_derivation

        if (self.training_data.ndim != 0 and self.training_data_label.ndim != 0):
            self.execute(training_data, training_data_label)
        else:
            pass

    def logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def logistic_derivation(self, logistic_x):
        return logistic_x * (1.0 - logistic_x)

    def error_term_derivation(self, x, y):
        return 2 * (x - y)

    def initial_weights(self, network_layer_sizes):
        # network_layer_sizes = [input_dimensions, hidden_layer1_sizes, ..., output_layer_sizes].
        # range of weight values (-1,1)
        # self.weights = weights array = [[Weights of level-01], [Weights of level-12], ..., [Weights of level-(L-1)(L)]].
        # For [Weights of level-01]=[[w01, w02, ..., w0d], [w11, w12, ..., w1d], ... [wd1, wd2, ..., wdd]]

        weights = []
        scale = 0  # optimal scale of weight is m**(-1/2)
        for l in range(1, len(network_layer_sizes)):
            scale = (network_layer_sizes[l - 1]) ** (-1 / 2)
            w = ((self.weight_high) - (self.weight_low)) * np.random.normal(
                size=(network_layer_sizes[l - 1], network_layer_sizes[l])) + (self.weight_low)
            w = np.array(w)
            weights.append(w)
            print('W', w.shape)
            np.random.random
        self.weights = np.array(weights)
        # print(self.weights.shape)
        # print(self.weights)
        return self.weights

    def set_layer_sizes(self, training_data, training_data_label):
        # Construct the whole neural network structure, include [input layer sizes, hidden layer 1 sizes, ...hidden layer L sizes, output layer sizes]
        dim = 0
        network_layer_sizes = []
        dim = training_data.ndim;
        if dim != 0:
            self.input_numbers, self.input_dimensions = training_data.shape
        else:
            pass
        dim = training_data_label.ndim;
        if dim != 0:
            if dim == 1:
                self.output_numbers = training_data_label.shape[0]
                self.output_dimensions = 1;
            else:
                self.output_numbers, self.output_dimensions = training_data_label.shape
        else:
            pass

        network_layer_sizes.append(self.input_dimensions + 1)  # add X0

        for i in self.hidden_layer_sizes:
            network_layer_sizes.append(i)

        network_layer_sizes.append(self.output_dimensions)
        self.network_layer_sizes = np.array(network_layer_sizes)

        return self.network_layer_sizes

    def feed_forward(self, input_data):
        X = [np.concatenate((np.ones(1).T, np.array(input_data)), axis=0)]  # add bias unit [array([])]
        W = self.weights
        wijxi = []
        xj = []

        for l in range(0, len(W)):
            wijxi = np.dot(X[l], W[l])
            xj = self.activation(wijxi)
            # Setup bias term for each hidden layer, x0=1
            if l < len(W) - 1:
                xj[0] = 1
            X.append(xj)

        self.X = np.array(X)
        return X[-1]  # return the feed forward result of final level.

    def back_propagate(self, output, label_data):
        X = self.X
        W = list(
            self.weights)  # self.weights=<class list>[array([ndarray[100],ndarray[100],...X961]), array(ndarray[1],ndarray[1],...X100)]
        avg_err = []
        _Delta = []
        x = []
        d = []
        w = []
        y = []

        y = np.atleast_2d(label_data)
        x = np.atleast_2d(output)
        # Base level L delta calculation.
        avg_err = np.average(x - y)
        _Delta = [self.error_term_derivation(x, y) * self.activation_derivation(
            x)]  # Delta = error term derivation * activation function derivation
        # #<class list>[array([])]

        # Calculate all deltas and adjust weights
        for l in range(len(X) - 2, 0, -1):
            d = np.atleast_2d(_Delta[-1])
            x = np.atleast_2d(X[l])
            w = np.array(W[l])

            _Delta.append(self.activation_derivation(x) * _Delta[-1].dot(w.T))
            W[l] = W[l] - self.learning_rate * x.T.dot(d)

        # Calculate the weight of input layer and update weight array
        x = np.atleast_2d(X[l - 1])
        d = np.atleast_2d(_Delta[-1])
        W[l - 1] = W[l - 1] - self.learning_rate * x.T.dot(d)

        self.weights = W
        return avg_err

    def predict(self, x):
        r = []
        r = self.feed_forward(x[0])
        print("\nact2=\n{a}".format(a=r))
        enable_binary_classification = self.enable_binary_classification

        # Enable the binary classification on predict results.
        if enable_binary_classification and self.activation == self.logistic:
            for i in range(len(r)):
                if r[i] >= THRESHOLD:
                    r[i] = 1
                else:
                    r[i] = 0
        else:
            pass
        return r

    def execute(self, training_data, training_data_label):
        '''
        Execute function to train the neural network.
        '''
        self.training_data = np.array(training_data)
        self.training_data_label = np.array(training_data_label)
        network_layer_sizes = self.set_layer_sizes(self.training_data, self.training_data_label)
        max_iter = self.max_iteration
        input_numbers = self.input_numbers
        avg_err = 0
        counter = 0

        self.initial_weights(network_layer_sizes)

        # Execute training.
        for idx in range(0, max_iter):
            i = np.random.randint(self.training_data.shape[0])
            _result = self.feed_forward(training_data[i])
            avg_err = self.back_propagate(_result, training_data_label[i])
            if abs(avg_err) <= self.tol:
                counter += 1
                if counter >= CONSECUTIVE_TIMES:
                    break
                else:
                    pass
            else:
                counter = 0
        print('Neural Network Converge at iteration =', idx + 1)
        print('Total input numbers=', input_numbers)

    def score(self, X, Y):
        prediction = self.predict(X)
        accuracy = (prediction == Y).sum().astype(float) / len(Y)
        print("\nprediction=\n{a}".format(a=prediction))
        print("\nY=\n{a}".format(a=Y))
        return accuracy

'''
Main program for the NeuralNetwork class execution.
'''

if __name__ == '__main__':
    # read train data list
    train_images, train_labels = readData('downgesture_train.list')

    nn = NeuralNetwork(hidden_layer_sizes=[100, ], activation='logistic', learning_rate=0.1, iteration=1000,
                       weight_low=0, weight_high=1, enable_binary_classification=True)
    nn.execute(train_images, train_labels)

    # read test data list
    test_images, test_labels = readData('downgesture_test.list')
    print("\nscore=\n{a}".format(a=nn.score(test_images, test_labels)))
