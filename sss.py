import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(2, 3),
                 datapath='/kaggle/input/cifer10/cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = np.random.uniform(-1/np.sqrt(all_dims[layer_n-1]),1/np.sqrt(all_dims[layer_n-1]),(all_dims[layer_n-1],all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            print(f"W{layer_n}")
            print(np.shape(self.weights[f"W{layer_n}"]))

    def relu(self, x, grad=False):
        numpy_x= np.array(x)
        if grad:
            return np.where(numpy_x <= 0, 0, 1)
        return np.maximum(0, numpy_x)

    def sigmoid(self, x, grad=False):
        numpy_x=np.array(x)
        if grad:
            return (1/(1 + np.exp(-numpy_x)))*(1-(1/(1 + np.exp(-numpy_x))))
        return 1/(1 + np.exp(-numpy_x)) 

    def tanh(self, x, grad=False):
        numpy_x=np.array(x)
        if grad: 
            return 1- ((np.exp(numpy_x)-np.exp(-numpy_x))/(np.exp(numpy_x) + np.exp(-numpy_x))*(np.exp(numpy_x)-np.exp(-numpy_x))/(np.exp(numpy_x) + np.exp(-numpy_x)))
        return (np.exp(numpy_x)-np.exp(-numpy_x))/(np.exp(numpy_x) + np.exp(-numpy_x)) 

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x,grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x,grad)
        elif self.activation_str == "tanh":
            return self.tanh(x,grad)
        else:
            raise Exception("invalid")
        return 0

    def softmaxSingle(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum() 

    def softmax(self, x):
        numpy_x= np.array(x)
        if numpy_x.ndim>1:
            result=[[]]*np.shape(x)[0]
            for i in range(len(result)):
                result[i]=list(self.softmaxSingle(x[i]))
            return list(result)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, x):
        cache = {"Z0": x}
        if self.seed is not None:
            np.random.seed(self.seed)
        for layer_n in range(1, self.n_hidden + 2):
            layer_nmo=layer_n-1
            cache[f"A{layer_n}"] = np.add(np.transpose(np.transpose(self.weights[f"W{layer_n}"]).dot(np.transpose(cache[f"Z{layer_nmo}"]))),np.repeat(self.weights[f"b{layer_n}"],len(x),axis=0))
            if layer_n == self.n_hidden + 1:
                cache[f"Z{layer_n}"] = np.apply_along_axis(self.softmax, 1,cache[f"A{layer_n}"])
            else: 
                cache[f"Z{layer_n}"] = np.apply_along_axis(self.activation, 1,cache[f"A{layer_n}"])
        return cache

    
    def columnTimesMatrix(self,v,M):
        temp= np.zeros((np.shape(M)[1],len(v)))
        for i in range(np.shape(M)[0]):
            temp=temp+M[i,:][:,None]*v[None,:]      
        return temp/np.shape(M)[0]
    
        
    
    def backward(self, cache, labels):
#         print(cache)
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        for layer_n in list(reversed(range(1, self.n_hidden + 2))):
            layer_nmo=layer_n-1
            layer_npo=layer_n+1
            if layer_n== self.n_hidden+1:
                print(layer_n)
                grads[f"dA{layer_n}"]= np.mean(output-labels,axis=0)
                grads[f"dW{layer_n}"]= self.columnTimesMatrix(grads[f"dA{layer_n}"], cache[f"Z{layer_nmo}"])
                grads[f"db{layer_n}"]= grads[f"dA{layer_n}"]
                print(f"dA{layer_n}")
                print(np.shape(grads[f"dA{layer_n}"]))
                print(f"dW{layer_n}")
                print(np.shape(grads[f"dW{layer_n}"]))
                print(f"db{layer_n}")
                print(np.shape(grads[f"db{layer_n}"]))
            else:
                print("layern is")
                print(layer_n)
                print(f"W{layer_npo}")
                print(np.shape(self.weights[f"W{layer_npo}"]))
                print(f"dA{layer_npo}")
                print(np.shape(grads[f"dA{layer_npo}"]))
                grads[f"dZ{layer_n}"]= self.weights[f"W{layer_npo}"].dot(grads[f"dA{layer_npo}"])
                print(f"A{layer_n}")
                print(np.shape(cache[f"A{layer_n}"]))
                print(f"dZ{layer_n}")
                print(np.shape(grads[f"dZ{layer_n}"]))
                grads[f"dA{layer_n}"]= np.diag(grads[f"dZ{layer_n}"]).dot(np.mean(self.activation(cache[f"A{layer_n}"]),axis=0)) #np.mean(np.diag(grads[f"dZ{layer_n}"]).dot(np.transpose(self.activation(cache[f"A{layer_n}"],grad=True))),axis=1)
                print(f"dA{layer_n}")
                print(np.shape(grads[f"dA{layer_n}"]))
                grads[f"dW{layer_n}"]= self.columnTimesMatrix(grads[f"dA{layer_n}"], cache[f"Z{layer_nmo}"])
                grads[f"db{layer_n}"]= grads[f"dA{layer_n}"]
                print(f"dW{layer_n}")
                print(np.shape(grads[f"dW{layer_n}"]))
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"]=self.weights[f"W{layer}"]-self.lr*grads[f"dW{layer}"]
            self.weights[f"b{layer}"]=self.weights[f"b{layer}"]-self.lr*grads[f"db{layer}"]

    def one_hot(self, y):
        return np.eye(self.n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        N = prediction.shape[0]
        ce = -np.sum(labels*np.log(prediction))/N
        return ce



    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                pass

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        # WRITE CODE HERE
        pass
        return 0


aNeuralNet= NN()
aNeuralNet.initialize_weights([3072,10])
print(aNeuralNet.activation([[2,3],[2,3]]))
#aNeuralNet.loss(np.array([0,1,0,0,0,0,0,0,0,0]),np.array([0.2,0.3,0.1,0.05,0.05,0,0,0,0.2,0.2]))
theCache=aNeuralNet.forward(aNeuralNet.train[0][0:10])
aNeuralNet.update(aNeuralNet.backward(theCache,aNeuralNet.one_hot(aNeuralNet.train[1][0:10])))
#print(aNeuralNet.weights)

