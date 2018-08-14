import numpy as np
import pyduke.common.core_util as cu

from enum import Enum
ActivationFunction = Enum('ActivationFunction', 'relu sigmoid')

class NeuralNetwork:
    
    def __init__(self, n_hidden=(20, 7, 5), K=1, learning_rate=0.0075, iter_count=2500, rlambda=0, print_cost=False, seed=1):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.iter_count = iter_count
        self.rlambda = rlambda
        self.print_cost = print_cost
        self.seed = seed
        self.K = K
        
    def fit (self, X, y):
        self.m = X.shape[1]
        y = y.reshape(1, -1)
        assert y.shape == (1,self.m), 'y.shape={} m={}'.format(y.shape, self.m)
        
        self.K = np.max([self.K, np.max(y)])
        
        if self.K > 1:
            Y = np.zeros((self.K, self.m))
            for i, x in enumerate(y[0]):
                if x > 0:
                    Y[x-1][i] = 1
        else:
            Y = y
        self.n, self.L = self.get_layer_sizes(X, Y)
        AL, cost, self.param = self.model_neural_network(X, Y)
        return AL, cost    
        
    def predict (self, X, threshold=0.5):
        AL = self.get_prediction_forward_propagation (X, self.param)
        Ycap = (AL > threshold).astype(int)
        return cu.index_of(Ycap, 1) if self.K > 1 else Ycap
    
    def model_neural_network (self, X, Y, activation_type=None):
        
        if activation_type is None:
            activation_type = self.get_activation_type()
        
        assert (self.L == len(activation_type))
        
        # Hyper parameter : seed 
        np.random.seed(self.seed)
    
        # Init Weight & Bias
        param = self.init_weight_bias ()
        
        # Gradient descent
        # Hyper parameter: Learning rate, iteration count
        AL, cost, param = self.gradient_descent(X, Y, param, activation_type=activation_type)
        
        return AL, cost, param    
    
    def get_prediction_forward_propagation (self, X, param, activation_type=None):
        if activation_type is None:
            activation_type = self.get_activation_type()
        AL, _ = self.model_forward_propagation(X, param, activation_type)
        return AL
        
    def gradient_descent (self, X, Y, param, activation_type=None):
        
        if self.print_cost:
            cu.heading ("Gradient Descent")   
        
        if activation_type is None:
            activation_type = self.get_activation_type()
    
        list_cost = [] 
        for i in range(self.iter_count):
            
            AL, cache = self.model_forward_propagation (X, param, activation_type=activation_type)
            
            cache = self.model_back_propagation(Y, param, cache, activation_type)
    
            param = self.update_param (param, cache, self.learning_rate)
            
            if self.print_cost and (i % 100 == 0 or i == self.iter_count-1):
                cost = self.cost_function(Y, AL, param)
                print ("Cost after iteration [%4i] = %2.4f" %(i, cost))
                list_cost.append(cost)         
            
        return AL, cost, param  
    
    def cost_function (self, Y, A, param):
        
        regularization = 0
        for l in range(1, self.L+1):
            W = param["W" + str(l)]
            regularization =  regularization + np.sum(np.square(W))
        regularization  = (self.rlambda / 2*self.m ) * regularization                  
        
        cost = (-1/self.m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) + regularization
        
        # Convert from say [[17]] to 17
        cost = np.squeeze(cost)         
        assert(cost.shape == ())
        return cost       
    
    
    def update_param (self, param, cache, learning_rate):
        for l in range(1, self.L+1):
            param["W" + str(l)] -= learning_rate * cache["dW" + str(l)]  
            param["b" + str(l)] -= learning_rate * cache["db" + str(l)]
        return param    
        
    def model_forward_propagation (self, X, param, activation_type=None):
        
        if activation_type is None:
            activation_type = self.get_activation_type()
        
        # Count of W,b pairs should be equal to number of layers  
        assert (len(activation_type) == self.L == len(param.keys())/2)
        
        # Cache shall have all the Zs and As generated during forward propagation
        cache = {}
        cache["A0"] = X
        for l in range (1, self.L+1):
            W        = param["W" + str(l)]
            b        = param["b" + str(l)]
            A_prev   = cache["A" + str(l-1)]        
            Z        = W.dot(A_prev) + b
            
            cache["Z" + str(l)] = Z
            cache["A" + str(l)] = self.do_activation(activation_type[l-1], Z)
    
        # Return AL and cache
        return cache["A" + str(self.L)], cache
    
    def model_back_propagation (self, Y, param, cache, activation_type=None):
        
        if activation_type is None:
            activation_type = self.get_activation_type()        
        
        # Count of W,b pairs should be equal to number of layers
        assert (self.L == len(param)/2)
        
        # Cache shall have all the Zs and As generated during forward propagation
        # The loop shall run through all the layers from last to first
        for l in range (self.L, 0, -1):
            
            # The dZ calculation various for the last layer
            if (l == self.L):
                Z = cache["Z" + str(l)]
                AL = cache["A" + str(l)]
                dZ = ( -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) ) * self.do_derivation (activation_type[l-1], Z) 
            else:
                Z = cache["Z" + str(l)]
                W_next  = param["W" + str(l+1)]
                dZ_next = cache["dZ" + str(l+1)]
                dZ = W_next.T.dot(dZ_next) * self.do_derivation (activation_type[l-1], Z) 
    
            # The dW calculation is the same for all the layers
            A_prev = cache["A" + str(l-1)]  
            dW = (1/self.m) * dZ.dot(A_prev.T)
            db = (1/self.m) * np.sum(dZ,axis=1,keepdims=True)
                        
            cache["dZ" + str(l)] = dZ
            cache["dW" + str(l)] = dW
            cache["db" + str(l)] = db
                
        return cache    
            
    
    def get_layer_sizes (self, X, Y):
        n = (X.shape[0],) + tuple(self.n_hidden) + (Y.shape[0],)
        L = len(n) - 1
        return n, L
        
    def init_weight_bias (self):
        cu.heading ("Init Weight & Bias")
        L = len(self.n)-1
        param = {}
        for l in range (1, L+1):
            # param["W" + str(l)] = np.random.randn(n[l], n[l-1]) * 0.01     
            param["W" + str(l)] = np.random.randn(self.n[l], self.n[l-1]) / np.sqrt(self.n[l-1])
            param["b" + str(l)] = np.zeros((self.n[l], 1))  
            print ("W" + str(l), param["W" + str(l)].shape, "b" + str(l), param["b" + str(l)].shape)        
        return param   
    
    def get_activation_type (self):   
        activation_type = np.array([ActivationFunction.relu] * self.L)
        activation_type[self.L-1] = ActivationFunction.sigmoid
        return activation_type        
            
    def do_activation (self, activation_type, Z):
        A_output = None
        if activation_type == ActivationFunction.relu:
            A_output = NeuralNetwork.relu(Z)
        elif activation_type == ActivationFunction.sigmoid:
            A_output = NeuralNetwork.sigmoid(Z)
        return A_output
    
    def do_derivation (self, activation_type, Z):
        dZ = None
        if activation_type == ActivationFunction.relu:
            dZ = NeuralNetwork.relu_derivative(Z)
        elif activation_type == ActivationFunction.sigmoid:
            dZ = NeuralNetwork.sigmoid_derivative(Z)
        return dZ
    
    @staticmethod     
    def get_accuracy (Y, Ycap):
        score = np.sum((Ycap == Y).astype(int))
        total = Y.shape[1]
        return (score / total) * 100    
   
    @staticmethod
    def sigmoid(Z):
        """
        Implements the sigmoid function
    
        :param Z: The Z vector
        :return : g(Z) = 1 / (1 + e^-z)
        """
        A = 1/(1+np.exp(-Z))
        assert(A.shape == Z.shape) 
        return A
    
    @staticmethod
    def relu(Z):
        """
        The RELU function
        
        :param Z: The Z vector
        :type Z: g(Z) = max (0, z)
        """
        
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        return A
    
    @staticmethod
    def relu_derivative(Z):
        """
        Derivative of relu (which evaluates to either 0 or 1)
    
        :param Z: The Z vector
        :return : 1 if (z > 0) else 0
        """
    
        # Negative values become 0, positive values become 1
        dZ = (Z>0).astype(int)
        
        assert (Z.shape == dZ.shape)    
        return dZ;
    
    @staticmethod
    def sigmoid_derivative(Z):
        """
        Derivative of the sigmoid function (which evaluates to z(1 -z))
    
        :param Z: The Z vector
        :return: z(1-z)
        """
    
        dZ = 1/(1+np.exp(-Z))
        dZ = dZ * (1-dZ)
        
        assert (Z.shape == dZ.shape)    
        return dZ
    
    @staticmethod
    def normalize_image_data (list_x):
        """
        Normalize a list_x of array where each array element has image pixel component (RGB) in the range 0-255    
        """
        for i in range(len(list_x)):
            list_x[i] = list_x[i] / 255;
        return list_x 
    
    @staticmethod
    def normalize (list_x):
        for i in range(len(list_x)):
            X = list_x[i]
            n, m = X.shape
            
            # Zero mean
            X = X - ( np.mean(X, axis=1).reshape(n, 1) )        
            X = X / ((1/m) * np.sum(X ** 2, axis=1).reshape(n, 1))
            return X
        
    @staticmethod    
    def normalize_basic (list_x):
        for i in range(len(list_x)):
            X = list_x[i]
            (n,) = X.shape
            X = (X - np.mean(X, axis=1).reshape(n, 1)) / (np.max(X, axis=1) - np.min(X, axis=1))
            return X    
    
