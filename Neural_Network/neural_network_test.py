import unittest
import ml_util
import numpy as np
from neural_network import NeuralNetwork
  
class NeuralNetworkTest (unittest.TestCase):    

    def test_model_forward_propagation(self):
        np.random.seed(6)
        X = np.random.randn(5,4)
        W1 = np.random.randn(4,5)
        b1 = np.random.randn(4,1)
        W2 = np.random.randn(3,4)
        b2 = np.random.randn(3,1)
        W3 = np.random.randn(1,3)
        b3 = np.random.randn(1,1)
        Y  = np.random.randn(1,4)
      
        param = {
            "W1": W1, "b1": b1,
            "W2": W2, "b2": b2,
            "W3": W3, "b3": b3
        }
        
        nn = NeuralNetwork (X, Y, ((4, 3)))
        AL, _ = nn.model_forward_propagation (param)
        
        AL_expect = np.round(np.array ([[0.03921668,  0.70498921,  0.19734387,  0.04728177]]), 4)
        AL_actual = np.round(AL, 4)
        assert (np.array_equal(AL_expect, AL_actual) == True)
        
    def test_compute_cost(self):
        Y = np.asarray([[1, 1, 1]])
        AL = np.array([[.8,.9,0.4]])        
        cost = ml_util.cost_function(Y, AL)
        assert (np.round(cost,4) == 0.4149)
        
    def test_model_back_propagation(self):
        np.random.seed(3)
    
        # X = 4 Layers: 3, 1    
        A2 = np.random.randn(1, 2)
        Y = np.array([[1, 0]])
    
        X  = np.random.randn(4,2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        Z1 = np.random.randn(3,2)    
    
        A1 = np.random.randn(3,2)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        Z2 = np.random.randn(1,2)
        
        nn = NeuralNetwork (X, Y, ((3,)))
        
        param = {
            "W1": W1, "b1": b1,
            "W2": W2, "b2": b2
        }    
        
        cache = {
            "A0": X,
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2
        }    
        
        cache = nn.model_back_propagation (param, cache)
        
        dW1_expect = np.round(np.array([[ 0.41010002, 0.07807203, 0.13798444, 0.10502167], [ 0., 0., 0., 0. ], [ 0.05283652, 0.01005865, 0.01777766, 0.0135308 ]]), 4)
        dW1_actual = np.round(cache["dW1"], 4)
        
        db1_expect = np.round(np.array([[-0.22007063], [ 0. ], [-0.02835349]]), 4)
        db1_actual = np.round(cache["db1"], 4)
    
        assert (np.array_equal(dW1_expect, dW1_actual) == True)
        assert (np.array_equal(db1_expect, db1_actual) == True)
        
        
if __name__ == '__main__':
    unittest.main()    
    