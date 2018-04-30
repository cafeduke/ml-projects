import matplotlib.pyplot as plt
import data_source
from neural_network import NeuralNetwork
import pyduke.common.core_util as cu

def main():
    init_plot()    
    
    X_train, Y_train, X_dev, Y_dev = organize_image_data()
    
    # Normalize the image data loaded
    X_train, X_dev = NeuralNetwork.normalize_image_data ([X_train, X_dev])
    
    # Hyper parameter: Neural network layer sizes
    n_hidden = ((20, 7, 5))
    
    # Neural Network
    nn = NeuralNetwork (n_hidden, print_cost=True)
    AL, cost = nn.fit(X_train, Y_train)
    
    # Train accuracy 
    cu.heading ("Train Neural Network")
    Ycap_train = nn.predict(X_train)
    accuracy   = NeuralNetwork.get_accuracy(Y_train, Ycap_train)
    print ("Train accuracy = %3.4f" %(accuracy))
    
    # Dev acccuracy
    cu.heading ("Dev Run Using Trained Parameters")
    Ycap_dev = nn.predict(X_dev)
    accuracy = NeuralNetwork.get_accuracy (Y_dev, Ycap_dev)
    print ("Dev accuracy = %3.4f" %(accuracy))

def init_plot():
    plt.ioff()
    plt.rcParams['figure.figsize'] = (5.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
def organize_image_data():    
    cu.heading ("Organize Data")
    
    # Load data from file
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = data_source.load_cat_data()
    print ("Shape: train_x_orig = ", train_x_orig.shape, " test_x_orig = ", test_x_orig.shape)
    print ("Shape: train_y_orig = ", train_y_orig.shape, " test_y_orig = ", test_y_orig.shape)
    
    # Assert image size (width, height) for train and dev are same
    m_train, width, height, rgb = train_x_orig.shape
    m_dev = test_x_orig.shape[0]
    n_x = width * height * 3
    n_y = train_y_orig.shape[0]
    assert (test_x_orig.shape == ((m_dev, width, height, 3)))
    assert (test_y_orig.shape == ((n_y, m_dev)))
    
    # Reshape to get training data - Each column will be an image 
    X_train = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    Y_train = train_y_orig
    print ("Shape: X_train = ", X_train.shape, " Y_train = ", Y_train.shape)
    assert (X_train.shape == ((n_x, m_train)))
    assert (Y_train.shape == ((n_y, m_train)))
    
    # Reshape to get dev data - Each column will be an image
    X_dev = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    Y_dev = test_y_orig
    print ("Shape: X_dev = ", X_dev.shape, " Y_dev = ", Y_dev.shape)    
    assert (X_dev.shape == ((n_x, m_dev)))
    assert (Y_dev.shape == ((n_y, m_dev)))
    
    return X_train, Y_train, X_dev, Y_dev
    
if __name__ == '__main__':
    main()



