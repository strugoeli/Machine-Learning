import tensorflow as tf
import matplotlib.pyplot as plt
from comparison import examine_models
from model import *


def q12(X, y):
    """
    Draw 3 images of samples labeled with '0' and three labeled with '1'
    :param X: train set
    :param y: train label
    """
    fig, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    im_idx = np.concatenate([np.argwhere(y == 1)[:3], np.argwhere(y == 0)[:3]])

    for i in range(6):
        plottable_image = np.reshape(X[im_idx[i]], (28, 28))
        ax[i].imshow(plottable_image, cmap='gray_r')

    fig.show()


def rearrange_data(X):
    """
    :param X: mx28x28
    :return: reshaped X with mx784
    """
    return np.reshape(X, (X.shape[0], 784))


def draw_points(m, data, labels):
    """
        Draw m point from m samples and and there labels uniformly returns it( rearrange with mx784)
        :param m: Number of samples
        :param data:  used to draw from
        :param labels: used to draw from

        :return: matrix X of size mx784 and label vector y of size m with y=1 or y=0
        """
    idx = np.random.randint(0, m + 1, m)
    idx.reshape(-1, 1)
    X = rearrange_data(data[idx])
    y = labels[idx]
    return X.T, y.T


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]

    # q12(x_train, y_train)
    train_m = [50, 100, 300, 500]

    # Using the same function from q10 with different parameters
    models = [Logistic(), SoftSVM(1, 0, 0.1), DecisionTree(), KNeighbors(1)]
    examine_models(train_m, None, 50, 'accuracy', models, draw_points, x_train, y_train, x_test, y_test)
