import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

FONT = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16, }

NON_CAT_FEATURES = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat',
                    'sqft_living15', 'sqft_lot15', 'long_ordered']


def fit_linear_regression(X, y):
    """
    :param X: A design matrix pxn
    :param y:response vector
    :return: coefficient vector w and S singular values of X
    """
    w = np.matmul(pinv(X), y)
    singular_values = np.linalg.svd(X.drop('ones', axis=1), full_matrices=False, compute_uv=False)
    return w, singular_values


def predict(X, w):
    """
    :param X: A design matrix pxn
    :param w: coefficient vector
    :return: array of predicted values
    """
    return np.dot(X, w)


def mse(prediction, y):
    """
    :param y: response vector
    :param prediction: array of predicted values
    :return: MSE over the samples
    """
    norm_2_loss = np.linalg.norm(np.subtract(prediction, y)) ** 2
    return norm_2_loss / y.size


# ===========Processing functions============

def load_data(data_path):
    """
    :param data_path: path to the data
    :return: the data set after the pre-processing
    """
    data = pd.read_csv(data_path)

    # Validation of the data
    data = validate_data(data)

    # Change zipcode into One Hot encoding
    data = pd.concat([data, pd.get_dummies(data.zipcode)], axis=1)

    # Create design matrix and response vector
    response_vec = data['price']
    data = data.drop(['id', 'price', 'zipcode', 'date', 'long'], axis=1)
    data.insert(loc=0, column='ones', value=1)
    return data, response_vec


def validate_data(data):
    """
    :param data: date to validate
    :return: validated date
    """
    data.drop_duplicates('id', inplace=True)
    data.dropna(inplace=True)

    pos_filter = data['price'] > 0
    square_features = [feature for feature in data if 'sqft' in feature]
    for feature in square_features:
        pos_filter &= (data[feature] >= 0)
    data = data[pos_filter]
    data = data[(1 <= data["grade"]) & (data["grade"] <= 13)]
    data = data[(0 <= data["view"]) & (data["view"] <= 4)]
    data = data[(1 <= data["condition"]) & (data["condition"] <= 5)]

    # Order on long
    data['long_ordered'] = [-abs(long + 122.2) for long in data.long]

    # Check if we removed missing date correctly
    if data.isnull().values.any():
        print("There is missing date")

    plt.show()
    return data


def examine_date(data):
    data['year'] = [int(date[:4]) for date in data.date]
    data['month'] = [int(date[4:6]) for date in data.date]

    data = data[(data["year"] == 2014) | (data["year"] == 2015)]
    data = data[(data["month"] > 0) & (data["month"] < 13)]

    av_price_2014 = np.mean(data[data['year'] == 2014]['price'])
    av_price_2015 = np.mean((data[data['year'] == 2015]['price']))

    plt.figure(figsize=(8, 5))
    plt.scatter(data['year'], data['price'], alpha=0.4)
    plt.legend()
    plt.show()

    plt.scatter(data['month'], data['price'])
    plt.show()

    plt.xlabel("-|long +122.2| ")
    plt.ylabel("price")
    plt.title("Price as function of minus the long difference from -122.2")
    data['long'] = [-abs(long + 122.2) for long in data.long]

    plt.scatter(data['long'], data['price'])
    plt.show()


def plot_singular_values(sv):
    """
    This function is given a collection of singular
    values and plots them in descending order
    :param sv: numpy array of singular values
    """

    plt.figure(figsize=(8, 5))
    plt.title('Scree Plot: Singular values of the design matrix')
    plt.ylabel(r'Value of $\sigma_i$')
    plt.xlabel('Component number')
    plt.plot(sv, 'ro-', linewidth=2, label='Singular values')
    plt.legend()
    plt.show()


def print_rank(X):
    XTX = np.dot(X.T, X)
    s = np.linalg.eigvalsh(XTX)
    print(np.linalg.matrix_rank(XTX, 1e-14))
    print(np.linalg.matrix_rank(XTX))


def plot_data_test(total_data):
    """
    For every p in [0,100] fits a model based on the first p% of the training set. Then using the
    `predict` function test the performance of the fitted model on the test-set.
    """
    # train_error = []
    test_error = []
    train_set, test_set = train_test_split(total_data, test_size=0.25)
    test_response = test_set['price']
    X_test = test_set.drop('price', axis=1)
    test_prediction = []
    for i in range(1, 101):
        # Get the first p%
        slice_idx = (train_set.shape[0] * i) // 100
        curr_train = train_set[:slice_idx]

        # Get the response vector from the curr train set
        train_response = curr_train['price']

        # Get the design matrix and w
        X_train = curr_train.drop('price', axis=1)
        w, sing_values = fit_linear_regression(X_train, train_response)
        # Get predictions

        test_prediction = predict(X_test, w)
        # train_pred = predict(X_train, w)

        # Append error
        test_error.append(mse(test_prediction, test_response))
        # train_error.append(mse(train_pred, train_response))

    plt.plot(test_error, label="Test error")
    # plt.plot(train_error, label="Train error")
    plt.title("MSE over the test set as a function of p%")
    plt.yscale('log')
    plt.xlabel("Percent")
    plt.ylabel("MSE ")
    plt.legend()
    plt.show()

    # Plot Prediction vs True price
    plt.title("Test Prediction vs True price")
    plt.xlabel("Prediction")
    plt.ylabel("True price ")
    plt.scatter(test_prediction, test_response)
    plt.show()


def feature_evaluation(X, y):
    """
    This function, given the design matrix and response vector, plots for
    every non-categorical feature, a graph (scatter plot) of the feature values and the response
    values. It then also computes and shows on the graph the Pearson Correlation between
    the feature and the response.

    :param X: Design matrix
    :param y: response vector
    """
    std_y = np.std(y)
    for feature in NON_CAT_FEATURES:
        std_x = np.std(X[feature])
        cov_x_y = np.cov([X[feature], y])[0][1]
        corr_x_y = cov_x_y / (std_x * std_y)

        fig, ax = plt.subplots(1, figsize=(8, 5))
        fig.subplots_adjust(bottom=0.2)
        ax.scatter(X[feature], y, label='(' + feature + ",price)", alpha=0.5)

        # Create text for correlation
        font = {'fontsize': 20, 'ha': 'center', 'va': 'center', 'bbox': dict(boxstyle="round", fc="white")}
        corr_text = r'$\rho_{(price,' + feature.replace('_', '\\_') + ')} = $'
        fig.text(0.5, 0.05, corr_text + str(corr_x_y), font)

        plt.xlabel(feature)
        plt.ylabel("Price")
        plt.title("Price as function of the value of " + feature)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Processing

    X, y = load_data("kc_house_data.csv")
    # Q.15
    w, sing_values = fit_linear_regression(X, y)

    plot_singular_values(sing_values)

    # Q.16
    plot_data_test(pd.concat([X, y], axis=1))

    # Q.17
    feature_evaluation(X, y)
