from linear_model import *

PATH_TO_DATA = "covid19_israel.csv"


def plot_log_detected(X, w, y):
    """
    plots the estimated curve for log detected
    :param X: Design matrix
    :param w: coefficient vector
    :param y: response vector
    """
    x = X['day_num']
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="(day number,log detected)")
    prediction = np.dot(X, w)
    plt.plot(x, prediction, label="prediction", color="orange")
    plt.title("Log the number of cases as a function of the number of days")
    plt.ylabel("Log(number of cases)")
    plt.xlabel("Number of days")
    plt.legend()
    plt.show()


def plot_detected(X, w, y):
    """
     plots the estimated curve for detected
    :param X: Design matrix
    :param w: coefficient vector
    :param y: response vector
    """
    x = X['day_num']
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="(day number,detected)", color='r')
    prediction = np.exp(np.dot(X, w))
    plt.plot(x, prediction, label="prediction", color="green")
    plt.title("Number of cases as a function of the number of days")
    plt.xlabel("Number of days")
    plt.ylabel("Number of cases")
    plt.legend()
    plt.show()


def fit_weighted(X, y, w):
    """
    :param X: A design matrix
    :param y: response vector
    :param w: weights
    :return: b_hat = argmin (Sum (w(i)* (y(i) - <x(i),b>)^2)
    """
    diag_w = np.diag(w)
    XT_W_X = np.dot(np.dot(X.T, diag_w), X)
    return np.dot(np.linalg.pinv(XT_W_X), np.dot(X.T, np.dot(diag_w, y)))


def get_data(path):
    """
    :param path: Path to the data
    :return: the date with additional log_detected column
    """
    data = pd.read_csv(path)
    data['log_detected'] = np.log(data['detected'])
    return data


if __name__ == '__main__':
    # Processing
    data = get_data(PATH_TO_DATA)
    y, log_y = data['detected'], data['log_detected']
    data.insert(loc=0, column='ones', value=1)  # Adding ones
    data = data.drop('date', axis=1)  # The date is redundant
    X = data.drop(['log_detected', 'detected'], axis=1)

    # Find w and plot
    w, sv = fit_linear_regression(X, log_y)
    plot_log_detected(X, w, log_y)
    # Using the estimator for log(y)
    plot_detected(X, w, y)
    # Better solution for great values of y
    w = fit_weighted(X, log_y, y)
    plot_detected(X, w, y)
