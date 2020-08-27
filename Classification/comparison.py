import matplotlib.pyplot as plt
from model import *
import time

TRUE_HYPOTHESIS_W = [0.3, -0.5, 0.1]


def draw_points(m, *arg):
    """
    Draw m point from multivariate normal distribution and returns it as a 2xm matrix and vector y
    s.t yi=f(xi)=sign(<(0.3,-0.5)>+0.1)
    :param m: Number of samples
    :param arg: unused
    :return: matrix X of size 2xm and label vector y of size m with y=1 or y=-1
    """
    X = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
    y = np.sign(np.dot(X, np.array([0.3, -0.5]).T) + 0.1)
    return X.T, y.T


def get_train_X_y(m, draw_func, data, labels):
    """
    Draw date by the given draw function and makes sure that the vector y consists at least labels
    from two different classes
    :param m: Number of samples
    :param draw_func: function used to draw data
    :param data: can be used to draw from
    :param labels: can be used to draw from
    :return: X_train.y_train
    """
    train_X, train_y = draw_func(m, data, labels)
    while np.unique(train_y).shape[0] != 2:
        train_X, train_y = draw_func(m, data, labels)
    return train_X, train_y


def get_hyperplane(w):
    """
    :param w: Coefficient vector
    :return: Hyperplane created by the given parameters
    """
    return np.poly1d([w[0], w[2]]) / -w[1]


def plot_hyperplanes(train_sizes, models):
    """
    for every m in train_sizes it draws m points from multivariate normal distribution scatters them,
    fits the given models and plot there hyperplane against each other and against the true hypothesis.
    Furthermore, it scatters the
    :param train_sizes: Sizes for the train set
    :param models: models to fit and get there hyperplanes

    """
    true_h = get_hyperplane(TRUE_HYPOTHESIS_W)
    for m in train_sizes:
        X, y = draw_points(m)
        x = X[0, :]
        for model in models:
            model.fit(X, y)
        perceptron_h, svm_h = get_hyperplane(models[0].w), get_hyperplane(models[1].w)

        plt.scatter(X[:, y == 1][0, :], X[:, y == 1][1, :], color='blue', marker='o', label="positive tag")
        plt.scatter(X[:, y == -1][0, :], X[:, y == -1][1, :], color='orange', marker='o', label="negative ""tag")

        plt.title("Hyperplanes of True hypothesis, Perceptron and SVM for " + str(m) + " samples")
        plt.plot(x, true_h(x), 'green', label="True hypothesis")
        plt.plot(x, perceptron_h(x), label="Perceptron")
        plt.plot(x, svm_h(x), label="SVM")
        plt.ylabel('y')
        plt.xlabel('x')
        plt.legend()
        plt.savefig('./figs/' + str(m) + '.png')
        plt.show()


def examine_models(train_sizes, test_size, iterations, type_of_score, models, draw_func, data=None, labels=None,
                   data_test=None, label_test=None):
    """
     For every m in train_sizes it draws m points by the draw function,fits the given models and plot
     there performance on test sets - given score type and running time for the process
    :param test_size:  Size for the test set, if None it will be as same as the training size
    :param iterations: number of iterations for calculating the mean score
    :param type_of_score: The of score to examine
    :param models: models to examine
    :param draw_func: function used to draw data
    :param data: can be used to draw data from(for train)
    :param labels:  can be used to draw labels from(for train)
    :param data_test: can be used to draw data from(for test)
    :param label_test: can be used to draw labels from(for test)
    """
    same_set_size = test_size is None
    # Init the score dictionary
    scores_mean = {model.get_type(): np.zeros(len(train_sizes)) for model in models}
    run_time = {model.get_type(): np.zeros(len(train_sizes)) for model in models}

    for m, size_idx in zip(train_sizes, range(len(train_sizes))):

        # Use given test size or use the same size of the current train set
        test_size = m if same_set_size else test_size

        for i in range(iterations):

            # Create train and test sets
            train_X, train_y = get_train_X_y(m, draw_func, data, labels)
            test_X, test_y = draw_func(test_size, data_test, label_test)

            # Fit the models and get add there score to the sum of scores
            for model in models:
                start = time.process_time()
                model.fit(train_X, train_y)
                model_type = model.get_type()
                scores_mean[model_type][size_idx] += model.get_from_score(test_X, test_y, type_of_score)
                run_time[model_type][size_idx] += time.process_time() - start

        # Divide by the number of iterations
        for model in models:
            model_type = model.get_type()
            scores_mean[model_type][size_idx] /= np.float(iterations)

        # Indication for finishing a round
        print("Pass m =", m)

    # Plot mean score for each model
    plot_score_run_time(models, run_time, scores_mean, train_sizes, type_of_score)


def plot_score_run_time(models, run_time, scores_mean, train_sizes, type_of_score):
    """

    :param models: models to examine
    :param run_time: array of runtime for each model
    :param scores_mean: array of scores for each model
    :param train_sizes: sizes used to examine the models
    :param type_of_score: type of score
    """
    for model in models:
        model_type = model.get_type()
        plt.plot(train_sizes, scores_mean[model_type], marker='o', label=model_type + " " + type_of_score)
        print(model_type, scores_mean[model_type])
    plt.title("Mean accuracy as a function of the number of train samples")
    plt.ylabel("Accuracy Percentage")
    plt.xlabel("Number of train samples")
    plt.legend()
    plt.show()
    plt.title("Running time as a function of the number of train samples")
    plt.xlabel("Number of train samples")
    plt.ylabel("Time in Sec")
    for model in models:
        model_type = model.get_type()
        plt.plot(train_sizes, run_time[model_type], marker='o', label=model_type + " run time")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_m = [5, 10, 15, 25, 70]
    models = [Perceptron(), SVM()]
    plot_hyperplanes(train_m, models)
    models = [Perceptron(), SVM(), LDA()]
    examine_models(train_m, 1000, 500, 'accuracy', models, draw_points)
