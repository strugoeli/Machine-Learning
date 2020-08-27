import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

EM_RES = 'EM GMM results for\n{} clusters, {} iterations'
CLUSTERS_VALUES = [1, 2, 3, 4, 5]
THRESHOLD = 1e-3
MAX_ITERATIONS = 100


def EM_GMM(Xn, k):
    """
    :param Xn: n i.i.d samples
    :param k: number of clusters
    :return: vector that contains the likelihood at each iteration
    and gmm object for the given data
    """
    gmm = GMM(Xn, k)
    L = [-np.inf]
    while gmm.advance():
        L.append(gmm.curr_likelihood)
    L.append(gmm.curr_likelihood)
    return gmm, L

class GMM:

    def __init__(self, Xn, k):
        self.Xn = Xn
        self.n, self.d = Xn.shape
        self.k = k
        self.pi = np.random.rand(k)
        self.pi /= np.sum(self.pi)
        self.pi[0] += 1 - np.sum(self.pi)
        self.mu = np.random.rand(k, self.d)
        self.sigma = np.zeros((self.k, self.d, self.d))
        self.sigma[:, 0, 0] = 1
        self.sigma[:, 1, 1] = 1
        self.w = np.zeros((self.n, self.k))
        self.curr_likelihood = self._cal_max_likelihood()
        self.t = 0

    def advance(self):
        """
        preforming iteration of the EM GMM algorithm
        :return: true if the l(tetha(t))~l(true tetha) false otherwise
        """
        self._e_step()
        self._m_step()
        next_likelihood = self._cal_max_likelihood()
        self.t += 1
        not_converges = next_likelihood - self.curr_likelihood >= THRESHOLD
        self.curr_likelihood = next_likelihood
        return not_converges and self.t <= MAX_ITERATIONS

    def _cal_max_likelihood(self):
        """
        calculates the max likelihood of the iteration t
        :return:
        """
        res = 0
        for i in range(self.n):
            temp_sum = 0
            for z in range(self.k):
                x_i, mu_z, sigma_z = self.Xn[i], self.mu[z], self.sigma[z]
                temp_sum += self.pi[z] * self._mv_gauss(x_i, z)
            res += np.log(temp_sum)
        return res

    def _mv_gauss(self, x_i, z):
        """
        :param x_i: sample
        :param z: current cluster index
        :return: f(x_i; mu_z ,sigma_z)
        """
        return mvn.pdf(x_i, mean=self.mu[z], cov=self.sigma[z])

    def _e_step(self):
        """
        Updating w for the current t iteration
        """
        for z in range(self.k):
            self.w[:, z] = self.pi[z] * mvn.pdf(self.Xn, self.mu[z, :],
                                                self.sigma[z])
        norm = np.sum(self.w, axis=1)[:, np.newaxis]
        self.w /= norm

    def _m_step(self):
        """
        Updating pi mu and sigma for the current t iteration
        """
        self.pi = np.mean(self.w, axis=0)
        self.mu = np.dot(self.w.T, self.Xn) / np.sum(self.w, axis=0)[:,
                                              np.newaxis]
        for z in range(self.k):
            self._update_sigma_z(z)

    def _update_sigma_z(self, z):
        """
        updating sigma[z]
        :param z: current cluster index
        """
        temp_sum = np.zeros([self.d, self.d])
        for i in range(self.n):
            if self.Xn[i].ndim == 1:
                data_temp = self.Xn[i].reshape(self.Xn.shape[1], 1)
                mu_temp = self.mu[z].reshape(self.mu.shape[1], 1)
                m = data_temp - mu_temp
            else:
                m = self.Xn[i] - self.mu[z]
            temp_sum += (self.w[i][z] / self.pi[z]) * np.dot(m, m.T)
        self.sigma[z] = temp_sum / float(self.n)


def plot_clusters(Xn, gmm):
    data_x = Xn[:, 0]
    data_y = Xn[:, 1]
    x = np.linspace(np.min(data_x) - 1, np.max(data_x) + 1)
    y = np.linspace(np.min(data_y) - 1, np.max(data_y) + 1)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.ravel(), Y.ravel()]).T
    plt.figure()
    for z in range(gmm.k):
        Z = mvn.pdf(x=XY, mean=gmm.mu[z], cov=gmm.sigma[z])
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z)
    plt.scatter(data_x, data_y)
    plt.title(EM_RES.format(k, gmm.t))
    plt.show()


def plot_log_likelihood(L, k):
    plt.title("log-likelihood as a function of t for k={}".format(k))
    plt.xlabel("t")
    plt.ylabel("log-likelihood")
    plt.plot(L)
    plt.show()


def plot_converged_likelihoods(L):
    plt.title("Converged log-likelihoods as a function of k")
    plt.xlabel("k")
    plt.ylabel("log-likelihood")
    plt.scatter(CLUSTERS_VALUES, L)
    plt.show()


def plot_BIC(BIC):
    plt.title("BIC")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.scatter(CLUSTERS_VALUES, BIC)
    plt.show()


if __name__ == '__main__':

    Xn = np.loadtxt(fname="EM_data.txt", dtype=np.float64)
    n, d = Xn.shape
    max_likelihoods = []
    BIC = []
    for k in CLUSTERS_VALUES:
        gmm, L = EM_GMM(Xn, k)
        plot_clusters(Xn, gmm)
        plot_log_likelihood(L, k)
        max_likelihoods.append(L[-1])
        BIC.append(k * d * (d + 1) / 2 * np.log(n) - 2 * L[-1])
    plot_converged_likelihoods(max_likelihoods)
    plot_BIC(BIC)
