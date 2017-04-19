import numpy as np


class RDT:
    """
    Calculate the rate-distortion function.
    It expects a dit discrete distribution object
    K is the number of points to calculate for the R(D) curve
    """
    def __init__(self, distribution, a=3, b=3, K=10, epsilon=0.001):
        self.dist_vec = np.vectorize(self.distortion)
        # I am not sure where np.vectorize statements such as these should go.
        # At the top of __init__ for now :/
        self.K = K
        self.pmf = distribution.pmf
        self.outcomes = distribution.outcomes
        self.m = len(self.pmf)
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.s = np.array([self.calc_s(k) for k in range(K)])
        self.dist_matrix = self.calc_dist_matrix()

    def all_points(self):
        """
        Calculate the R(D) function for as many points as given by K
        """
        return np.array([self.blahut(k) for k in range(self.K)]).T

    def blahut(self, k):
        """
        Calculate the algorithm for one value of k
        """
        self.init_q()
        self.A_vec = np.vectorize(lambda i, j: self.calc_A(k, i, j))
        self.A = np.fromfunction(self.A_vec, (self.m, self.m), dtype=int)
        Tu, Tl = self.blahut_step(k)
        while Tu - Tl >= self.epsilon:
            # print("{} >= {}".format(Tu - Tl, self.epsilon))
            Tu, Tl = self.blahut_step(k)
        D, RD_prov = self.blahut_wrapup(k)
        return D, RD_prov + (Tu + Tl) / 2

    def init_q(self):
        """
        Initialize q
        """
        self.q = np.array([1/self.m] * self.m)

    def blahut_step(self, k):
        """
        Calculate one step of the algorithm for a certain value of k
        """
        alpha = self.calc_alpha()
        c = self.calc_c(alpha)
        self.q = self.q * c
        logc = np.log2(c)
        Tu = -1 * np.einsum('i,i', self.q, logc)
        Tl = -1 * np.max(logc)
        return Tu, Tl

    def blahut_wrapup(self, k):
        """
        The final steps
        """
        alpha = self.calc_alpha()
        self.Q_vec = np.vectorize(lambda i, j: self.calc_Q(alpha, i, j))
        Q = np.fromfunction(self.Q_vec, (self.m, self.m), dtype=int)
        Qtimesd = np.einsum('...j, ...j', Q, self.dist_matrix)
        D = np.einsum('i, i', self.pmf, Qtimesd)
        RD_prov = self.s[k] * D - np.einsum('i, i', self.pmf, np.log2(alpha))
        # This is R(D) minus the (Tu + Tl)/2 correction
        return D, RD_prov

    def distortion(self, i, j):
        """
        Compare two strings of 0s and 1s (expected to be of the same length)
        and report the number of differences
        """
        return sum([chari != charj for chari, charj in zip(i, j)])

    def calc_dist_matrix(self):
        return np.array([[self.distortion(i, j) for j in self.outcomes] for i
                         in self.outcomes])

    def calc_s(self, k):
        return -self.a * np.exp(-self.b * k)

    def calc_A(self, k, i, j):
        return np.exp(self.s[k] * self.dist_matrix[i, j])

    def calc_alpha(self):
        return np.einsum('j, ij', self.q, self.A)

    def calc_c(self, alpha):
        p_over_alpha = self.pmf / alpha
        return np.einsum('i, ij', p_over_alpha, self.A)

    def calc_Q(self, alpha, i, j):
        return self.A[i, j] * self.q[j] / alpha[i]
