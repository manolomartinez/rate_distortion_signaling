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

    def all_points(self, iterator=None, outputfile=None):
        """
        Calculate the R(D) function for as many points as given by <iterator>
        (all of them by default)
        Save them to <file>. Each line of the file (row of the array) is [K,
        distortion, rate]
        """
        if not iterator:
            iterator = range(self.K)
        return np.array([self.blahut(k, outputfile) for k in
                         iterator]).T

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

    def blahut(self, k, outputfile):
        """
        Calculate the point in the R(D)-D curve with slope given by
        self.calc_s(<k>). Follows Cover & Thomas 2006, p. 334
        """
        s = self.calc_s(k)
        # we start with the uniform output distribution
        output = np.ones(self.m) / self.m
        cond = self.update_conditional(s, output)
        distortion = self.calc_distortion(cond)
        rate = self.calc_rate(cond, output)
        delta_dist = 2 * self.epsilon
        while delta_dist > self.epsilon:
            output = self.pmf @ cond
            cond = self.update_conditional(s, output)
            new_distortion = self.calc_distortion(cond)
            rate = self.calc_rate(cond, output)
            delta_dist = np.abs(new_distortion - distortion)
            distortion = new_distortion
        if outputfile:
            with open(outputfile, "a") as outf:
                outf.write("{}\t{}\t{}\n".format(k, rate, new_distortion))
        return rate, new_distortion

    def update_conditional(self, s, output):
        """
        Calculate a new conditional distribution from the <output> distribution
        and the <s> parameter.  The conditional probability matrix is such that
        cond[i, j] corresponds to P(x^_j | x_i)
        """
        cond = output * np.exp(s * self.dist_matrix)
        cond = cond / cond.sum(1)[:, np.newaxis]  # normalizel
        return cond

    def calc_distortion(self, cond):
        """
        Calculate the distortion for a given channel (individuated by the
        conditional matrix in <cond>
        """
        # return np.sum(self.pmf @ (cond * self.dist_matrix))
        return np.matmul(self.pmf, (cond * self.dist_matrix)).sum()

    def calc_rate(self, cond, output):
        """
        Calculate the rate for a channel (given by <cond>) and output
        distribution (given by <output>)
        """
        return np.sum(self.pmf @ (cond * np.log2(cond / output)))
