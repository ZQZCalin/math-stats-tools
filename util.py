from math import floor, ceil, sqrt
import numpy as np 
from scipy.special import binom
import matplotlib.pyplot as plt

def sample_stats(sample):
    sample_size = len(sample)
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1) # unbiased variance
    print("sample size is: %d; sample mean is: %f; sample variance is: %f" \
        % (sample_size, sample_mean, sample_variance))
    return sample_size, sample_mean, sample_variance

def two_sample_normal_pool_var(n, var_x, m, var_y):
    var_pool = ((n-1)*var_x+(m-1)*var_y) / (n+m+2)
    return var_pool

def two_sample_normal_ttest_df(n, var_x, m, var_y):
    df = (var_x/n+var_y/m)**2 / (var_x**2/((n-1)*n**2)+var_y**2/((m-1)*m**2))
    return floor(df)

class one_sample_normal_test():
    # if true variance is known, use true variance as var; otherwise use sample variance
    def __init__(self, sample_stats_):
        self.n, self.mean, self.var = sample_stats_
    
    # test H0: mean == null (normal/t test)
    def mean_test(self, null, alternative=0):
        """
        null: null hypothesis mean
        alternative: 0: known variance, 1: unknown variance
        """
        if alternative == 0:
            test_stat = (self.mean-null) / sqrt(self.var/self.n)
            print("test statistic is: z = %f" % test_stat)

        elif alternative == 1:
            test_stat = (self.mean-null) / sqrt(self.var/self.n)
            print("test statistic is: t = %f" % test_stat)

    # test H0: var == null (chi^2 test)
    def var_test(self, null):
        """
        null: null hypothesis "variance"
        """
        test_stat = (self.n-1)*self.var/null
        print("test statistic is: chi^2 = %f" % test_stat)

    
class two_sample_normal_test():
    def __init__(self, group_X, group_Y):
        """
        if true variance is known, use true variance as var_x, var_y; 
        if s_p^2 (pooled variance) is given, use s_p^2 as both var_x, var_y;
        otherwise use sample variance
        """
        self.n, self.mean_x, self.var_x = group_X
        self.m, self.mean_y, self.var_y = group_Y

    # test H0: mean_x - mean_y == null (normal/t test)
    def mean_test(self, null=0, alternative=2):
        """
        null: mean_x0 - mean_y0
        alternative: 0: known variance, 1: unknown but assumed equal, 2: unknown
        """
        if alternative == 0:
            test_stat = (self.mean_x-self.mean_y-null) / sqrt(self.var_x/self.n+self.var_y/self.m)
            print("test statistic is: z = %f" % test_stat)
        
        elif alternative == 1:
            var_pool = ((self.n-1)*self.var_x+(self.m-1)*self.var_y) / (self.n+self.m-2)
            test_stat = (self.mean_x-self.mean_y-null) / sqrt(var_pool*(1/self.n+1/self.m))
            print("pooled variance is: s_p^2 = %f; test statistic is: t = %f" % (var_pool, test_stat))

        elif alternative == 2:
            # df = floor( (self.var_x/self.n + self.var_y/self.m)**2 \
            #     / (self.var_x**2/self.n**2/(self.n-1) + self.var_y**2/self.m**2/(self.m-1)) )
            df = two_sample_normal_ttest_df(self.n, self.var_x, self.m, self.var_y)
            test_stat = (self.mean_x-self.mean_y-null) / sqrt(self.var_x/self.n+self.var_y/self.m)
            print("degree of freedom of t-dist is: df = %d; test statistic is: t = %f" % (df, test_stat))

    # test H0: var_x == var_y (F test)
    def variance_test(self):
        test_stat = self.var_y/self.var_x
        print("test statistic (s_y^2/s_x^2) is: %f" % test_stat)

def two_sample_binomial_test(x, n, y, m):
    p_pool = (x+y) / (n+m)
    test_stat = (x/n-y/m) / sqrt(p_pool*(1-p_pool)*(1/n+1/m))
    print("pooled prob is: pe = %f; test stat is: z = %f" % (p_pool, test_stat))

# ===========================
# Confidence Intervals
# ===========================

# 100(1-alpha)% CI for (mu_x-mu_y)
class two_sample_normal_CI():
    def __init__(self, group_X, group_Y, unknown=True):
        self.n, self.mean_x, self.var_x = group_X
        self.m, self.mean_y, self.var_y = group_Y
        if unknown:
            df = two_sample_normal_ttest_df(self.n, self.var_x, self.m, self.var_y)
            print("degree of freedom is: df = %d" % df)

    # z: z_{alpha/2}, t_{alpha/2,n+m-2}, or t_{alpha/2,nu}
    def mean_CI(self, z):
        lower = (self.mean_x-self.mean_y) - z*sqrt(self.var_x/self.n+self.var_y/self.m)
        upper = (self.mean_x-self.mean_y) + z*sqrt(self.var_x/self.n+self.var_y/self.m)
        print("CI = (%f, %f)" % (lower, upper))

    def variance_CI(self, f):
        return

# 100(1-alpha)% CI for (p_x-p_y), z_{alpha/2}
def two_sample_binomial_CI(x, n, y, m, z):
    pe = (x+y) / (n+m)
    lower = (x/n+y/m) - z*sqrt(pe*(1-pe)*(1/n+1/m))
    upper = (x/n+y/m) + z*sqrt(pe*(1-pe)*(1/n+1/m))
    print("pooled prob is: pe = %f; CI is: z = (%f, %f)" % (pe, lower, upper))


# ===========================
# Multinomial Distribution
# ===========================

# computes multinomial coefficient, args: list [n1,...,nk]
def multinom(args):
    if len(args) == 1:
        return 1
    return binom(sum(args), args[-1]) * multinom(args[:-1])

# X = multinomial distribution with params *args = p1,...,pn
class multinom_dist():
    def __init__(self, *args):
        # self.n = n
        self.p = list(args)

    # computes probability of X1=k1,...,Xn=kn, *args = k1,...,kn
    def prob(self, *args):
        if len(args) != len(self.p):
            print("unmatched parameters")
            return
        p = 1
        for i in range(len(args)):
            p *= self.p[i]**args[i]
        return multinom(args) * p


# ===========================
# GOF tests: known/unknown
# ===========================

# return: d = sum(ki^2/npi) - n (or p^_i if using MLE estimates)
def GOF_test_stat(count, np):
    if len(count) != len(np):
        print("unmatched length")
        return
    return sum([count[i]**2/np[i] for i in range(len(count))]) - sum(count)


# ===========================
# GOF test: independence
# ===========================

def GOF_independence(contingency_matrix):
    # 2d array/list; axis-0: along row, axis-1: along column

    matrix = np.array(contingency_matrix)
    n_row, n_col = np.shape(matrix)

    # add row total
    row_total = np.sum(matrix, axis=1)
    row_total = np.reshape(row_total, (n_row, 1))
    matrix = np.append(matrix, row_total, axis=1)

    # add column total
    col_total = np.sum(matrix, axis=0)
    col_total = np.reshape(col_total, (1, n_col+1))
    matrix = np.append(matrix, col_total, axis=0)

    # estimated frequency e_{i,j} = R_i * C_j / n
    est_matrix = col_total[:,:-1] * row_total / matrix[-1,-1]

    # formatting
    trait_A = [chr(i+65) for i in range(n_row)] + ["Col Total"]
    trait_B = [chr(i+65) for i in range(n_col)] + ["Row Total"]

    format_row_top = "{:<12}" * (n_col + 2)
    # first row
    print(format_row_top.format("", *trait_B))
    print("="*12*(n_col+2))
    # other rows
    format_row_1 = "{:<12}" + "{:<12n}" * (n_col+1) 
    format_row_2 = "{:<12}" + "{:<12.3f}"  * (n_col)
    for trait, row, est_row in zip(trait_A[:-1], matrix[:-1,:], est_matrix):
        print(format_row_1.format(trait, *row))
        print(format_row_2.format("", *est_row))
        print("_"*12*(n_col+2))
    # last row
    format_row_bot = "{:<12}" + "{:<12n}" * (n_col + 1)
    print(format_row_bot.format("Col Total", *matrix[-1,:]))

    # test statistic d2 = sum_isum_j (k_ij-e_ij)^2/ne_ij
    sq_diff = (contingency_matrix-est_matrix)**2 / est_matrix
    d2 = np.sum(sq_diff)
    print("\nThe test statistic is d2 = {:.3f}".format(d2))


# ===========================
# Linear Regression
# ===========================


def linear_regression(X, Y, gx=lambda x:x, fy=lambda x:x, verbose=True):
    # X, Y: 1d numpy arrays or lists
    # gx, fy: functions g(x), f(y); default = identity function

    if len(X) != len(Y):
        print("X, Y unmatched size")
        return

    X = np.array(X)
    Y = np.array(Y)
    n = len(X)

    gX = gx(X)
    fY = fy(Y)
    gXfY = gX * fY
    gXgX = gX**2

    b = (n*sum(gXfY) - sum(gX)*sum(fY)) / (n*sum(gXgX) - sum(gX)**2)
    a = (sum(fY) - b*sum(gX)) / n

    # formatting
    if verbose:
        # regression table
        top_row = "{:<12}" * 4
        print(top_row.format("f(x)", "g(y)", "f(x)g(y)", "f(x)^2"))
        print("=" * (12*4))
        format_row = "{:<12.2f}" * 4
        for x, y, xy, x2 in zip(gX, fY, gXfY, gXgX):
            print(format_row.format(x, y, xy, x2))

        # results
        print(format_row.format(sum(gX), sum(fY), sum(gXfY), sum(gXgX)))
        print("\nThe slope is b = {:.4f}, the y-intercept is a = {:.4f}".format(b, a))
        print("The least square line is y = {:.4f} + {:.4f}*x".format(a, b))

        # plot graph
        plt.scatter(gX, fY)
        plt.plot(gX, a+b*gX)
        plt.show()
    else:
        return a, b