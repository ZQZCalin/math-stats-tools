from math import floor, ceil, sqrt
import numpy as np 

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