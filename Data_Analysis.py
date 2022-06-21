from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, normaltest, anderson
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sweetviz as sv
from pandas_profiling import ProfileReport

import os


class Data_Analyse(object):

    def __init__(self):
        pass

    def histogram(self, data, data_name: str, save_path: Optional[str], plot: bool = False):
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        plt.hist(data)
        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(data_name) + '_histogram.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)

    def qqplot_data(self, data, data_name: str, save_path: Optional[str], plot: bool = False):
        """
        A perfect match for the distribution will be shown by a line of dots on a 45-degree angle from the bottom left
         of the plot to the top right.
        :return:
        """
        qqplot(data, line='s')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(data_name) + '_qqplot.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.savefig(plot_name, dpi=400)

    def shapiro_wilk_test(self, data, alpha: float = 0.05):
        '''
        The Shapiro-Wilk test evaluates a data sample and quantifies how likely it is that the data was drawn from a
        Gaussian distribution, named for Samuel Shapiro and Martin Wilk.

        In practice, the Shapiro-Wilk test is believed to be a reliable test of normality, although there is some
        suggestion that the test may be suitable for smaller samples of data, e.g. thousands of observations or fewer.


        :param alpha:
        :return:
        '''

        stat, p = shapiro(data)
        print('Statistics=%.3f, p=%.3f' % (stat, p))

        if float(p) > float(alpha):
            print("Sample looks Gaussian (fail to reject H0")
        else:
            print("Sample does not look Gaussian (reject H0)")

        return stat, p

    def dagostino_k2(self, data, alpha: float = 0.05):

        '''

        The D’Agostino’s K^2 test calculates summary statistics from the data, namely kurtosis and skewness,
        to determine if the data distribution departs from the normal distribution, named for Ralph D’Agostino.

        - Skew is a quantification of how much a distribution is pushed left or right, a measure of asymmetry in the distribution.
        - Kurtosis quantifies how much of the distribution is in the tail. It is a simple and commonly used statistical test for normality.



        :return:
        '''

        k2, p = normaltest(data)
        print('K2=%.3f, p=%.3f' % (k2, p))

        if float(p) > float(alpha):
            print("Sample looks Gaussian (fail to reject H0")
        else:
            print("Sample does not look Gaussian (reject H0)")

        return k2, p, (float(p) > float(alpha))

    def anderson_darling(self, data):
        '''
        Critical values in a statistical test are a range of pre-defined significance boundaries at which the H0 can be failed to be rejected if the calculated statistic is less than the critical value. Rather than just a single p-value, the test returns a critical value for a range of different commonly used significance levels.

        :param data:
        :return:
        '''
        result = anderson(data)
        print('Statistic: %.3f' % result.statistic)
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        return result.statistic

    def heatmap(self, data_x, dataheadings_x, data_name: str, save_path: Optional[str], plot: bool = True):
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        fig, ax = plt.subplots(figsize=(24, 18))

        data_df = pd.DataFrame(data_x, columns=dataheadings_x)
        sns.heatmap(data_df.corr(method='pearson'), cmap="Spectral", annot=True, linewidths=.5)

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(data_name) + '_heatmap.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)

    def box_plot(self, data_x, dataheadings_x, data_name: str, save_path: Optional[str], plot: bool = False):
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        fig, ax = plt.subplots(figsize=(24, 18))
        data_df = pd.DataFrame(data_x, columns=dataheadings_x)
        sns.boxplot(data=data_df, orient="h", palette="Set2")
        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(data_name) + '_box_plot.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)

    def variance_inflation_factor(self, data_x, dataheadings_x):
        '''
        VIF quantifies the severity of multicollinearity in an ordinary least squares regression analysis. It provides an index that measures how much the variance (the square of the estimate’s standard deviation) of an estimated regression coefficient is increased because of collinearity.
        https://en.wikipedia.org/wiki/Variance_inflation_factor

        The square root of the variance inflation factor indicates how much larger the standard error increases compared to if that variable had 0 correlation to other predictor variables in the model.

        Example
        If the variance inflation factor of a predictor variable were 5.27 (√5.27 = 2.3), this means that the standard error for the coefficient of that predictor variable is 2.3 times larger than if that predictor variable had 0 correlation with the other predictor variables.
        :param data_x:
        :param dataheadings_x:
        :return: variance inflation factor. Less than 10 preferred. If receiving INF, then there is perfect colinearity
        '''
        vif = pd.DataFrame()
        vif["features"] = dataheadings_x
        vif["vif_Factor"] = [variance_inflation_factor(data_x, i) for i in range(data_x.shape[1])]
        print(vif)
        return vif

    def sweet_viz(self, data, feature_names,target: Optional[str]=None, save_path: Optional[str]=None):

        if isinstance(data, pd.DataFrame):
            temp_def = data
        else:
            temp_def =  pd.DataFrame(data,columns=feature_names)
        advert_report = sv.analyze(temp_def,target_feat=target)

        file_name = 'data_x_sweet_viz.html'
        file_name = os.path.join(save_path, file_name)
        advert_report.show_html(filepath=file_name,
                 open_browser=True,
                 layout='widescreen',
                 scale=None)

    def pandas_profiling(self, data, save_path):
        prof = ProfileReport(data)
        file_name = 'pandas_profile.html'
        file_name = os.path.join(save_path, file_name)
        prof.to_file(output_file=file_name)