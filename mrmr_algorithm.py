import pandas as pd
from sklearn.feature_selection import f_regression


class MRMR(object):

    def __init__(self, X, y, column_names, K):

        self.X = pd.DataFrame(X, columns=column_names)
        self.y = y
        self.column_names = column_names
        self.selected = []
        self.not_selected = self.X.columns.to_list()
        self.K = K

        self.F = None
        self.corr = None
        self.compute_F_statistic()

    def compute_F_statistic(self):
        self.F = pd.Series(f_regression(self.X, self.y)[0], index=self.X.columns)
        self.corr = pd.DataFrame(0.00001, index=self.X.columns, columns=self.X.columns)

    def computing_correlations(self):

        for i in range(self.K):
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
            if i > 0:
                last_selected = self.selected[-1]
                self.corr.loc[self.not_selected, last_selected] = self.X[self.not_selected].corrwith(self.X[last_selected]).abs().clip(.00001)
            # compute FCQ score for all the (currently) excluded features (this is Formula 2)
            score = self.F.loc[self.not_selected]/self.corr.loc[self.not_selected, self.selected].mean(axis=1).fillna(.00001)

            #Find the best feature
            best = score.index[score.argmax()]
            self.selected.append(best)
            self.not_selected.remove(best)

        return self.selected, self.not_selected
