import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class Outliers_Recognization():
    def __init__(self, df, date_column):
        self.df = df
        self.date_column = date_column

    def Z_score(self, threshold):
        z = np.abs(stats.zscore(self.df[self.df.select_dtypes(include=[np.number]).columns]))
        self.df_Without_Outliers = self.df.mask((z > threshold))
        self.df_Without_Outliers[self.date_column] = self.df[self.date_column]

    def Quantile(self, Q1, Q2):
        L_Q = self.df.select_dtypes(include=[np.number]).quantile(Q1, interpolation="nearest")
        U_Q = self.df.select_dtypes(include=[np.number]).quantile(Q2, interpolation="nearest")
        self.df_Without_Outliers = self.df.mask((self.df.select_dtypes(include=[np.number]) > U_Q) | (self.df.select_dtypes(include=[np.number]) < L_Q))
        self.df_Without_Outliers[self.date_column]=self.df[self.date_column]
    
    def Modified_Z_score(self,threshold):
        median = self.df.select_dtypes(include=[np.number]).median()
        MAD = stats.median_abs_deviation(self.df.select_dtypes(include=[np.number]))
        z = 0.6745 * np.abs((self.df[self.df.select_dtypes(include=[np.number]).columns] - median) / MAD)
        self.df_Without_Outliers = self.df.mask((z >= threshold).any(axis=1))
        self.df_Without_Outliers[self.date_column]=self.df[self.date_column]
    
    def Isolation_Forest(self,contamination):
        X = self.df.select_dtypes(include=[np.number]).values
        clf = IsolationForest(random_state=0, contamination=contamination)
        clf.fit(X)
        outliers = clf.predict(X) == -1
        self.df_Without_Outliers = self.df.copy()
        self.df_Without_Outliers.loc[outliers, self.df.select_dtypes(include=[np.number]).columns] = np.nan
        self.df_Without_Outliers[self.date_column]=self.df[self.date_column]
    
    def plot_results(self, Working_Column):
        fig1, ax1 = plt.subplots()
        ax1.boxplot(self.df[Working_Column], vert=False)
        ax1.set_xlabel('Value')
        ax1.set_ylabel(' ')
        ax1.set_title(f'Box Plot of Original Data with Outliers for {Working_Column}')
        array= self.df_Without_Outliers[Working_Column]
        array = array[~np.isnan(array)]
        fig2, ax2 = plt.subplots()
        ax2.boxplot(array, vert=False)
        ax2.set_xlabel('Value')
        ax2.set_ylabel(' ')
        ax2.set_title(f'Box Plot of Data Without Outliers for {Working_Column}')
        fig3, ax3 = plt.subplots(figsize=(15, 5))
        ax3.plot(self.df.index,self.df[Working_Column], label='Data With Outliers', color='b', alpha=1)
        ax3.plot(self.df_Without_Outliers.index, self.df_Without_Outliers[Working_Column], label='Data Without Outliers', color='r', alpha=1)
        ax3.grid()
        ax3.legend()
        ax3.set_xlabel('Samples')
        ax3.set_ylabel('Value ')
        ax3.set_title(f"Line Plot of Outlier Recognization for {Working_Column}")
        self.boxplot_original=fig1
        self.boxplot_filtered=fig2
        self.lineplot=fig3