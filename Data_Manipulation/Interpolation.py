import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

"""
Class: Interpolation

This class is used for interpolation in a time series dataset. Four different methods are used for interpolation. 

Methods: 1. Forward fill
         2. Linear
         3. Cubic
         4. Spline
"""

class Interpolation:

    def __init__(self, df, resample_time,date_column,order=2):
        self.df = df
        self.resample_time=resample_time
        self.date_column=date_column
        self.order=order
        self.set_index()

    def set_index(self):
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])    
        self.df = self.df.set_index(self.df[self.date_column])                   
        self.df = self.df.resample(self.resample_time).mean().reset_index()
        for column in self.df.select_dtypes(include=[np.number]):
            self.df[column].iloc[0] = self.df[column].iloc[self.df[column].first_valid_index()]
        for column in self.df.select_dtypes(include=[np.number]):
            self.df[column].iloc[-1] = self.df[column].fillna(method='ffill', limit=len(self.df)).iloc[-1]
    
    def ffill(self):
        self.interpolated_df = self.df.copy()
        self.interpolated_df = self.interpolated_df.ffill()
    
    def linear(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.interpolated_df = self.df.copy()
        self.interpolated_df[num_cols] = self.interpolated_df[num_cols].interpolate(method='linear', limit_direction='both')  
    
    def cubic(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.interpolated_df = self.df.copy()
        self.interpolated_df[num_cols] = self.interpolated_df[num_cols].interpolate(method='cubic', limit_direction='both')  
        self.interpolated_df[num_cols] = self.interpolated_df[num_cols].clip(lower=0)
        for col in num_cols:
            max_val_before = self.df[col].max()
            self.interpolated_df[col] = self.interpolated_df[col].clip(upper=max_val_before)
    
    def spline(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.interpolated_df = self.df.copy()
        self.interpolated_df[num_cols] = self.interpolated_df[num_cols].interpolate(method='spline',order= self.order, limit_direction='both')  
        self.interpolated_df[num_cols] = self.interpolated_df[num_cols].clip(lower=0)
        for col in num_cols:
            max_val_before = self.df[col].max()
            self.interpolated_df[col] = self.interpolated_df[col].clip(upper=max_val_before)
    
    def plot_results(self, Working_Column,graph_limit):
        fig1, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(self.interpolated_df.index[:graph_limit], self.interpolated_df[Working_Column][:graph_limit], label='Interpolated Data',color='r', alpha=0.8)
        ax1.plot(self.df.index[:graph_limit], self.df[Working_Column][:graph_limit], label='Original Data',color='b', alpha=0.8)
        ax1.grid()
        ax1.legend()
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Value ')
        ax1.set_title(f"Line Plot of Data Interpolation for {Working_Column}")
        self.lineplot=fig1