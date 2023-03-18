import pandas as pd                                
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime


class Smoothing_class:

    def __init__(self, GUI_smooth_object):  

        self.data = GUI_smooth_object.data                                                                                              
        self.x = GUI_smooth_object.x_axis
        self.y = GUI_smooth_object.y_axis
        self.x_start = GUI_smooth_object.x_axis_start
        self.x_end = GUI_smooth_object.x_axis_end
        self.window = GUI_smooth_object.window_size                                                               
        self.grade = GUI_smooth_object.poly_degree
        self.alpha = GUI_smooth_object.alpha
        self.index_min = int(self.data.index[self.data[self.x] == self.x_start].to_list()[0])                    # Get start and end of range as integers
        self.index_max = int(self.data.index[self.data[self.x] == self.x_end].to_list()[0])
        self.new_x_axis = self.data[self.x][self.index_min: self.index_max]                                    # Create a new array from the original bigger array
        self.new_y_axis = self.data[self.y][self.index_min: self.index_max]                                    # Indexes = (start to end), not 0 to 1
        self.new_x_axis = self.new_x_axis.reset_index(drop = True)
        self.new_y_axis = self.new_y_axis.reset_index(drop = True)
        self.y_filtered = []
        self.y_filt_complete = []


    def savgol_filter(self):                                                                                                                        
        y_filtered = savgol_filter(self.new_y_axis, self.window, self.grade)      
        x1 = []
        y1 = []
        y2 = []
        for i in range (0, len(y_filtered), 1):
            x1.append(i)
            y1.append(self.new_y_axis[i])
            y2.append(y_filtered[i])
        fig, ax = plt.subplots(figsize=(15, 5))
        sc = ax.plot(x1, y1)
        sc = ax.plot(x1, y2)
        ax.grid()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        self.fig = fig                    



    def mov_average_filter(self):                                         
                                                
        i = 0
        moving_averages = []                                                                
        while i < len(self.new_y_axis) - self.window :                                   
            window_sum = self.new_y_axis[i : i + self.window]                               
            window_average = round(sum(window_sum) / self.window, 2)                        
            moving_averages.append(window_average)                                         
            i += 1                                                              
        for i in range (0, round(self.window/2), 1):                                       
            moving_averages.insert(0,None)                                                  
        for i in range (0, round(self.window/2), 1):
            moving_averages.append(None)
        y_filtered = moving_averages                                                      
        x1 = []
        y1 = []
        y2 = []
        
        for i in range (0, len(self.new_y_axis), 1):
            x1.append(i)
            y1.append(self.new_y_axis[i])
            
        for i in range(0, len(y_filtered), 1):
            y2.append(y_filtered[i])

        fig, ax = plt.subplots(figsize=(15, 5))
        min_len = min(len(y1), len(y2))
        ax.plot(x1, y1,'*-', label= 'Real Data')                                  
        ax.plot(x1[:min_len], y2[:min_len],c= 'black', label= 'Smoothed Data')
        ax.grid()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        self.fig = fig                                                        # Get new array of filtered data and print the result.



    def exponential_filter(self):

        y_filtered = [self.new_y_axis[0]]   
        for i in range(1, len(self.new_y_axis)):
            smoothed_val = self.alpha * self.new_y_axis[i] + (1 - self.alpha) * y_filtered[i-1]
            y_filtered.append(smoothed_val)
        x1 = []
        y1 = []
        y2 = []
        for i in range (0, len(y_filtered), 1):
            x1.append(i)
            y1.append(self.new_y_axis[i])
            y2.append(y_filtered[i])
        fig, ax = plt.subplots(figsize=(15, 5))
        sc = ax.plot(x1, y1)
        sc = ax.plot(x1, y2)
        ax.grid()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        self.fig = fig


    def create_new_df(self, method_name, df_to_change):
        df = self.data
        column_to_skip = self.x
        
        if method_name == 'Savitzky-Golay':
            for column in df.columns:
                if column != column_to_skip:
                    df_to_change[column] = savgol_filter(df[column], self.window, self.grade)


        if method_name == "Moving Average":

            for column in df.columns:
                if column !=column_to_skip:
                    mov_avg_complete = []
                    i = 0
                    while i < len(df[column]) - self.window :                                   
    
                        window_sum_com = df[column][i : i + self.window]                               
                        window_avg_com = round(sum(window_sum_com) / self.window, 2)                        
                        mov_avg_complete.append(window_avg_com)                                         
                        i += 1                                                              
        
                    for i in range (0, round(self.window/2), 1):                                       
                        mov_avg_complete.insert(0,None)   
                                                               
                    for i in range (0, round(self.window/2), 1):
                        mov_avg_complete.append(None)
        
                    self.y_filt_complete = mov_avg_complete
                    
                    if len(df[self.y]) < len(self.y_filt_complete):
                        diff = len(self.y_filt_complete) - len(df[self.y])
                        df[self.y] = np.pad(df[self.y], (0 , diff), 'constant', constant_values = (0))
                    
                    if len(self.y_filt_complete) < len(df[self.y]):
                        diff = len(df[self.y]) - len(self.y_filt_complete)
                        self.y_filt_complete = np.pad(self.y_filt_complete, (0, diff), 'constant', constant_values=(0))
                    
                    df_to_change[column] = self.y_filt_complete

        if method_name == "Exponential":

            for column in df.columns:
                if column != column_to_skip:
            
                    self.y_filt_complete = [df[column][0]]

                    for i in range(1, len(df[column])):
                        smoothed_val_com = self.alpha * df[column][i] + (1 - self.alpha) * self.y_filt_complete[i-1]
                        self.y_filt_complete.append(smoothed_val_com)
                    df_to_change[column] = self.y_filt_complete

       

        return df_to_change