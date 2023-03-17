import streamlit as st
import matplotlib.pyplot as plt

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class
from Data_Manipulation.Interpolation import Interpolation


# Setup for page configuration
st.set_page_config(
	page_title = 'Data Interpolation',
	layout = 'wide'
	)


def parameter_selection() :
	'''
	GUI implementation to obtain user inputs common to all methods
	'''

	st.markdown('# ')
	st.markdown('# ')

	# declaring global variables tp use outside function
	global Working_Column
	global resample_time
	global graph_limit

	working_column_col, resample_time_col, graph_limit_col = st.columns([1, 1, 1])

	working_options = list(working_df.columns)
	working_options.remove(date_column)

	Working_Column = working_column_col.selectbox(label = 'Select Working Column :', options = working_options, index = 0, help = 'This is only used to display the plot. The selected method with the selected parameters work on complete dataframe.')

	resample_time = resample_time_col.text_input(label = 'Enter Resample Time :', value = '10MIN', help = '')

	graph_limit = graph_limit_col.slider(label = 'Select Number of Points to Plot :', min_value = 20, max_value = len(working_df), value = 100)



# loading information from the cache
df_original = st.session_state['GUI_data'].data
df_Without_Outliers = st.session_state['df_Without_Outliers']
date_column = st.session_state['date_column']

working_df = df_Without_Outliers



header_cont = st.container()
inputs_cont = st.container()
result_cont = st.container()



with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> INTERPOLATION  </h2>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <b>Interpolation is a technique used to estimate missing data points in a time series dataset by filling in the gaps between known data points.</b> </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br> In a time series data, missing values can occur due to various reasons such as data loss, sensor failure, human error, or due to the removal of outliers which can result in incomplete or inaccurate data. Interpolation can help to reduce the impact of missing data on the analysis by approximating the missing values based on the existing data points. </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br> Interpolation is important in data cleaning and manipulation before data analysis because missing data can significantly impact the accuracy of the analysis. If a significant portion of the data is missing, the analysis may not accurately capture the relationship between the variables, which can result in inaccurate predictions. Interpolation can help to reduce the impact of missing data and improve the accuracy of data analysis. </div>", unsafe_allow_html = True)

	st.markdown('# ')
	
	st.markdown(':bulb: <small> <i> :orange[Interpolation works best with Time-series type datasets.] </i> </small>', unsafe_allow_html = True)

	st.markdown('# ')
	
	st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

	st.markdown('''

			- **Working Column** : The column/attribute of the data frame, for which the user wants to see the interpolated plot
			- **Resample Time** : Refers to the sampling resolution you want for the time series data 

			''', unsafe_allow_html = True)

	st.markdown('# ')

	st.markdown(':bulb: <small><i>:red[Tooltip for Resample Time :]</i></small>', unsafe_allow_html = True)

	test_col_1, test_col_2 = st.columns([1, 1])

	test_col_1.markdown('''

		Some valid string aliases representing a fixed frequency offset are : 
		- **'D'** : daily frequency 
		- **'W'** : weekly frequency 
		- **'M'** : month end frequency 
		- **'Q'** : quarter end frequency 
		- **'Y'** : year end frequency

				''', unsafe_allow_html = True)

	test_col_2.markdown('''

		Some valid string aliases representing a custom frequency offset are : 
		- **'2H'** : every 2 hours
		- **'3T' / '3MIN'** : every 3 minutes
		- **'5S'** : every 5 seconds
		- **'80MIN' / '1H20MIN** : every 80 minutes or every 1 hour 20 minutes
		- **'1D10H** : every 1 day and 10 hours

				''', unsafe_allow_html = True)

	st.markdown('# ')
	
	st.dataframe(working_df)

	st.markdown('# ')
	st.markdown('# ')



with inputs_cont :
	choice_text, choice_box = st.columns([1, 2])

	# selection of method
	choice_text.markdown('Select Method to Interpolate the Data :')
	method = choice_box.selectbox(label = '', label_visibility = 'collapsed', options = ['Forward Fill Method', 'Linear Interpolation', 'Cubic Interpolation', 'Spline Interpolation'], index = 0)


	if method == 'Forward Fill Method' :
		st.markdown("<div style ='text-align: justify;'><br>In the <b>Forward Fill Method</b>, when a missing value occurs in a time series, this method replaces it with the last observed value prior to the missing value until the next known value is reached. This means that the missing value is filled with the most recently observed value. <br><br><b>This method is particularly useful in cases where the missing data occurs in short intervals or where the data is expected to remain constant over time.</b></div>", unsafe_allow_html = True)

		# calling the user-input function
		parameter_selection()

		# initializing class object, calling relevant methods and generating results
		inter_object = Interpolation(df = working_df, 
									resample_time = resample_time,  
									date_column = date_column)

		inter_object.ffill()
		inter_object.plot_results(Working_Column = Working_Column, graph_limit = graph_limit)



	elif method == 'Linear Interpolation' :
		st.markdown("<div style ='text-align: justify;'><br><b>Linear Interpolation</b> is a technique used to estimate missing data points in a time series dataset by filling in the gaps between known data points with a straight line. This method assumes that the relationship between the data points is linear and can be approximated by a straight line.<br><br><b>Linear interpolation can be a useful method for estimating missing data points in cases where the relationship between the data points is approximately linear.</b> However, it's important to note that linear interpolation may not be appropriate for all datasets, especially those with complex relationships between the data points. </div>", unsafe_allow_html = True)

		parameter_selection()

		inter_object = Interpolation(df = working_df, 
									resample_time = resample_time,  
									date_column = date_column)

		inter_object.linear()

		inter_object.plot_results(Working_Column = Working_Column, graph_limit = graph_limit)



	elif method == 'Cubic Interpolation' :
		st.markdown("<div style ='text-align: justify;'><br><b>Cubic Interpolation</b> is a technique used to estimate missing data points in a time series dataset by fitting a cubic polynomial to the known data points and using it to interpolate the missing values. This method assumes that the relationship between the data points is smooth and can be approximated by a cubic polynomial. <br><br><b>Cubic interpolation can be a useful method for estimating missing data points in cases where the relationship between the data points is smooth and can be approximated by a cubic polynomial.</b> However, it's important to note that cubic interpolation may not be appropriate for all datasets, especially those with non-smooth relationships between the data points.  </div>", unsafe_allow_html = True)

		parameter_selection()

		inter_object = Interpolation(df = working_df, 
									resample_time = resample_time,  
									date_column = date_column)

		inter_object.cubic()

		inter_object.plot_results(Working_Column = Working_Column, graph_limit = graph_limit)



	elif method == 'Spline Interpolation' :
		st.markdown("<div style ='text-align: justify;'><br><b>Spline Interpolation</b> is a technique used to estimate missing data points in a time series dataset by fitting a piecewise polynomial function to the known data points and using it to interpolate the missing values. This method is often used to interpolate data that has a non-linear relationship between the data points. <br><br><b>Spline interpolation can be a useful method for estimating missing data points in cases where the relationship between the data points is non-linear.</b> It's also a more flexible method compared to other interpolation techniques, such as linear and cubic interpolation, because it can handle data with more complex relationships between the data points. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		parameter_selection()

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Additional Parameter : </h5>", unsafe_allow_html = True)

		st.markdown('''

			- **Order** : The order of the spline determines the degree of the polynomial used to approximate the data. The order can be any non-negative integer, but in practice, the most used values are `1`, `2`, and `3`.
				
				- For order = 1, the function is linear on each interval. 
				- For order = 2, the function is quadratic on each interval. 
				- For order = 3, the function is cubic on each interval. 

			<small> <i>Higher orders can also be used, but they can be more prone to overfitting and can result in oscillations or other undesirable behavior. The specific order to use for spline interpolation depends on the characteristics of the data being interpolated and the desired level of smoothness or flexibility in the interpolated curve. In general, lower-order splines are more appropriate for smooth data with minimal noise, while higher-order splines may be more appropriate for data with more complex patterns or noise. </i> </small>

			''', unsafe_allow_html = True)

		st.markdown('# ')

		order = st.slider(label = 'Select Order :', min_value = 1, max_value = 5, value = 2)

		inter_object = Interpolation(df = working_df, 
									resample_time = resample_time,  
									date_column = date_column,
									order = order)

		inter_object.spline()

		inter_object.plot_results(Working_Column = Working_Column, graph_limit = graph_limit)




with result_cont :
	st.markdown('# ')
	st.markdown('# ')

	st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

	st.markdown('# ')
	st.markdown('**Complete Dataframe after Interpolation :**', unsafe_allow_html = True)

	st.dataframe(inter_object.interpolated_df)

	st.download_button(label = 'Download Interpolated Data', 
					file_name = 'Interpolated_Data.csv',
					data = inter_object.interpolated_df.to_csv(), 
					mime = 'text/csv')

	st.markdown('# ')
	st.pyplot(inter_object.lineplot)

	# cache-ing relevant information to use in subsequent pages
	st.session_state['interpolated_df'] = inter_object.interpolated_df