import streamlit as st
import matplotlib.pyplot as plt

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class, GUI_child_Smoothing
from Data_Manipulation.Smoothing import Smoothing_class


# Setup for page configuration
st.set_page_config(
	page_title = 'Data Smoothening',
	layout = 'wide'
	)


# loading information from the cache
GUI_data = st.session_state['GUI_data']
df_original = st.session_state['GUI_data'].data
interpolated_df = st.session_state['interpolated_df']
date_column = st.session_state['date_column']

working_df = interpolated_df



def xy_axes_selection() :
	'''
	GUI implementation to obtain user inputs common to all methods
	'''

	y_options = list(working_df.columns)

	global x_axis
	global y_axis
	global x_axis_start
	global x_axis_end

	x_axis = date_column

	y_options.remove(x_axis)

	y_axis = st.selectbox(label = 'Select Column to use as Y-Axis :', options = y_options, index = 0)

	st.markdown('# ')

	x_axis_range = st.select_slider(label = 'Select Range for X-Axis :', options = sorted(working_df[x_axis].unique()), value = (sorted(working_df[x_axis])[0], sorted(working_df[x_axis])[-1]))

	st.markdown("Selected Range : &nbsp;&nbsp; *from &nbsp; **%s**  &nbsp; to &nbsp; **%s***" % (str(x_axis_range[0]), str(x_axis_range[1])))

	x_axis_start = x_axis_range[0]
	x_axis_end = x_axis_range[1]

	st.markdown('# ')




header_cont = st.container()
inputs_cont = st.container()
result_cont = st.container()



with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> DATA SMOOTHING  </h2>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <b>Smoothing is applied to reduce the effect of high-frequency noise and bring out key aspects of the signal, such as trends, patterns, and anomalies.</b> </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br> Time series analysis, image processing, signal processing, and data analysis are a few examples of the many applications it is used in. It can be performed by using several types of filters. Smoothing can improve the accuracy of analyses and forecasts in each of these applications, as well as give a clearer perspective of critical elements in the data. </div>", unsafe_allow_html = True)

	st.markdown('# ')

	st.dataframe(working_df)

	st.markdown('# ')
	st.markdown('# ')


with inputs_cont :
	choice_text, choice_box = st.columns([1, 2])

	# selection of method
	choice_text.markdown('Select Method to Smoothen the Data :')
	method = choice_box.selectbox(label = '', label_visibility = 'collapsed', options = ['Savitzky-Golay', 'Moving Average', 'Exponential'], index = 0)



	if method == 'Savitzky-Golay' :
		st.markdown("<h4 style = 'text-align : left; color : #0096c7;'> Savitzky-Golay Filter </h4>", unsafe_allow_html = True)

		st.markdown("<div style ='text-align: justify;'> The Savitzky-Golay smoothing filter works by fitting a polynomial of a certain order to a set of data points and using this polynomial to estimate the value of the signal at any given point. It is useful in situations where it is important to preserve features such as peaks and valleys in the data. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

			- **Window Size** : Refers to the number of data points used to calculate the smoothed value for a given point
			- **Degree of Polynomial** : Refers to the order of the polynomial that is fit to the data within the window

			''', unsafe_allow_html = True)

		st.markdown('# ')
		st.markdown('# ')

		xy_axes_selection()

		window_col, test_col, poly_degree_col = st.columns([1.5, 0.2, 1.5])

		window_size = window_col.slider(label = 'Select the Window Size :', min_value = 2, max_value = 15, step = 1, value = 5)
		poly_degree = poly_degree_col.slider(label = 'Select the Degree of Polynomial :', min_value = 1, max_value = 5, step = 1, value = 2)

		# initializing dataclass object with user-inputs
		GUI_smoothing_data = GUI_child_Smoothing(working_df, GUI_data.filename, x_axis, x_axis_start, x_axis_end, y_axis, window_size, poly_degree)

		# initializing class object with dataclass object
		Smoothing_data = Smoothing_class(GUI_smoothing_data)

		# calling relevant methods
		Smoothing_data.savgol_filter()



	elif method == 'Moving Average' :
		st.markdown("<h4 style = 'text-align : left; color : #0096c7;'> Moving Average Filter  </h4>", unsafe_allow_html = True)

		st.markdown("<div style ='text-align: justify;'> The moving average filter works by taking the average of a set of data points over a certain window size, and using this average as the estimate of the signal value at any given point. It is used for signal processing, finance, and engineering, where the goal is to remove high-frequency noise and obtain a clearer representation of the underlying signal. </div>", unsafe_allow_html = True)

		st.markdown('### ')

		st.markdown(':bulb: <small> <i> :orange[After performing Moving Average, some rows will be deleted.] </i> </small>', unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

			- **Window Size** : Refers to the number of data points used to calculate the smoothed value for a given point

			''', unsafe_allow_html = True)

		st.markdown('# ')
		st.markdown('# ')

		xy_axes_selection()

		window_size = st.slider(label = 'Select the Window Size :', min_value = 2, max_value = 15, step = 1, value = 5)

		GUI_smoothing_data = GUI_child_Smoothing(working_df, GUI_data.filename, x_axis, x_axis_start, x_axis_end, y_axis, window_size)

		Smoothing_data = Smoothing_class(GUI_smoothing_data)

		Smoothing_data.mov_average_filter()



	elif method == 'Exponential' :
		st.markdown("<h4 style = 'text-align : left; color : #0096c7;'> Exponential Filter  </h4>", unsafe_allow_html = True)

		st.markdown("<div style ='text-align: justify;'> Exponential smoothing works by weighting the past data points in a signal exponentially, with more recent data points receiving higher weight than older data points. The smoothed signal value at any given point is a weighted average of the past data points, with the weights decaying exponentially over time. It is particularly useful when the signal has trends or seasonality, as it can effectively capture these patterns. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters :  </h5>", unsafe_allow_html = True)

		st.markdown('''

			- **Alpha** : The smoothing factor that controls the weight given to past values in the calculation of a smoothed value. A value of alpha close to 1 gives a high weight to recent observations, while a value of alpha close to 0 gives a low weight to recent observations, and results in a smoother, less responsive forecast.

			''', unsafe_allow_html = True)

		st.markdown('# ')
		st.markdown('# ')

		xy_axes_selection()

		alpha = st.slider(label = 'Select the value of Alpha :', min_value = 0., max_value = 1., step = 0.1, value = 0.5)

		GUI_smoothing_data = GUI_child_Smoothing(working_df, GUI_data.filename, x_axis, x_axis_start, x_axis_end, y_axis, alpha = alpha)

		Smoothing_data = Smoothing_class(GUI_smoothing_data)

		Smoothing_data.exponential_filter()



with result_cont :
	st.markdown('# ')
	st.markdown('# ')

	st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

	st.pyplot(Smoothing_data.fig)

	st.markdown('# ')

	st.session_state['perform_smoothing'] = False
	st.session_state['perform_smoothing'] = st.button(label = 'Smoothen Complete Dataframe', type = 'primary')

	if st.session_state['perform_smoothing'] :
		st.markdown('# ')
		st.markdown('**Complete Dataframe after Smoothing :**', unsafe_allow_html = True)

		smoothed_df = Smoothing_data.create_new_df(method_name = method, df_to_change = working_df)

		st.dataframe(smoothed_df)

		st.session_state['smoothed_df'] = smoothed_df

		st.download_button(label = 'Download Smoothed Data', 
					file_name = 'Smoothed_Data.csv',
					data = smoothed_df.to_csv(), 
					mime = 'text/csv')

	st.session_state['perform_smoothing'] = False