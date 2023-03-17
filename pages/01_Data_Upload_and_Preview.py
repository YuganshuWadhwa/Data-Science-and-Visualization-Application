import streamlit as st
import pandas as pd

# import GUI_class from GUI package
from GUI.GUI_Class import GUI_class


# Setup for page configuration
st.set_page_config(
	page_title = 'Data Upload and Preview',
	layout = 'wide'
	)


# container for dataset selection/upload
dataset_container = st.container()


with dataset_container :

	# loading user-choice from cache in subsequent runs
	if 'index' in st.session_state :
		index = st.session_state['index']

	else :
		index = 0

	options = ['Dataset on Divorce', 'Dataset on Energy', 'Choose your own data']

	# st.selectbox		drop down menu widget
	choice = st.selectbox(label = 'Choose Dataset :', options = options, index = index)

	# cache user-choice in current session state to eliminate dataset selection in subsequent runs
	st.session_state['index'] = options.index(choice)


	if choice == 'Dataset on Divorce' :
		df = pd.read_csv('data/divorce.csv', sep = ';')

		# initializing class object
		GUI_data = GUI_class(df, arg_filename = choice)

		# cache object to use in subsequent application pages
		st.session_state['GUI_data'] = GUI_data

		GUI_data.print_dataframe()
		GUI_data.showInfo()


	elif choice == 'Dataset on Energy' :
		df = pd.read_csv('data/energydata_complete.csv', sep = ',')
		GUI_data = GUI_class(df, arg_filename = choice)

		st.session_state['GUI_data'] = GUI_data

		GUI_data.print_dataframe()
		GUI_data.showInfo()


	elif choice == 'Choose your own data' :

		# st.file_uploader		GUI widget to implement upload function
		uploaded_file = st.file_uploader(label = 'Upload Dataset in .csv Format', type = ['csv'])


		# detect delimiters in uploaded datasets
		if uploaded_file is not None :
			df = pd.read_csv(uploaded_file, sep=';|,', engine='python')
			GUI_data = GUI_class(df)

			st.session_state['GUI_data'] = GUI_data

			GUI_data.print_dataframe()
			GUI_data.showInfo()


		else :
			df = pd.DataFrame()
			GUI_data = GUI_class(df)
			st.session_state['GUI_data'] = GUI_data