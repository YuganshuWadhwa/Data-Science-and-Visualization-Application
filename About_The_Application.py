import streamlit as st  					#pip install streamlit
import requests								# pip install requests
from streamlit_lottie import st_lottie		# pip install streamlit-lottie



def load_lottieurl(url: str):
	'''
	Function to generate animations on the interface using streamlit_lottie

	Parameters :
	url 	url of the animation
	'''
	r = requests.get(url)

	if r.status_code != 200:
		return None

	return r.json()


# Setup for page configuration in local streamlit application
st.set_page_config(
	page_title = 'Yuganshu Wadhwa',
	layout = 'wide'
	)




# Defining containers for headings, features, navigation
# Containers and columns are used to hold multiple GUI elements

# st.container 		define streamlit container
# st.columns		define streamlit column(s)		the passed list defines the proportional size of each column
header_cont = st.container()
header_text_col, header_animation_col = st.columns([1.5, 1])
features_cont = st.container()
navigation_cont = st.container()



# Working with the header container
with header_cont :

	# st.markdown		display string formatted as Markdown
	st.markdown("<h1 style = 'text-align : center; color : #023e8a;'> DATA SCIENCE AND VISUALIZATION APPLICATION </h1>", unsafe_allow_html=True)

	st.markdown("<h3 style = 'text-align : center; color : #0077b6;'> Master Automation and IT &nbsp;&nbsp; 2022-23 </h3>", unsafe_allow_html = True)

	st.markdown("# ")



with header_text_col :

	st.markdown("<div style ='text-align: justify;'> An Application for Data Science and Visualization made by students of the Master of Automation and IT program. To offer a hands-on, visual, user-friendly experience with data, the application draws on a range of fields like data manipulation, regression, classification, and artificial intelligence. In order to make use of all these concepts, at each level of the program, the user is given a variety of features along with textual prompts and visual outputs to analyze the inputs and outputs. </div>", unsafe_allow_html = True)



with header_animation_col :
	header_animation = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json')

	# using imported function to generate animation
	st_lottie(
		header_animation, 
		quality = 'high',
		height = 160,
		width = 300 	
		)



with features_cont : 
	st.markdown("<h2 style = 'text-align : left; color : #0096c7;'> Features of the Application </h2>", unsafe_allow_html = True)

	st.markdown(""" 

		- **Data Selection :**
			- Provides the user with the option to ***upload their own dataset*** or work with one of the ***two pre-defined datasets***
			- Works on Comma Seperated Value (***CSV***) files
			- Automatic ***delimiter detection*** for the CSV files

		""")

	st.markdown("""

		- **Data Preview :**
			- Provides the user with a ***preview of the selected dataset in the form of a dataframe***
			- Displays ***statistical information*** for the selected dataset (along with ***information about the scientific background for the pre-defined datasets***)
			- Provides the user with additional features to ***remove NaN values*** from the dataset (if any), and ***reset the index***

		""")

	st.markdown("""

		- **Data Smoothing**, **Interpolation**, and **Outlier Recognition :**
			- Allows the user to ***smoothen***, ***perform interpolation***, and ***perform outlier recognition on the used dataset*** using various methods
			- Allows the user to tune the ***parameters*** regarding each method and analyze the corresponding effects
			- Provides the user with ***information about each method and the corresponding parameters***
			- Allows the user to ***download filtered datasets*** after outlier recognition

		""")

	st.markdown("""

		- **AI Based Classification** and **Regression :**
			- Allows the user to ***perform classification on classification-type datasets*** 
			- Allows the user to ***perform regression on time-series type datasets***
			- Provides the user with ***information about each method and the corresponding parameters***
			- Provides the user with necessary ***textual and graphical results***
			- Allows the user to ***save the models after training***
			- Allows the user to ***upload pre-trained models*** to test the accuracy 

		""")

	st.markdown("""

		- **ML Based Classification** and **Regression :**
			- Allows the user to ***perform classification on classification-type datasets*** 
			- Allows the user to ***perform regression on time-series type datasets*** 
			- Provides the user with ***information about each method and the corresponding parameters***
			- Provides the user with necessary ***textual and graphical results***
			- Allows the user to ***test the trained models*** by uploading test datasets

		""")

	st.markdown("# ")



with navigation_cont : 
	st.markdown("<h2 style = 'text-align : left; color : #0096c7;'> Navigation </h2>", unsafe_allow_html = True)

	st.markdown("""

	- The application is structured as a ***collection of multiple pages*** performing certain individual functions.
	- These pages can be ***accessed from the side-bar***.
	- On the page ***"Data Upload and Preview"***, the working dataset can be chosen or uploaded.
	- It is necessary to ***load the "Data Upload and Preview"*** page and ***select / upload / change the working dataset***, before proceeding any further through the application.
	- In order to generate the ***Processed Data*** for regression, please use the ***Outlier Recognition***, ***Interpolation*** and ***Smoothing*** (in the mentioned order).
	- After choosing the desired parameters on the "Data Smoothing" page, ***please press "Smoothen Complete Dataframe"*** to finally generate processed data.
	- While training new AI models, the trained models are ***automatically saved*** in the working root directory.
	- On each page, some ***generic*** and ***method- specific*** information is provided.
	- Look out for ***hints*** :bulb: on each page for further assistance regarding the ***type of datasets the page works best with***.
	- In some of the pages, ***tooltips*** :grey_question: are provided for further assistance regarding the ***parameters***.

	""")



st.markdown("# ")

st.markdown(':ledger: <small> :blue[[**Documentation**](https://mait-oop22-23-documentation.readthedocs.io/en/latest/)]', unsafe_allow_html=True)
st.markdown(':open_file_folder: <small> :blue[[**Project Repository**](https://github.com/YuganshuWadhwa/Data-Science-and-Visualization-Application)]', unsafe_allow_html=True)