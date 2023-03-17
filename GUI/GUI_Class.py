import streamlit as st
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


class GUI_class :
    '''
    Primary Class for GUI Implementation

    Parameters :
    arg_df          user selected/uploaded dataframe
    arg_filename    corresponding filename (if present)
    '''

    def __init__(self, arg_df, arg_filename='') :
        '''
        Constructor
        '''
        self.data = arg_df
        self.original_data = arg_df
        self.filename = arg_filename

        # Extraction of additional information; can be used as per individual requirements of other classes
        self.elements = self.data.size
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.nans = self.data.isna().sum().sum()



    def print_dataframe(self) :
        '''
        Method to display the selected/uploaded dataframe on the interface along with some additional metrics;
        Added functionality to download the dataset
        ''' 

        st.markdown("# ")

        st.dataframe(self.data, use_container_width = True)

        num_elements, num_rows, num_cols, num_nans, download_data = st.columns(5)

        num_elements.metric('Elements', self.elements)
        num_rows.metric('Rows', self.rows)
        num_cols.metric('Columns', self.cols)
        num_nans.metric('Missing (NaN) Values', self.nans)

        # GUI download button
        download_data.download_button(label = 'Download Data', help = 'Download Dataframe as CSV File', data = self.data.to_csv(), mime = 'text/csv')

        # Functionality to remove NULL values and reset index if necessary
        if self.data.isnull().values.any() :
            self.null_and_index()



    def showInfo(self) :
        '''
        Display additional information for the two pre-uploaded datasets
        '''

        if self.filename == 'Dataset on Divorce' :
            self.showInfoDivorce()

        elif self.filename == 'Dataset on Energy' :
            self.showInfoEnergy()



    def showInfoDivorce(self) :
        '''
        GUI implementation to display additional information for 'divorce.csv' dataset (Classification type)
        '''

        st.markdown("# ")

        st.markdown("<h2 style = 'text-align : left; color : #0096c7;'> About The Data </h2>", unsafe_allow_html=True)

        st.markdown("<div style ='text-align: justify;'> Based on the <b>Gottman Couples Therapy</b>, couples were asked, how much they agree or disagree with 54 different questions (the attributes of the dataset), based on the state of their relationship. <br><br>The answers could range from :  </div>", unsafe_allow_html = True)

        st.markdown(""" 

            - **0** : Completely Disagree
            - **1** : Disagree
            - **2** : Neutral
            - **3** : Agree
            - **4** : Completely Agree

            """)

        st.markdown(" <div style ='text-align: justify;'> <br>Among the 170 participants interviewed, 86 belonged to <b>Class 0</b> (Married Couples) and 84 belonged to <b>Class 1</b> (Divorced Couples). The divorced participants were told to consider their past marriage when answering the questions. Only married participants who considered themselves happy, and had never thought of divorce, were included in the study.<br><br>The 54 attributes in the dataset were the ones remaining with a factor load of .40 or higher after performing an Exploratory Factor Analysis. Data collection was performed using face-to-face interviews or digitally. The data could then be used to find patterns to predict whether a couple would split up in the future or not.</div>", unsafe_allow_html = True)


        choice = st.selectbox(label = '' , options = ['Further Information :', 'About The Experimental Group', 'About The Source', 'About The Scientific Background Of The Study'])


        if choice == 'About The Experimental Group' :
            st.markdown(""" 

                - **Region** : Turkey, Europe

                - **170 Participants** : 
                    - 49% Divorced, 51% Married 
                    - 49% Male, 51% Female 
                    - 43.5% Married for Love (74), 56.5% Arranged Marriage (96)
                    - 74.7% Have Children (127), 25.3% Had No Children (43)
                    - **Age** : 20 to 63 (*Arithmetic Average : 36.04, Std. Deviation : 9.34*)  
                
                """)

            st.markdown(""" 

                - **Education** : 
                    - 10.58% were Primary School Graduates (18) 
                    - 8.8% were Secondary School Graduates (15) 
                    - 19.41% were High School Graduates (33) 
                    - 51.76% were College Graduates (88) 
                    - 8.8% had a Master’s Degree (15)

                """)

            st.markdown(""" 

                - **Monthly Income** : 
                    - 20%  had a monthly income *below 2,000 TL* (34) 
                    - 31.76%  had their monthly incomes between *2,001 and 3,000 TL* (54) 
                    - 16.47%  had their monthly incomes between *3,001 and 4,000 TL* (28)
                    - 31.76%  had a monthly income *above 4,000 TL* (54)
                
                <small> <b>*In 2018, 1 TL was roughly between 15 and 27 US$ Cents </b> <i>(Source: Google Finanzen)</i> </small> 
                
                """, unsafe_allow_html = True)


        elif choice == 'About The Source' :
            url = "https://dergipark.org.tr/en/pub/nevsosbilen/issue/46568/549416"

            st.markdown(""" 

                - **Institution :** Nevsehir Haci Bektas Veli University, SBE Dergisi, Turkey

                - **Introduction :** International Congress on Politics, Economics and Social Studies, 2018

                - **Research Article :** [Divorce Prediction using Correlation based Feature Selection and Artificial Neural Networks](%s) <small>(E-ISSN : 2149-3871), 2019</small>

                - **Authors :**
                    - Dr. Ögr. Üyesi Mustafa Kemal Yöntem
                    - Dr. Ögr. Üyesi Kemal Adem
                    - Prof. Dr. Tahsin Ilhan
                    - Ögr. Gör. Serhat Kilicarslan 

                """  % url, unsafe_allow_html = True)


        elif choice == 'About The Scientific Background Of The Study' :
            st.markdown(" <div style ='text-align: justify;'>The <b>Gottman Couples Therapy</b> modelled the reasons for divorce in married couples. Over time, the studies related to this model identified key factors that could cause divorce and showed accuracy when predicting, if a marriage will be long lasting or not. The <b>Divorce Predictors Scale <i>(DPS)</i></b> was developed by <b>Yöntem</b> and <b>Ilhan</b> and is based on the research done by Gottman.<br><br></div>", unsafe_allow_html = True)

            st.markdown(" <div style ='text-align: justify;'>The research group of the study decided to use <b>data mining technologies to predict the possibility of a divorce</b>. Data Mining had been successfully used in other fields of psychology and psychiatry. However, it had not yet been thoroughly used for divorce prediction. The aim of the study was <b>to contribute to the prevention of divorces by predicting them early</b>. Another target of the study was <b>to identify the most significant factors in the DPS that influenced the possibility of a divorce</b>. <br><br></div>", unsafe_allow_html = True)

            st.markdown(""" 

                The team applied the following algorithms to analyse the success of the Divorce Predictors Scale : 
                - Multilayer Perceptron Neural Network 
                - C4.5 Decision Tree algorithms

                """, unsafe_allow_html = True)

            st.markdown(" <div style ='text-align: justify;'><br>The best results were reached using an <b>Artificial Neural Net (ANN)</b> algorithm after selecting the most important 6 questions by applying the correlation-based feature selection method. Overall, the divorce predictors taken from the Gottman couples therapy were confirmed for the Turkish sample group.  <br><br></div>", unsafe_allow_html = True)



    def showInfoEnergy(self) :
        '''
        GUI implementation to display additional information for 'energydata_complete.csv' dataset (Time-series type)
        '''

        st.markdown("# ")

        st.markdown("<h2 style = 'text-align : left; color : #0096c7;'> About The Data </h2>", unsafe_allow_html=True)

        st.markdown("<div style ='text-align: justify;'> Intending to analyze the <b>prediction of the energy consumption of household appliances</b>, scientists conducted measurements in a Belgian four-person household. Beside the energy consumption of the appliances itself, they also measured the energy used for lightning, as well as the temperature and the humidity in different rooms of the house. The measured data were merged over time with the corresponding outside weather conditions, taken from a nearby weather station. </div>", unsafe_allow_html = True)


        choice = st.selectbox(label = '' , options = ['Further Information :', 'About The Data', 'About The Source', 'About The Scientific Background Of The Study'])


        if choice == 'About The Data' :
            st.markdown("<div style ='text-align: justify;'> Data was collected over a timeframe of <b>137 days</b> from Jan. 11 to May 27, 2016, representing weather from winter to beginning of summer in central Europe. Measurements inside the house were taken every 3.3 minutes and averaged to 10 minutes time intervals. The outside weather data was measured in hourly intervals and enriched to 10 minutes time intervals by linear interpolation.<br> </div>", unsafe_allow_html = True)
            st.markdown("<div style ='text-align: justify;'> <br>The house used for the measurements has 280 square meters, but only 220 square meters are heated. The rooms were spread over two floors. The majority of the heating capacity is provided by a wood chimney. Overall, four people live in the house, two adults and two teenagers. One of the adults works from a home-office on a regular basis. The time is recorded in a 10 minutes interval format which includes year, month, day, minutes and seconds. Furthermore, the energy consumption by appliances and lightning applications (in kWh) is given. <br><br> </div>", unsafe_allow_html = True)

            st.markdown("# ")

            image = Image.open('./images/floor_plan.png')
            image = image.resize((700, 350))
            st.image(image, caption='Floor Plan of the House')

            st.markdown("# ")

            st.markdown("""

                The measurements from the house are the following (Temperature **T** in Degrees Celsius, *+- 0.5 degrees accuracy*; Relative Humidity **RH** in Percent, *+- 3% accuracy*) :                 

                """, unsafe_allow_html = True)

            infoenergy = pd.read_csv('data/infoenergy.csv')

            st.write(infoenergy)

            st.markdown("<div style ='text-align: justify;'> The house is equipped with a wide variety of electrical appliances. Also, it is equipped with a heat recovery ventilation and a hot water heat pump. <br> </div>", unsafe_allow_html = True)

            st.markdown("# ")

            st.markdown("""

            The outside weather data include :

            - **Outside Temperature *To*** in degrees *Celsius*
            - **Air Pressure** in *mmHg*    
            - **Outside Humidity *RH_out*** in *%*
            - **Wind Speed** in *meters per second*
            - **Visibility** in *kilometres*
            - **Dewpoint *Tdewpoint*** in *amplitude A degree Celsius*

            The two random variables ***rv1*** and ***rv2*** were introduced to test the Boruta algorithm, which was applied by the scientists to do a selection of relevant variables.

                """, unsafe_allow_html = True)



        elif choice == 'About The Source' :
            url = "https://www.sciencedirect.com/science/article/abs/pii/S0378778816308970?via%3Dihub"

            st.markdown(""" 

                - **Institution :** Thermal Engineering and Combustion Laboratory, University of Mons, Belgium

                - **Introduction :** Published in “Energy and Buildings”, season 140 (2017), Pp. 81-97

                - **Research Article :** [Data driven prediction models of energy use of appliances in a low-energy house](%s)

                - **Authors :**
                    - Luis M. Candanedo
                    - Véronique Feldheim
                    - Dominique Deramaix 

                """  % url, unsafe_allow_html = True)


        elif choice == 'About The Scientific Background Of The Study' :
            st.markdown(" <div style ='text-align: justify;'>Appliances in buildings represent between 20 and 30 % of the demand for electrical energy. Knowledge on demand patterns can be used for different applications like photovoltaic systems sizing. Patterns could be different demands during day and night time, week day or weekend day, season of the year, current outside weather conditions and many more. Light can be seen as an indicator, if a room is currently occupied by inhabitants.</div>", unsafe_allow_html = True)



    def null_and_index(self) :
        '''
        GUI implementation to add functionality of removing NULL values and resetting indices
        Implemented with checkboxes
        '''

        st.markdown("# ")

        RemoveNull, ResetIndex = st.columns(2)

        with RemoveNull :
            # GUI check-box
            agree_null = st.checkbox('Remove Missing Values')

            if agree_null :
                self.data = self.data.dropna()


        with ResetIndex :
            # GUI check-box
            agree_index = st.checkbox('Reset Data Frame Index')

            if agree_index :
                self.data = self.data.reset_index(drop = True)


        if agree_null or agree_index :
            self.print_dataframe()




class GUI_child_Smoothing(GUI_class) :
    '''
    Child Dataclass for Smoothing class
    
    Parameters :
    arg_df          user selected/uploaded dataframe        for parent class GUI_class
    arg_filename    corresponding filename                  for parent class GUI_class
    x_axis          user-input for x_axis                   dataset containing datetime
    x_axis_start    user-input for x_axis_start             starting value of the plot
    x_axis_end      user-input for x_axis_end               ending value of the plot
    y_axis          user-input for y_axis
    window_size     user-input for window window_size       default = 5
    poly_degree     user-input for degree of polynomial     default = 2
    alpha           user-input for alpha values             default = 0
    '''

    def __init__ (self, arg_df, arg_filename, x_axis, x_axis_start, x_axis_end, y_axis, window_size = 5, poly_degree = 2, alpha = 0) :

        GUI_class.__init__(self, arg_df, arg_filename)

        self.x_axis = x_axis
        self.x_axis_start = x_axis_start
        self.x_axis_end = x_axis_end
        self.y_axis = y_axis
        self.window_size = window_size
        self.poly_degree = poly_degree
        self.alpha = alpha