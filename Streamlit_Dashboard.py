import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import os
from datetime import datetime


#path = 'data'
#full_data_path = os.path.join(path, 'final_covid_data.csv')
#test_path = os.path.join(path, 'test_df.csv')
#train_path = os.path.join(path, 'train_df.csv')

full_data = pd.read_csv('final_covid_data.csv')
test = pd.read_csv('test_df.csv')
train = pd.read_csv('train_df.csv')

#full_data.date = pd.to_datetime(full_data.date, format = '%Y-%m-%d')

st.title("""
Forecast COVID-19 Daily New Cases at U.S State Levels:
Integration Method of Machine Learning and Autoregressive Integrated Moving Average Models
""")

st.text("""
Data set was retrieved from COVID-19 Open-Data[1]. The data set consists of variables from 4 categories which
they are COVID-19 related-data, geographic data, search trend of covid-related symptoms on Google search engine,
and government responses. Not all variables from COVID-19 Open-Data were included in our data set. Variables were 
selected based on domain knowledge that only variables are associated with predicting new cases were selected.  

Below shows the data table utilized in the final analysis after data processing and transformation and 
only included 6 rows from the full data table.
""")

st.dataframe(full_data.head(6))

indep_vars = len(full_data.columns)
nums_obs = len(full_data)

st.text(f"""
In total, this data set consists of {nums_obs} observations and {indep_vars} independent variables.
""")


st.sidebar.text("Variable Explainations:")

index_vars_explain = {"subregion1_code": "ISO 3166-2 or NUTS 2/3 code of the subregion. For example, Caifornia = CA",
                      "date": "Calendar data as in format of YYYY-MM-DD starting from 2020-03-02 to 2020-11-10"}

covid_vars_explain = {"new_confirmed": "Count of new cases confirmed after positive test on this date",
                      "cumulative_confirmed": "Cumulative sum of cases confirmed after positive test to date",
                      "cumulative_tested": "Cumulative sum of COVID-19 tests performed to date",
                      "cumulative_recovered": "Cumulative sum of recoveries from a positive COVID-19 case to date"}

geo_vars_explain = {"population_dex": "Population density",
                    "elder_perc": "Percentage of people older than 60 years-old in total population.",
                    "mobility_${place}" : "Percentage change in visits to places compared to baseline"}

policy_vars_explain = {"school/workplace_closing": "0 - no measures;\n 1 - recommend closing or opening with alterations;\n 2 - require closing;\n3 - require closing all levels",
                       "restrictions_on_gatherings": "0 - no restritionc; \n 1 - restrictions on very large gatherings; \n 2 - restrinctions on gatherings between 101-1000 ppl; \n 3 - restrictions on gatherings between 11-100 ppl; \n 4 - restrictions in gatherings of 10 ppl or less",
                       "public_transport_closing": "0 - no measures;\n 1 - recommend closing or opening with alterations;\n 2 - require closing",
                       "stay_at_home_requirements": "0 - no measures;\n 1 - recommend not leaving house;\n 2 - require not leaving house with exceptions for daily exercise, grocery shopping, and 'essential' trips;\n 3 - require not leaving house with minimal exceptions ",
                       "public_information_campaigns": "0 - no Covid-19 public information campaign; \n1 - public officials urging caution about Covid-19; \n 2- coordinated public information campaign (eg across traditional and social media)",
                       "testing_policy" : "0 - no testing policy; \n 1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg came into contact with a known case, returned from overseas); \n 2 - testing of anyone showing Covid-19 symptoms; \n 3 - open public testing (eg 'drive through' testing available to asymptomatic people)",
                       "facial_coverings" : "1 - Recommended; \n 2 - Required in some specified shared/public spaces outside the home with other people present, or some situations when social distancing not possible; \n 3 - Required in all shared/public spaces outside the home with other people present or all situations when social distancing not possible; \n4 - Required outside the home at all times regardless of location or presence of other people",
                       "vaccination_policy" : "0 - No availability;\n 1 - Available for one group of people in priority; \n 2 - Available for two group of people in priority; \n  3 - Available for all people in priority; \n 4 - Available for all people in priority with broad ages; \n 5 - Universal available",
                       "stringency_index": "0 - 100, overall stringency index" }

search_trend_vars_explain = {"search_trend_${covid-related symptons}": "Reflects the normalized search volume for this symptom, for the specified date and region"}


#### Side bar widgets ########
var_cate = ["Search Trend", "COVID-19 related data", "Policy", "Geographic information"]
which_vars = st.sidebar.selectbox("Please select variables for looking up the explanations: ", var_cate, index = 0)

if which_vars == "Search Trend":
    search_trend_vars_explain = pd.DataFrame(search_trend_vars_explain.items(), columns = ["Variables", "Descriptions"])
    st.sidebar.dataframe(search_trend_vars_explain)

elif which_vars == "COVID-19 related data":
    covid_vars_explain = pd.DataFrame(covid_vars_explain.items(), columns = ["Variables", "Descriptions"])
    st.sidebar.dataframe(covid_vars_explain)
elif which_vars == "Policy":
    policy_vars_explain = pd.DataFrame(policy_vars_explain.items(), columns = ["Variables", "Descriptions"])
    st.sidebar.dataframe(policy_vars_explain)
else :
    geo_vars_explain = pd.DataFrame(geo_vars_explain.items(), columns = ["Variables", "Descriptions"])
    st.sidebar.dataframe(geo_vars_explain)

st.sidebar.text("Predictions and Forecast")



#################################


st.subheader("""
Exploratory Data Analysis
""")


st.text("Please input the date range that you want to look at: ")
start_date = st.date_input('Start date')
end_date = st.date_input('End date')

data_start_date = pd.to_datetime('2020-03-02', format='%Y-%m-%d')
data_end_date = pd.to_datetime('2021-11-10', format='%Y-%m-%d')

if start_date < end_date:
    if (start_date < data_start_date) or (end_date > data_end_date):
        st.write("""The starting date in the data set was 2020-03-02, 
        the ending date in the data set was 2021-11-10. 
        Please make sure your time input is within this range.""")
else:
    st.error('Error: End date must fall after start date.')


start_date = "{:%Y-%m-%d}".format(start_date)
end_date = "{:%Y-%m-%d}".format(end_date)
#st.write(start_date)
#st.write(end_date)

all_state = st.selectbox("Do you want to see all states data?", ["Yes", "No"], index = 1)

all_state_list = full_data.subregion1_code.unique()

if all_state == 'Yes':
    final_plot_df = full_data
    fig = px.line(final_plot_df,
            x = 'date',
            y = 'new_confirmed',
            labels = {'subregion1_code': 'State Name'},
            color = 'subregion1_code')
    fig.update_layout(showlegend=True,
        title = "Number of new cases of all states",
        title_x = 0.5,
        xaxis_title = 'date',
        yaxis_title = 'New cases')
    st.write(fig)
else:

    selected_state = st.multiselect("Please select the state(s) you want to look at: ",
                                    all_state_list)

    plot_df = full_data[(full_data.subregion1_code.isin(selected_state))]

    final_plot_df = plot_df[(plot_df.date > start_date) & (plot_df.date < end_date)]
    st.dataframe(final_plot_df)
    cumu_fig = px.line(final_plot_df,
                       x='date',
                       y='cumulative_confirmed',
                       color = 'subregion1_code',
                       labels = {'subregion1_code': 'State Name'})
    cumu_fig.update_layout(
        showlegend=True,
        title = "Number of cumulative confirmed cases in state level(s)",
        title_x = 0.5,
        xaxis_title = 'date',
        yaxis_title = 'cumulative confirmed cases'
    )
    st.plotly_chart(cumu_fig)
    new_fig = px.line(final_plot_df,
                      x='date',
                      y='new_confirmed',
                      color='subregion1_code')
    new_fig.update_layout(
        showlegend=True,
        title=f"Number of new confirmed cases in state level(s)",
        title_x=0.5,
        xaxis_title='date',
        yaxis_title='New cases'
    )

    st.plotly_chart(new_fig)




#########################
#Preidction and Forecast#
#########################

st.subheader("Predict and Forecast by ML and ARIMA Models")

full_final = pd.read_csv('full_final_comp.csv')
full_final_states = full_final.subregion1_code.unique()
side_selectbox = st.sidebar.multiselect("Which state(s) you want to look at for prediction and forcasting: ",
                                      full_final_states, default = 'TX')

final_plot = full_final[full_final.subregion1_code.isin(side_selectbox)]

fig = px.line(final_plot,
                  x = 'date',
                  y = 'value',
                  color = 'model')
fig.update_layout(
        showlegend = True,
        title = f"Comparison of Models",
        title_x = 0.5,
        xaxis_title = 'date',
        yaxis_title = 'New cases'
    )
fig.add_vline(x = '2021-11-05', line_color = 'darkred', line_dash = 'dash')

st.write(fig)
st.write("Left side of the dash line shows prediction values, right side shows forecast.")


