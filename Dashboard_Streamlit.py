#create a 3 page streamlit dashboard
import json
import zipfile
import geopandas
import random
import numpy as np
import pandas as pd
import streamlit as st
from millify import millify
import random
from streamlit_extras.metric_cards import style_metric_cards
import joblib
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import datetime

# Load the data
@st.cache_data
def read_and_preprocess_data():
    data = pd.read_csv("data/bike-sharing-hourly.csv")
    #create 2 new features that are the lagged data of the cnt column by 1 and 2 hours
    model_data = data.drop(columns=['instant','dteday'])
    model_data = pd.get_dummies(model_data, columns=['holiday', 'weekday', 'weathersit', 'workingday'])
    #model_data['cnt_lag1'] = model_data['cnt'].shift(1)
    #model_data['cnt_lag2'] = model_data['cnt'].shift(2)

    #create 4 new features that are the lagged data of the casual and registered column by 1 and 2 hours
    #model_data['casual_lag1'] = model_data['casual'].shift(1)
    #model_data['casual_lag2'] = model_data['casual'].shift(2)
    #model_data['registered_lag1'] = model_data['registered'].shift(1)
    #model_data['registered_lag2'] = model_data['registered'].shift(2)

    #create new features that are the lagged data of the cnt,casual and registered column by 1 and 2 days
    model_data['cnt_lag24'] = model_data['cnt'].shift(24)
    model_data['casual_lag24'] = model_data['casual'].shift(24)
    model_data['registered_lag24'] = model_data['registered'].shift(24)
    model_data['cnt_lag48'] = model_data['cnt'].shift(48)
    model_data['casual_lag48'] = model_data['casual'].shift(48)
    model_data['registered_lag48'] = model_data['registered'].shift(48)

    #create feature that is the rolling average of the cnt column over the last 1,2,3,4,5,6,7,8,9,10,11,12,24,48 hours
    #model_data['cnt_rolling_avg_2'] = model_data['cnt'].rolling(window=2).mean()
    #model_data['cnt_rolling_avg_3'] = model_data['cnt'].rolling(window=3).mean()
    #model_data['cnt_rolling_avg_4'] = model_data['cnt'].rolling(window=4).mean()
    #model_data['cnt_rolling_avg_5'] = model_data['cnt'].rolling(window=5).mean()
    #model_data['cnt_rolling_avg_6'] = model_data['cnt'].rolling(window=6).mean()
    #model_data['cnt_rolling_avg_7'] = model_data['cnt'].rolling(window=7).mean()
    #model_data['cnt_rolling_avg_8'] = model_data['cnt'].rolling(window=8).mean()
    #model_data['cnt_rolling_avg_9'] = model_data['cnt'].rolling(window=9).mean()
    #model_data['cnt_rolling_avg_10'] = model_data['cnt'].rolling(window=10).mean()
    #model_data['cnt_rolling_avg_11'] = model_data['cnt'].rolling(window=11).mean()
    model_data['cnt_rolling_avg_12'] = model_data['cnt'].rolling(window=12).mean()
    model_data['cnt_rolling_avg_24'] = model_data['cnt'].rolling(window=24).mean()
    model_data['cnt_rolling_avg_48'] = model_data['cnt'].rolling(window=48).mean()

    #create random variable
    model_data['random'] = random.sample(range(1, 17380), data.shape[0])

    #Create feature that is called ridability which is a combination of the temperature, humidity, windspeed and weathersit
    model_data['ridability'] = model_data['temp'] * model_data['hum'] * model_data['windspeed'] * (model_data['weathersit_1'] * 100 +0.01) * (model_data['weathersit_2'] * 64 +0.01) * (model_data['weathersit_3'] * 32 +0.01) * (model_data['weathersit_4'] * 15 +0.01)
    model_data.dropna(inplace=True)
    model_data = model_data.drop(columns=['cnt', 'casual', 'registered'])
    return data, model_data

#Create a function that creates a donut chart that shows the number of bikes being used right now compared to the total number of bikes
def make_donut(value, title, color):
    fig = {
    "data": [
        {
            "values": [value, 1000-value],
            "labels": [title, "'%' of bikes not being used"],
            "domain": {"column": 0},
            "name": title,
            "hoverinfo":"label+percent+name",
            "hole": 0.1,
            "type": "pie",
            "marker": {"colors": [color, "lightgray"]}  # Change the second color here
        }],
    "layout": {
        "title": title,
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": str(value),
                "x": 0.5,
                "y": 0.5
            }
        ]
    }
    }
    st.plotly_chart(fig, use_container_width=True)


#Create a function that creates a bar chart that shows the bike usage throughout the last 24 hours
def make_bar_chart(data):
    # Filter the data to include only the last 24 records
    data_last_24_hours = data.tail(24)

    # Create the bar chart
    fig = px.bar(data_last_24_hours, x='hr', y='cnt', title='Bike Usage Throughout the Last 24 Hours', labels={'cnt': 'Number of Bikes Used', 'hr': 'Hour'})
    st.plotly_chart(fig, use_container_width=True)

#creating a pie chart that shows usage on working days and non working days
def make_pie_chart_workingday(data):
    # Map the 'workingday' column to the desired labels
    data['workingday'] = data['workingday'].map({1: 'Working Day', 0: 'Non-Working Day'})

    # Create the pie chart
    color_map = {'Working Day': '#519DE9', 'Non-Working Day': '#7CC674'}

    fig = px.pie(data, names='workingday', color='workingday', title='Bike Usage by Working Day', color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

#create a pie chart that shows the distibution of total users in whole data set by season
def make_pie_chart(data):
    # Map season values to labels
    season_dict = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
    data['season'] = data['season'].map(season_dict)

    # Define the color map with color codes
    color_map = {'winter': '#DCB12D', 'spring': '#8AAF22', 'summer': '#8871A0', 'fall': '#A2C3DB'}

    # Create the pie chart
    fig = px.pie(data, names='season', color='season', title='Bike Usage by Season', color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

#create a chart that shows the average bike usage by hour
def make_line_chart(data):
    # Create the line chart
    fig = px.histogram(data, x='hr', y='cnt', title='Average Bike Usage by Hour', labels={'cnt': 'Number of Bikes Used', 'hr': 'Hour'})
    st.plotly_chart(fig, use_container_width=True)

#create a histogram chart that shows bike usage with respect to month
def make_line_chart_month(data):
    # Create a dictionary to map month numbers to names
    month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    # Group the data by 'mnth' and calculate the mean 'cnt'
    grouped_data = data.groupby('mnth')['cnt'].mean().reset_index()

    # Replace the month numbers with names
    grouped_data['mnth'] = grouped_data['mnth'].map(month_dict)

    # Create the line chart
    fig = px.bar(grouped_data, x='mnth', y='cnt', title='Average Bike Usage by Month', labels={'cnt': 'Average Number of Bikes Used', 'mnth': 'Month'})
    st.plotly_chart(fig, use_container_width=True)

def make_pie_chart_weathersit(data):
    # Map weathersit values to labels
    weathersit_dict = { 1: 'Clear, Few clouds, Partly cloudy, Partly cloudy', 2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist', 3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds', 4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog' }

    data['weathersit'] = data['weathersit'].map(weathersit_dict)

    # Calculate the counts of each weather situation
    weathersit_counts = data['weathersit'].value_counts()

    # Create a color map
    color_map = {weathersit_counts.index[i]: color for i, color in enumerate(['#A2C3DB','#8AAF22','#DCB12D', '#8871A0'])}

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=weathersit_counts.index, values=weathersit_counts.values, hole=.3, marker=dict(colors=[color_map[i] for i in weathersit_counts.index]))])
    fig.update_layout(title_text='Weather Situation Distribution',
                      title_y=0.9,  # Adjust the position of the title
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=-0,
                                  xanchor="right",
                                  x=0))
    st.plotly_chart(fig)


#Create a graph that shows the distribution of bike usage by weekday histogram
def make_line_chart_weekday(data):
    # Create a dictionary to map weekday numbers to names
    weekday_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Group the data by 'weekday' and calculate the mean 'cnt'
    grouped_data = data.groupby('weekday')['cnt'].mean().reset_index()

    # Replace the weekday numbers with names
    grouped_data['weekday'] = grouped_data['weekday'].map(weekday_dict)

    # Create the line chart
    fig = px.bar(grouped_data, x='weekday', y='cnt', title='Average Bike Usage by Weekday', labels={'cnt': 'Average Number of Bikes Used', 'weekday': 'Weekday'})
    st.plotly_chart(fig, use_container_width=True)

#create a graph showing casual users on working days and non working days by hour
def make_area_chart_casual_users(data):
    # Separate the data into working days and non-working days
    working_days_data = data[data['workingday'] == 1]
    non_working_days_data = data[data['workingday'] == 0]

    # Group the data by 'hr' and calculate the mean 'casual' users
    working_days_users = working_days_data.groupby('hr')['casual'].mean().reset_index()
    non_working_days_users = non_working_days_data.groupby('hr')['casual'].mean().reset_index()

    # Create the area chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=working_days_users['hr'], y=working_days_users['casual'], fill='tozeroy', name='Working Days'))
    fig.add_trace(go.Scatter(x=non_working_days_users['hr'], y=non_working_days_users['casual'], fill='tozeroy', name='Non-Working Days'))

    fig.update_layout(title='Casual Users by Hour', xaxis_title='Hour', yaxis_title='Number of Casual Users')

    st.plotly_chart(fig, use_container_width=True)

def make_area_chart_registered_users(data):
    # Separate the data into working days and non-working days
    working_days_data = data[data['workingday'] == 1]
    non_working_days_data = data[data['workingday'] == 0]

    # Group the data by 'hr' and calculate the mean 'registered' users
    working_days_users = working_days_data.groupby('hr')['registered'].mean().reset_index()
    non_working_days_users = non_working_days_data.groupby('hr')['registered'].mean().reset_index()

    # Create the area chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=working_days_users['hr'], y=working_days_users['registered'], fill='tozeroy', name='Working Days'))
    fig.add_trace(go.Scatter(x=non_working_days_users['hr'], y=non_working_days_users['registered'], fill='tozeroy', name='Non-Working Days'))

    fig.update_layout(title='Registered Users by Hour', xaxis_title='Hour', yaxis_title='Number of Registered Users')

    st.plotly_chart(fig, use_container_width=True)

#create a area chart that shows the difference in bike usage by hour by if the day is a working day or not
def make_area_chart(data):
    # Create the area chart
    fig = px.area(data, x='hr', y='cnt', title='Bike Usage by Hour', labels={'cnt': 'Number of Bikes Used', 'hr': 'Hour'})
    st.plotly_chart(fig, use_container_width=True)

def make_area_chart_users_temp(data):
    # Group the data by 'temp' and calculate the mean 'casual' and 'registered' users
    casual_data = data.groupby('temp')['casual'].mean().reset_index()
    registered_data = data.groupby('temp')['registered'].mean().reset_index()

    # Create the area chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=casual_data['temp'], y=casual_data['casual'], fill='tozeroy', name='Casual Users'))
    fig.add_trace(go.Scatter(x=registered_data['temp'], y=registered_data['registered'], fill='tonexty', name='Registered Users'))

    fig.update_layout(title='Users by Temperature', xaxis_title='Temperature', yaxis_title='Number of Users')

    st.plotly_chart(fig, use_container_width=True)


def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 20,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


def area_chart2(data):
    # Assuming 'data' is your DataFrame and it has 'workingday', 'hr' and 'cnt' columns
    working_day_data = data[data['workingday'] == 1]  # Filter data for working days
    non_working_day_data = data[data['workingday'] == 0]  # Filter data for non-working days

    # Group by hour and calculate the mean bike usage
    working_day_chart_data = working_day_data.groupby('hr')['cnt'].mean().reset_index()
    non_working_day_chart_data = non_working_day_data.groupby('hr')['cnt'].mean().reset_index()

    # Add a new column to distinguish between working and non-working days
    working_day_chart_data['day_type'] = 'Working Day'
    non_working_day_chart_data['day_type'] = 'Non-Working Day'

    # Concatenate the two dataframes
    chart_data = pd.concat([working_day_chart_data, non_working_day_chart_data])

    # Create an Altair chart
    chart = alt.Chart(chart_data).mark_area(opacity=0.3).encode(
        x='hr:Q',
        y='cnt:Q',
        color='day_type:N'
    )

    st.altair_chart(chart, use_container_width=True)

#create a graph that shows cnt by dteday
def make_line_chart_dteday(data):
    # Group the data by 'dteday' and calculate the mean 'cnt'
    grouped_data = data.groupby('dteday')['cnt'].mean().reset_index()

    # Create the line chart
    fig = px.area(grouped_data, x='dteday', y='cnt', title='Bike Usage by Date', labels={'cnt': 'Average Number of Bikes Used', 'dteday': 'Date'})
    st.plotly_chart(fig, use_container_width=True)




#Create a correlation matrix between the variables
def make_corr_matrix(data):
    # Create the correlation matrix
    corr_matrix = data.corr()

    # Create the heatmap
    fig = px.imshow(corr_matrix, title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

#create a chart that shows the correlation between temprature and bike usage
def make_line_chart_users_over_time(data):

    # Group the data by 'dteday' and calculate the sum of 'casual' and 'registered' users
    grouped_data = data.groupby('dteday').agg({'casual': 'sum', 'registered': 'sum'}).reset_index()

    # Create the line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=grouped_data['dteday'], y=grouped_data['casual'], mode='lines', name='Casual Users'))
    fig.add_trace(go.Scatter(x=grouped_data['dteday'], y=grouped_data['registered'], mode='lines', name='Registered Users'))

    fig.update_layout(title='Number of Casual and Registered Users Over Time',
                      xaxis_title='Date',
                      yaxis_title='Number of Users')

    st.plotly_chart(fig, use_container_width=True)

# Create a pie chart that shows the distribution of bike usage by casual and registered users
def make_pie_chart_user(data):
    # Sum up the 'registered' and 'casual' columns
    user_data = data[['registered', 'casual']].sum().reset_index()
    user_data.columns = ['user', 'count']

    # Create the pie chart
    fig = px.pie(user_data, names='user', values='count', title='Bike Usage by User Type')
    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(
        layout="wide", page_title=" :bike: Bike Sharing Data", page_icon=":bike:"
    )

data, model_data = read_and_preprocess_data()

style_metric_cards(border_left_color="#0d09e3")
# Create a title for the app
st.markdown("<h1 style='text-align: center; color: black;'> Bike Sharing Data </h1>", unsafe_allow_html=True)
st.title(':bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike: :bike:')  

# Create a sidebar for the user to select the page they want to view
page = st.sidebar.selectbox('Select a page', ['Current', 'Predictions', 'Data'])

# Create the homepage
if page == 'Current':
    dash_1 = st.container()

    with dash_1:
        st.markdown("<h2 style='text-align: center;'>Current status</h2>", unsafe_allow_html=True)
        st.markdown("<small><i><p style='text-align: center;'>In comparison to 24h</p></i></small>", unsafe_allow_html=True)
        st.write("")

    # Create a dictionary to map weekday numbers to names
    weekday_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Get the current weekday name
    current_weekday = weekday_dict[data['weekday'].iloc[-1]]

    # Get the previous weekday name
    previous_weekday = weekday_dict[data['weekday'].iloc[-24]]  

    weathersit_dict = {
    1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',
    2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
    3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
    4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'
    }

    # Get the current weather situation description
    current_weathersit = weathersit_dict[data['weathersit'].iloc[-1]]
    # creates the container for metric card
    dash_2 = st.container()

    with dash_2:
        currenttemp = data['temp'].iloc[-1]
        col1, col2, col3 = st.columns(3)
        
        # create column span with border
    with col1:
        st.markdown(f"<h2 style='text-align: center; color: black;'>Current Weather Situation</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: black;'>{current_weathersit}</h3>", unsafe_allow_html=True)

    with col2:
        #plot_gauge(data['cnt'].iloc[-12], "#29B09D", " Bikes being used", "Bike utilisation", 1000)
        st.markdown(f"<h2 style='text-align: center; color: black;'>Current Weekday</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: black;'>{current_weekday}</h3>", unsafe_allow_html=True)

    with col3:
        # Display the metric with smaller font size
        #st.markdown(f"<h2 style='text-align: center; color: black;'>Current Weather Situation</h2>", unsafe_allow_html=True)
        #st.markdown(f"<h3 style='text-align: center; color: black;'>{current_weathersit}</h3>", unsafe_allow_html=True)
        plot_gauge(data['cnt'].iloc[-1], "#29B09D", " Bikes being used", "Bike utilisation", 1000)

     # creates the container for metric card
    dash_3 = st.container()

    with dash_3:
        col1, col2, col3 = st.columns(3)
        
        # create column span with border
        with col1:
            st.metric(label="Current windspeed", value= millify(data['windspeed'].iloc[-1], precision=2),delta=millify(data['windspeed'].iloc[-1]-data['windspeed'].iloc[-24], precision=2))

            
        with col2:
            st.metric(label="Current humidity", value=millify(data['hum'].iloc[-1], precision=2),delta=millify(data['hum'].iloc[-1]-data['hum'].iloc[-24], precision=2))
            
        with col3:
            st.metric(label="Temperature", value=millify(data['temp'].iloc[-1]*100, precision=2) + "°C", delta=millify(data['temp'].iloc[-1]-data['temp'].iloc[-24], precision=2))    
    


    make_bar_chart(data)

# Create the data analysis page
if page == 'Predictions':
    st.markdown("<h2 style='text-align: center;'>Predictions</h2>", unsafe_allow_html=True)
    st.write("")

    model = joblib.load('decision_tree_model.pkl')

        # Make predictions for the next 24 hours
    predictions = model.predict(model_data.iloc[-24:])

    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])

        # Create a line chart of the predictions
    import plotly.graph_objects as go

# Create a list of opacities, changing every 4 bars
    opacities = [1,1,1,0.8,0.8,0.8,0.7,0.7,0.7,0.6,0.6,0.6,0.5,0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.2,0.2,0.2,0.1,0.1,0.1,0,0,0]

    dash_10 = st.container()

    with dash_10:
        col1, col2, col3 = st.columns(3)
        
        # create column span with border
        with col1:
            st.metric(label="Next hour prediction", value= millify(predictions[0], precision=2))

        with col2:
            st.metric(label="In 2 hour prediction", value= millify(predictions[1], precision=2))
            
        with col3:
            st.metric(label="In 3 hour prediction", value= millify(predictions[2], precision=2))
# Create a bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=predictions_df.index,
            y=predictions_df['Prediction'],
            marker=dict(
                color='rgb(0,0,225)',
                opacity=opacities
                )
            )
        ])
    st.plotly_chart(fig, use_container_width=True)


# Create the data visualization page
if page == 'Data':

    # Convert 'dteday' column to datetime
    data['dteday'] = pd.to_datetime(data['dteday'])

    # Set default values
    default_start_date = pd.to_datetime('2011-01-01')
    default_end_date = pd.to_datetime('2012-12-31')

    # Get selected date range, or use default values
    start_date, end_date = st.date_input(
        "Select a date range",
        [default_start_date, default_end_date],
        format="MM.DD.YYYY",
    )

    # Convert date to datetime
    start_date = pd.to_datetime(start_date) if start_date else default_start_date
    end_date = pd.to_datetime(end_date) if end_date else default_end_date

    # Filter data for selected date range
    filtered_data = data[(data['dteday'] >= start_date) & (data['dteday'] <= end_date)]

    with st.expander('Data Preview'):
            st.dataframe(data)

    # Add a 3 metric cards with total number of bikes used, Number of casual users and Number of registered users
    
    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Place each metric in a separate column
    with col1:
        st.metric(label="Total Bikes Used", value= millify(filtered_data['cnt'].sum(), precision=2))

    with col2:
        st.metric(label="Total Casual Users", value= millify(filtered_data['casual'].sum(), precision=2))

    with col3:
        st.metric(label="Total Registered Users", value= millify(filtered_data['registered'].sum(), precision=2))

    make_line_chart_dteday(filtered_data)

    #make 2 columns
    col1, col2 = st.columns(2)

    # Place the bar chart in the first column
    with col1:
           make_line_chart_month(filtered_data)

    # Place the donut chart in the second column
    with col2:
        make_pie_chart(filtered_data)

    # Create two columns
    col1, col2 = st.columns(2)

# Place the bar chart in the first column
    with col1:
        make_line_chart(filtered_data)

# Place the donut chart in the second column
    with col2:
        make_pie_chart_weathersit(filtered_data)


    area_chart2(filtered_data)

    areachart1, areachart2 = st.columns(2)

    with areachart1:
        make_area_chart_casual_users(filtered_data)

    with areachart2:
        make_area_chart_registered_users(filtered_data)


    graph1, graph2 = st.columns([3, 1])

    with graph1:
        make_line_chart_users_over_time(filtered_data)

    with graph2:
        make_pie_chart_user(filtered_data)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the correlation matrix in the first column
    with col1:
        make_corr_matrix(data)

    # Display the pie chart in the second column
    with col2:
        make_pie_chart_workingday(filtered_data)

    make_area_chart_users_temp(filtered_data)
