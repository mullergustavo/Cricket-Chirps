# https://cricketchirpsvstemperature.streamlit.app

import streamlit as st
from PIL import Image
import pandas as pd
import xlrd
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

#config
st.set_page_config(page_title='Cricket Chirps vs. Temperature | Calculator', page_icon='📈')

cricket_vs_temperature = Image.open('cricket_vs_temperature.jpg')

df = pd.read_excel('https://raw.githubusercontent.com/mullergustavo/Cricket-Chirps/main/slr02.xls', engine='xlrd')

st.title('Cricket Chirps vs. Temperature')
st.image(cricket_vs_temperature)
st.write('As the temperature increases, crickets chirp faster. This is because temperature affects the rate at which cricket muscles contract. When it is warmer, cricket muscles contract faster, causing them to chirp faster.')
st.write('Below there is a scatter plot with the values of cricket chirps vs. temperature in degrees fahrenheit recorded. There are three tabs, each plotting the same data, but using different libraries in order to see the pros of each one.')

tab1, tab2, tab3 = st.tabs(['Plotly', 'Altair', 'Pyplot'])

with tab1:
    fig = px.scatter(
        df,
        x='X',
        y='Y',
        labels={'X': 'Chirps/sec for the Striped Ground Cricket', 'Y': 'Temperature in degrees Fahrenheit'},
        title='Chrips vs. Temperature Chart')
    st.plotly_chart(fig, theme='streamlit')
        
with tab2:
    chart = alt.Chart(df, title='Chrips vs. Temperature Chart').mark_circle().encode(
        alt.X('X', title='Chirps/sec for the Striped Ground Cricket', scale=alt.Scale(domain=[14, 20.5])),
        alt.Y('Y', title='Temperature in degrees Fahrenheit', scale=alt.Scale(domain=[67, 95]))).interactive().configure_title(anchor='middle')
    st.altair_chart(chart, theme='streamlit')

with tab3:
    fig, ax = plt.subplots()
    ax.scatter(df['X'], df['Y'])
    plt.title('Chrips vs. Temperature Chart')
    plt.xlabel('Chirps/sec for the Striped Ground Cricket')
    plt.ylabel('Temperature in degrees Fahrenheit')
    st.pyplot(fig)

if st.checkbox('Display interective original dataframe'):
    st.dataframe(df)

st.write('Source: [Houghton Mifflin Harcourt](https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr02.html)')

st.divider()

X = df['X'].values.reshape(-1,1)
y = df['Y'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=23)

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_score = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader('Cricket Chirps vs. Temperature Calculator')
st.write('The following calculator can be used to determine the degrees fahrenheit according to the number of Chirps/sec for the striped ground cricket you recorded')
x_input = st.slider('Select the number of Chirps/sec')
y_input = float(model.predict(np.array(x_input).reshape(-1, 1)))

if x_input != 0:
    st.write('At', str(x_input), 'Chirps, the average temperature is %.2f °F' % y_input)
    if st.checkbox('Display the results of the calculation using Linear Regression'):
        st.write('Coefficient: %.2f' % model.coef_)
        st.write('Intercept: %.2f' % model.intercept_)
        st.write('R2 Score: %.2f' % r2_score)
        st.write('RMSE: %.2f' % rmse)

    fig, ax = plt.subplots()
    ax.scatter(df['X'], df['Y'])
    plt.scatter(x_input, y_input, color="red")
    plt.plot(X_test, y_pred, color='red')
    plt.title('Chrips vs. Temperature Chart')
    plt.xlabel('Chirps/sec for the Striped Ground Cricket')
    plt.ylabel('Temperature in degrees Fahrenheit')
    st.pyplot(fig)

st.divider()

st.write("Do you have questions, suggestions, or want to contact me? Visit my profile on [LinkedIn](https://www.linkedin.com/in/gustavomuller) or [GitHub](https://github.com/mullergustavo)")
