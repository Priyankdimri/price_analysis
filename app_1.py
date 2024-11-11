import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
import matplotlib.dates as mdates
# ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz='UTC'))


# Function to create the sidebar navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    pages = ["Home", "Data Intake", "Exploratory Data Analysis", "Predictive Analytics", "Configuration Setup"]
    return st.sidebar.radio("Go to", pages)

# Function to display the Home page
def home_page():
    st.title("Pricing Analytics WebApp")
    st.header("Industry Name")
    industry = st.selectbox("Select Industry", ["Retail", "Manufacturing", "Logistics"])
    st.header("AI-based Solutions")
    st.write("""
    - Supply Chain Optimization
    - Demand Forecasting
    - Sales
    - Inventory Management
    - Manufacturing
    - Logistics
    - Back Order Management
    """)

# Function to display the Data Intake page
def data_intake_page():
    st.title("Data Intake")
    st.write("Upload your data files here.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    return None

# Function to display the Exploratory Data Analysis page
def eda_page(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    st.title("Exploratory Data Analysis")
    df_numeric = [column for column in df.columns if df[column].dtype != "object"]
    df_numeric = df[df_numeric]
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_numeric.corr(method="pearson"), annot=True)
    st.pyplot(plt)
    st.write(""" Markdown 1-5 indicate promotions made anonymously, meaning a promotion made on a specific product or in a store.
          There is a very high correlation between Markdown 1 and 4, suggesting that these are related promotions. 
          The correlation between fuel prices and the Date parameter is also very high, indicating that fuel prices must be volatile.
          The correlation between the isHoliday and Markdown-3 parameters may indicate a promotion and discount made only on special 
          days and holidays. There is also a correlation, albeit less than the others, between Weekly Sales and Store Size, meaning 
          that the size of the store may have some impact on weekly sales. The correlation between CPI and Temperature seemed very 
          strange; somehow, they are slightly connected. In other words, where unemployment increases, temperature also increases. 
          The relationship between Temperature and Date is already inevitable, and since the Date has a significant correlation with 
              fuel prices, it also makes it highly correlated with Temperature. The correlation of store size with certain Markdowns
              suggests that these Markdowns might be promotions that can be applied in larger stores or to clear out stock.""")
    
    plt.subplots(2, 2, figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=df["MarkDown1"], y=df["MarkDown5"])
    plt.title("Relation between MarkDown1 and MarkDown5 ")

    plt.subplot(2, 2, 2)
    sns.scatterplot(x=df["MarkDown1"], y=df["MarkDown4"])
    plt.title("Relation between MarkDown1 and MarkDown4 ")

    plt.subplot(2, 2, 3)
    sns.scatterplot(x=df["Date"], y=df["Fuel_Price"])
    plt.title("Relation between Date and Fuel_Price ")

    plt.subplot(2, 2, 4)
    sns.barplot(x=df["IsHoliday"], y=df["MarkDown3"])
    plt.title("Relation between IsHoliday and MarkDown3 ")
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=df["Store"], y=df["Weekly_Sales"], hue=df["Size"])
    plt.title("Weekly Sales with respect to Store and Store Size")
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=df["IsHoliday"], y=df["Weekly_Sales"])
    plt.title("Weekly Sales with respect to IsHoliday ")
    st.pyplot(plt)
    # df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    # st.write(df['Date'])
    # st.write(type(df['Date'].iloc[0]))

    # df['Date'] = pd.to_datetime(df['Date'])
    # df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz='UTC'))

    

# Check the data type of the 'Date' column
    # st.write(df['Date'].dtype)
    # sns.scatterplot(x=df["Date"], y=df["Temperature"])
    # plt.title("Relation between Date and Temperature ")
    # st.pyplot(plt)

    # monthly_sales = df.groupby(pd.Grouper(key='Date', freq='M'))['Weekly_Sales'].sum().round(2)
    # monthly_sales = monthly_sales.astype('int64')
    # fig = px.line(x=list(monthly_sales.index.to_list()), y=list(monthly_sales.values), title='Total Sales by Monthly')
    # fig.update_xaxes(title_text='Date')
    # fig.update_yaxes(title_text='Total Sales in $')
    # fig.update_traces(mode='markers+lines')
    # st.plotly_chart(fig)
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    decom1 = seasonal_decompose(df.set_index('Date')["Weekly_Sales"], model="additive", period=12)
    # decom2 = seasonal_decompose(df.set_index('Date')["Weekly_Sales"], model="multiplicative", period=12)
    decom1.plot()
    # decom2.plot()
    st.pyplot(plt)

# Function to display the Predictive Analytics page
def predictive_analytics_page(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    st.title("Predictive Analytics on Pricing")
    st.write("Select options for dynamic pricing and view predictive charts.")
    pricing_option = st.selectbox("Select Pricing Option", ["Competitor-based", "Historical Data-based", "Season-based"])
    product_filter = st.selectbox("Select Product", ["Product A", "Product B", "Product C"])
    store_filter = st.selectbox("Select Store", ["Store 1", "Store 2", "Store 3"])
    df = df[df["Date"] <= "2012-10-31"]
    df.set_index('Date', inplace=True)  # Set 'Date' column as index
    temp_df = df[(df['Store'] == 1) & (df['Dept'] == 1)]
    monthly_holiday = temp_df.groupby(pd.Grouper(key='Date', freq='M'))['IsHoliday'].sum()
    temp_data = df.groupby(pd.Grouper(key='Date', freq='M'))['Weekly_Sales'].sum().astype('int').reset_index()
    temp_data.set_index('Date')

    train = temp_data.loc[:23, :]
    test = temp_data.loc[24:, :]

    test = test.set_index('Date')
    
    temp_data = temp_data.rename(columns={
        'Date': 'ds',
        'Weekly_Sales': 'y'
    })

    model = Prophet()
    model.fit(temp_data)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

def main():
    page = sidebar_navigation()
    if page == "Home":
        home_page()
    elif page == "Data Intake":
        df = data_intake_page()
        if df is not None:
            st.session_state['df'] = df
    elif page == "Exploratory Data Analysis":
        if 'df' in st.session_state:
            eda_page(st.session_state['df'])
        else:
            st.write("Please upload a data file first.")
    elif page == "Predictive Analytics":
        if 'df' in st.session_state:
            predictive_analytics_page(st.session_state['df'])
        else:
            st.write("Please upload a data file first.")
    elif page == "Configuration Setup":
        st.write("Configuration Setup page")

if __name__ == "__main__":
    main()
