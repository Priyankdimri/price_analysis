import streamlit as st
import pandas as pd 
import pickle
import numpy as np  
import seaborn as sns   
import matplotlib as plt   
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.linear_model import Lasso
# model=Ridge(alpha=9)
# from xgboost import XGBRegressor
# import lightgbm as lgb 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go

# import streamlit as st

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
    return uploaded_file

# Function to display the Exploratory Data Analysis page

def eda_page(file):
    st.title("Exploratory Data Analysis")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Streamlit app layout
    
    df_numeric=[column for column in df.columns if df[column].dtype!="object"]
    df_numeric=df[df_numeric]
    plt.figure(figsize=(10,5))
    sns.heatmap(df_numeric.corr(method="pearson"),annot=True)
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
    
# Filter options for features
    st.title('Feature Data Visualization')
    st.sidebar.header('Filter Options')
    features = df.columns.tolist()
    features.remove('Unnamed: 0')  # Remove 'Date' from features list if you want to keep it as a separate filter
    x_feature = st.sidebar.selectbox('Select X-axis Feature', features)
    y_feature = st.sidebar.selectbox('Select Y-axis Feature', features)

# Filter options for stores
    stores = df['Store'].unique()
    selected_stores = st.sidebar.multiselect('Select Stores', stores, default=stores[:2])

# Filter the DataFrame based on the selected stores
    filtered_df = df[df['Store'].isin(selected_stores)]
   
# Streamlit header
    st.header(f'{x_feature} vs {y_feature}')

# Loop through the selected stores
    for store in selected_stores:
    # Filter the data for the specific store
        store_data = filtered_df[filtered_df['Store'] == store]

    # Create a subplot figure with 2 rows and 1 column
        fig = make_subplots(rows=2, cols=1)

    # Add the line plot to the first subplot
        fig.add_trace(
            go.Scatter(x=store_data[x_feature], y=store_data[y_feature], mode='lines', name=f'Store {store}'),
            row=1, col=1
        )

    # Add the scatter plot to the second subplot
        scatter_fig = px.scatter(store_data, x=x_feature, y=y_feature)
        for trace in scatter_fig.data:
            fig.add_trace(trace, row=2, col=1)

    # Update layout for the entire figure
        fig.update_layout(
            height=800, width=1200,  # Adjust the size as needed
            title_text=f"Relation between {x_feature} and {y_feature} for Store {store}",
            xaxis_title=x_feature,
            yaxis_title=y_feature
        )

    # Display the figure in Streamlit
        st.plotly_chart(fig)


# Plot the graph based on the selected features
    # st.header(f'{x_feature} vs {y_feature}')
    # for store in selected_stores:
    #     plt.figure(figsize=(18,10))
    #     store_data = filtered_df[filtered_df['Store'] == store]
    #     plt.subplot(2,1,1)
    #     plt.plot(store_data[x_feature], store_data[y_feature], label=f'Store {store}')
    #     plt.subplot(2,1,2)
    #     sns.scatterplot(x=store_data[x_feature],y=store_data[y_feature],hue=store)
    #     # plt.label(f'Store {store})
    #     # plt.title(f"Relation between {store_data[x_feature]} and {store_data[y_feature]} ")
        

    # plt.xlabel(x_feature)
    # plt.ylabel(y_feature)
    # plt.legend()
    # st.pyplot(plt)
    # # st.pyplot(plt)

    # df_final = pd.read_csv(file)
    # df_final["Date"] = pd.to_datetime(df_final["Date"], format="mixed")
    
    # # Dropdown to select feature
    # feature = st.selectbox("Select Feature for Analysis", df_final.columns)
    
    # # Generate graphs based on selected feature
    # if df_final[feature].dtype != 'object':
    #     plt.figure(figsize=(10, 5))
    #     sns.histplot(df_final[feature], kde=True)
    #     st.pyplot(plt)
        
    #     plt.figure(figsize=(10, 5))
    #     sns.boxplot(x=df_final[feature])
    #     st.pyplot(plt)
        
    #     plt.figure(figsize=(10, 5))
    #     sns.scatterplot(x=df_final["Date"], y=df_final[feature])
    #     st.pyplot(plt)
    # else:
    #     st.write("Selected feature is not numeric. Please select a numeric feature.")
#     st.title("Exploratory Data Analysis")
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
    
#     df_final=pd.read_csv(file)
#     df_final["Date"]=pd.to_datetime(df_final["Date"],format= "mixed")
#     # df1=pd.read_csv(Path2)
#     # df2=pd.read_csv(Path3)
#     # df3=pd.merge(df,df1,on=['Store',"Date","IsHoliday"],how="left")
#     # df_final=pd.merge(df3,df2,on=["Store"],how="left")
#     df_numeric=[column for column in df_final.columns if df_final[column].dtype!="object"]
#     df_numeric=df_final[df_numeric]
    
    
    
#     plt.figure(figsize=(10,5))
#     sns.heatmap(df_numeric.corr(method="pearson"),annot=True)
#     st.pyplot(plt)
#     # plt.show()
#     st.write(""" Markdown 1-5 indicate promotions made anonymously, meaning a promotion made on a specific product or in a store.
#           There is a very high correlation between Markdown 1 and 4, suggesting that these are related promotions. 
#           The correlation between fuel prices and the Date parameter is also very high, indicating that fuel prices must be volatile.
#           The correlation between the isHoliday and Markdown-3 parameters may indicate a promotion and discount made only on special 
#           days and holidays. There is also a correlation, albeit less than the others, between Weekly Sales and Store Size, meaning 
#           that the size of the store may have some impact on weekly sales. The correlation between CPI and Temperature seemed very 
#           strange; somehow, they are slightly connected. In other words, where unemployment increases, temperature also increases. 
#           The relationship between Temperature and Date is already inevitable, and since the Date has a significant correlation with 
#               fuel prices, it also makes it highly correlated with Temperature. The correlation of store size with certain Markdowns
#               suggests that these Markdowns might be promotions that can be applied in larger stores or to clear out stock.""")
    
#     plt.subplots(2,2,figsize=(18,10))
#     plt.subplot(2,2,1)
# # plt.plot(1,2,1)
#     sns.scatterplot(x=df_final["MarkDown1"],y=df_final["MarkDown5"])
#     plt.title("Relation between MarkDown1 and MarkDown5 ")

# # plt.figure(figsize=(10,5))
#     plt.subplot(2,2,2)
#     sns.scatterplot(x=df_final["MarkDown1"],y=df_final["MarkDown4"])
#     plt.title("Relation between MarkDown1 and MarkDown4 ")
# # plt.show()

# # plt.figure(figsize=(10,8))
#     plt.subplot(2,2,3)
# # plt.plot(1,2,1)
#     sns.scatterplot(x=df_final["Date"],y=df_final["Fuel_Price"])
#     plt.title("Relation between Date and Fuel_Price ")
# # plt.show()

# # plt.figure(figsize=(15,5))
#     plt.subplot(2,2,4)
# # plt.plot(1,2,1)
#     sns.barplot(x=df_final["IsHoliday"],y=df_final["MarkDown3"])
#     plt.title("Relation between IsHoliday and MarkDown3 ")
#     st.pyplot(plt)

# # plt.tight_layout()
#     # plt.show()
#     plt.figure(figsize=(10,5))
#     sns.barplot(x=df_final["Store"],y=df_final["Weekly_Sales"],hue=df_final["Size"])
#     plt.title("Weekly Sales with respect to Store and Store Size")

#     st.pyplot(plt)
    
#     plt.figure(figsize=(10,5))
#     sns.barplot(x=df_final["IsHoliday"],y=df_final["Weekly_Sales"])
#     plt.title("Weekly Sales with respect to IsHoliday ")
#     st.pyplot(plt)
#     # sns.scatterplot(x=df_final["Date"],y=df_final["Temperature"])
#     # plt.title("Relation between Date and Temperature ")
#     # st.pyplot(plt)
    
#     # monthly_sales = df_final.groupby(pd.Grouper(key = 'Date', freq = 'ME'))['Weekly_Sales'].sum().round(2)
#     # monthly_sales = monthly_sales.astype('int64')
#     # fig = px.line(x = list(monthly_sales.index.to_list()), y = list(monthly_sales.values), title = 'Total Sales by Monthly')
#     # fig.update_xaxes(title_text = 'Date')
#     # fig.update_yaxes(title_text = 'Total Sales in $')
#     # fig.update_traces(mode = 'markers+lines')
#     # st.pyplot(fig)
#     import statsmodels.api as sm
#     # df_model=df_final[["Date","Weekly_Sales"]]
#     df_model=df_final.groupby("Date")["Weekly_Sales"].sum().reset_index()
#     # df_model.set_index(["Date"],inplace=True)
#     from statsmodels.tsa.seasonal import seasonal_decompose
#     decom1=seasonal_decompose(df_model["Weekly_Sales"],model="additive",period=12)
#     # decom2=seasonal_decompose(df_model["Weekly_Sales"],model="multiplicative",period=12)
#     decom1.plot()
#     # decom2.plot()
#     st.pyplot(plt)
# # plt.show()
# # plt.xticks(rotation=135)
# # plt.show()
    
#     # st.write("Visualize and explore your data here.")

# Function to display the Predictive Analytics page
def predictive_analytics_page(file):
    df_final=pd.read_csv(file)
    df_final["Date"]=pd.to_datetime(df_final["Date"])
    st.title("Predictive Analytics on Pricing")
    st.write("Select options for dynamic pricing and view predictive charts.")
    pricing_option = st.selectbox("Select Pricing Option", ["Competitor-based", "Historical Data-based", "Season-based"])
    product_filter = st.selectbox("Select Product", ["Product A", "Product B", "Product C"])
    store_filter = st.selectbox("Select Store", ["Store 1", "Store 2", "Store 3"])
    import numpy as np
    np.float_ = np.float64
    # py.init_notebook_mode()
    df_final=df_final[df_final["Date"]<="2012-10-31"]
    temp_df = df_final[(df_final['Store'] == 1) & (df_final['Dept'] == 1)]
    monthly_holiday = temp_df.groupby(pd.Grouper(key = 'Date', freq = 'M'))['IsHoliday'].sum()
    temp_data = df_final.groupby(pd.Grouper(key = 'Date', freq = 'M'))['Weekly_Sales'].sum().astype('int').reset_index()
    temp_data.set_index('Date')

    train = temp_data.loc[:23, :]
    test = temp_data.loc[24:, :]

    test = test.set_index('Date')
    
    temp_data = temp_data.rename(columns = {
    'Date':'ds',
    'Weekly_Sales':'y'
        })
    ax = temp_data.set_index('ds').plot(figsize = (10,4))
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales in Month')
    st.pyplot(plt)
    prop_model = Prophet(interval_width = 0.95)
    prop_model.fit(temp_data)
    
    future_dates = prop_model.make_future_dataframe(periods = 36, freq = 'MS')
    forecast = prop_model.predict(future_dates)
    forecast_lookup = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    forecast_lookup['yhat'] = forecast['yhat'].astype('int')
    
    
    fig = prop_model.plot(forecast, uncertainty = True, figsize = (10,5))
    st.pyplot(fig)
    from prophet.plot import add_changepoints_to_plot

    fig = prop_model.plot(forecast, figsize = (11, 4))
    a = add_changepoints_to_plot(fig.gca(), prop_model, forecast)
    st.pyplot(fig)
    st.write("Changepoints are breakpoints in a time series. When we look at the monthly total sales series, we see that there is a breakpoint at the end of each month. This means that sales do not experience smooth increases or decreases; they move very sharply. This suggests that the sharp movements are not due to natural sales patterns but rather supported by high marketing budgets in certain months of the year. The immediate drop after a significant rise each month indicates that there are campaigns encouraging people to buy, and once these campaigns end, consumer interest also drops sharply. However, there could be other reasons as well.")

    st.write(forecast_lookup[["ds","yhat"]].tail(20))
    from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
    y_true = temp_data["y"].values
    y_pred = forecast_lookup["yhat"][:33].values

# Calculate mean squared error
    mse = mean_squared_error(y_true, y_pred)
    MAPE=mean_absolute_percentage_error(y_true, y_pred)
# Calculate root mean squared error
    rmse = np.sqrt(mse)

    st.write("mse: ", rmse)
    st.write("MAPE: ", MAPE)
    st.write(  """ # Recommendations
    ## 1. Increase markdowns during holiday weeks to boost sales.
    ## 2. Focus on stores with higher predicted sales for targeted marketing campaigns.
    ## 3. Optimize inventory levels based on predicted sales to reduce stockouts and overstock situations.
    ## 4. Implement personalized promotions for stores with lower predicted sales to increase foot traffic and revenue.
    """)
    
    # st.write("Dynamic Pricing Charts and Insights will be displayed here.")
    # st.write("Predictive Charts for future demand will be displayed here.")
    # st.write("Factors Influencing Demand will be displayed here.")
    # st.write("What-If Simulator will be displayed here.")
    # st.write("Profit Margin Projections will be displayed here.")
    # st.write("Price Recommendations will be displayed here.")
    # st.write("Side-by-side Price Comparisons will be displayed here.")

# Function to display the Configuration Setup page
def configuration_setup_page():
    st.title("Configuration Setup")
    st.write("pricing strategy parameters and data source integrations here.")
    st.write("Model tuning options will be available here.")

# Main function to run the Streamlit app
def main():
    page = sidebar_navigation()
    if page == "Home":
        home_page()
    elif page == "Data Intake":
        data_intake_page()
    elif page == "Exploratory Data Analysis":
        eda_page(data_intake_page())
    elif page == "Predictive Analytics":
        predictive_analytics_page(data_intake_page())
    elif page == "Configuration Setup":
        configuration_setup_page()

if __name__ == "__main__":
    main()
