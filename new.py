import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(r"C:\Users\priyankd3\Downloads\retails_dataset\Features data set.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Streamlit app layout
st.title('Feature Data Visualization')
st.sidebar.header('Filter Options')

# Filter options for features
features = df.columns.tolist()
features.remove('Date')  # Remove 'Date' from features list if you want to keep it as a separate filter
x_feature = st.sidebar.selectbox('Select X-axis Feature', features)
y_feature = st.sidebar.selectbox('Select Y-axis Feature', features)

# Filter options for stores
stores = df['Store'].unique()
selected_stores = st.sidebar.multiselect('Select Stores', stores, default=stores[:2])

# Filter the DataFrame based on the selected stores
filtered_df = df[df['Store'].isin(selected_stores)]

# Plot the graph based on the selected features
st.header(f'{x_feature} vs {y_feature}')
for store in selected_stores:
    store_data = filtered_df[filtered_df['Store'] == store]
    plt.plot(store_data[x_feature], store_data[y_feature], label=f'Store {store}')

plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.legend()
st.pyplot(plt)
