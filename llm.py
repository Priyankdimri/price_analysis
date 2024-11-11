import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
st.title("Welcome to RAG base Chat conversation ")
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
# GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

st.write("Upload file here")
pdf_doc = st.file_uploader("Upload file", type=["pdf"])

if pdf_doc is not None:
    pdf_doc.seek(0)  # Reset file pointer to the beginning
    text = ""
    doc = PdfReader(pdf_doc)
    for page in doc.pages:
        text += page.extract_text()
    

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    raw_text=text_splitter.split_text(text)
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    Vector_db=FAISS.from_texts(raw_text,embedding=embedding)
    prompt=("""System:You are intellgent AI Assistant , you have to read all document carefully and you have to 
                        answer to user query from the document {question} if user will ask out of context question you have to say no politly,answer from {context}only 
                        """),
            
    # prompt=("""System:You are intellgent AI Assistant , you have to read all document carefully and you have to 
    #                     answer to user query from the document if user will ask out of context question you have to say no politly,answer from only""")
    promt_temp=ChatPromptTemplate.from_messages(prompt)
  
    parser=StrOutputParser()
    retriver=Vector_db.as_retriever(search_type="similarity",search_kwargs={"k":3})
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    def doc_format(document):
        return "\n\n".join(doc.page_content for doc in document)
    chain={"context":retriver|doc_format,"question":RunnablePassthrough()}|promt_temp|llm|parser
    
    user_input = st.text_input("Enter your Query")  # Ensure the key is unique
    # print(f"*User : {user_input}")
    # if user_input.lower() in ["bye", "quit"]:
        # break
    response = chain.invoke(user_input)  # Using __call__ method
    st.write(response)

# def document_extract(PDF_data):
#     text=""
#     for doc in PDF_data:
#         docs=PdfReader(doc)
#         for pages in docs.pages:
#             text+=pages.extract_text()
#     return text

# def main():
#     PDF=st.file_uploader("Upload the file")
#     if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = document_extract(PDF)
#                 st.write(raw_text)

# if __name__=="__main__":
    # main()















# from azure.ai.openai import OpenAIClient
# from azure.identity import DefaultAzureCredential

# Azure OpenAI setup
# def setup_openai():
#     credential = DefaultAzureCredential()
#     client = OpenAIClient(endpoint="https://<your-openai-endpoint>.openai.azure.com/", credential=credential)
#     return client

# # Function to create the sidebar navigation
# def sidebar_navigation():
#     st.sidebar.title("Navigation")
#     pages = ["Home", "Data Intake", "Exploratory Data Analysis", "Predictive Analytics", "Configuration Setup"]
#     return st.sidebar.radio("Go to", pages)

# # Function to display the Home page
# def home_page():
#     st.title("Pricing Analytics WebApp")
#     st.header("Industry Name")
#     industry = st.selectbox("Select Industry", ["Retail", "Manufacturing", "Logistics"])
#     st.header("AI-based Solutions")
#     st.write("""
#     - Supply Chain Optimization
#     - Demand Forecasting
#     - Sales
#     - Inventory Management
#     - Manufacturing
#     - Logistics
#     - Back Order Management
#     """)

# # Function to display the Data Intake page
# def data_intake_page():
#     st.title("Data Intake")
#     st.write("Upload your data files here.")
#     uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
#     if uploaded_file is not None:
#         st.write("File uploaded successfully!")
#     return uploaded_file

# # Function to display the Exploratory Data Analysis page
# def eda_page(file, client):
#     st.title("Exploratory Data Analysis")
#     df_final = pd.read_csv(file)
#     df_final["Date"] = pd.to_datetime(df_final["Date"], format="mixed")
#     df_numeric = df_final.select_dtypes(include=[np.number])
    
#     plt.figure(figsize=(10, 5))
#     sns.heatmap(df_numeric.corr(method="pearson"), annot=True)
#     st.pyplot(plt)
    
#     # Generate explanation using Azure OpenAI
#     prompt = f"Generate an explanation for the following correlations in the dataset: {df_numeric.corr(method='pearson').to_string()}"
#     response = client.completions.create(engine="davinci", prompt=prompt, max_tokens=500)
#     explanation = response.choices.text.strip()
    
#     st.write(explanation)
    
#     fig, axs = plt.subplots(2, 2, figsize=(18, 10))
#     sns.scatterplot(x=df_final["MarkDown1"], y=df_final["MarkDown5"], ax=axs[0, 0])
#     axs[0, 0].set_title("Relation between MarkDown1 and MarkDown5")
    
#     sns.scatterplot(x=df_final["MarkDown1"], y=df_final["MarkDown4"], ax=axs[0, 1])
#     axs[0, 1].set_title("Relation between MarkDown1 and MarkDown4")
    
#     sns.scatterplot(x=df_final["Date"], y=df_final["Fuel_Price"], ax=axs[1, 0])
#     axs[1, 0].set_title("Relation between Date and Fuel_Price")
    
#     sns.barplot(x=df_final["IsHoliday"], y=df_final["MarkDown3"], ax=axs[1, 1])
#     axs[1, 1].set_title("Relation between IsHoliday and MarkDown3")
    
#     st.pyplot(fig)
    
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x=df_final["Store"], y=df_final["Weekly_Sales"], hue=df_final["Size"])
#     plt.title("Weekly Sales with respect to Store and Store Size")
#     st.pyplot(plt)
    
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x=df_final["IsHoliday"], y=df_final["Weekly_Sales"])
#     plt.title("Weekly Sales with respect to IsHoliday")
#     st.pyplot(plt)
    
#     df_model = df_final.groupby("Date")["Weekly_Sales"].sum().reset_index()
#     from statsmodels.tsa.seasonal import seasonal_decompose
#     decom1 = seasonal_decompose(df_model["Weekly_Sales"], model="additive", period=12)
#     decom1.plot()
#     st.pyplot(plt)

# # Function to display the Predictive Analytics page
# def predictive_analytics_page(file, client):
#     df_final = pd.read_csv(file)
#     df_final["Date"] = pd.to_datetime(df_final["Date"])
#     st.title("Predictive Analytics on Pricing")
#     st.write("Select options for dynamic pricing and view predictive charts.")
#     pricing_option = st.selectbox("Select Pricing Option", ["Competitor-based", "Historical Data-based", "Season-based"])
#     product_filter = st.selectbox("Select Product", ["Product A", "Product B", "Product C"])
#     store_filter = st.selectbox("Select Store", ["Store 1", "Store 2", "Store 3"])

#     # Example of using Azure OpenAI for generating insights
#     prompt = f"Generate insights for {pricing_option} pricing for {product_filter} in {store_filter}."
#     response = client.completions.create(engine="davinci", prompt=prompt, max_tokens=100)
#     st.write(response.choices.text)

# # Main function to run the app
# def main():
#     client = setup_openai()
#     page = sidebar_navigation()
    
#     if page == "Home":
#         home_page()
#     elif page == "Data Intake":
#         file = data_intake_page()
#     elif page == "Exploratory Data Analysis" and file is not None:
#         eda_page(file, client)
#     elif page == "Predictive Analytics" and file is not None:
#         predictive_analytics_page(file, client)

# if __name__ == "__main__":
#     main()
