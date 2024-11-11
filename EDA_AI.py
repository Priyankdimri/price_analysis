import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document  # Use this to structure input documents

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print(os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():
    prompt_template = """
    Your AI intellgent user will provide you query regading find relationship between two features from the database
    you have to explore relation between two features accoding to graph and explain to the use ans also you need to suggest 
    effect on weekly_sales of the feature in context 
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_comment(context):
    chain = get_conversational_chain()
    question = "Can you provide comments on this graph with respect to pricing analysis?"

    # Use Document to properly structure the input
    doc = Document(page_content=context)
    response = chain({"input_documents": [doc], "question": question})
    
    return response["output_text"]

def main():
    import streamlit as st

    st.set_page_config("EDA with Pricing Analysis")
    st.header("EDA with Pricing Analysis using AI solution ")

    data_file = st.file_uploader("Upload your CSV or Excel File", type=['csv', 'xlsx'])
    if data_file is not None:
        if data_file.name.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)

        st.session_state['data'] = df

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_columns = st.multiselect("Select columns for EDA", numeric_columns)

        if selected_columns:
            for col in selected_columns:
                st.subheader(f"Distribution of {col}")
                sns.histplot(df[col], kde=True)
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {col}')
                st.pyplot(plt.gcf())
                plt.clf()

            price_column = st.selectbox("Select the price column for analysis", numeric_columns)

            if price_column:
                for col in selected_columns:
                    if col != price_column:
                        st.subheader(f"Relationship between {col} and {price_column}")
                        sns.scatterplot(x=df[col], y=df[price_column])
                        plt.xlabel(col)
                        plt.ylabel(price_column)
                        plt.title(f'Relationship between {col} and {price_column}')
                        st.pyplot(plt.gcf())
                        plt.clf()

                        # Generate comments on the graph using the model
                        context = f"Here is the relationship between {col} and {price_column}."
                        comment = generate_comment(context)
                        st.write("Comment: ", comment)

if __name__ == "__main__":
    main()