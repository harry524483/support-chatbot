import streamlit as st
from langchain_helpers import get_qa_chain, create_vector_db

st.title("Support Chatbot")
btn = st.button("Create Knwoledge Base")

if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer:")
    st.write(response["result"])
