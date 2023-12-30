import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from decouple import config

def configure():
    config()

embeddings = OpenAIEmbeddings(api_key=config('api_key'))
new_db = FAISS.load_local("faiss_index", embeddings)

llm = ChatOpenAI(api_key=config('api_key'))
qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_db.as_retriever())

def ask(user_query):
    res = qa_chain({"query": user_query})
    return res["result"]

st.title("ChatBoT")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

if prompt := st.chat_input("say"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role":"user","content": prompt})

    response = ask(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant","content":response})
