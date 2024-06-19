import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api = os.getenv("GROQ_API_KEY")


st.title("Q/A chatBot with input PDF")

llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate responce based on query
    <context>
    {context}
    <context>
    Question:{input}
    """
)
input_text = st.text_input("Enter your question")

#Createing button so that by pressing that button entire pdfs will be converted into chunks.
#and without pressing that button a question is given then wiki will ans that


embeddings=OllamaEmbeddings()
loader=PyPDFDirectoryLoader("./pdfs")
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) 
final_documents=text_splitter.split_documents(docs[:1])
vect=FAISS.from_documents(final_documents,embeddings)

st.write("Vector Store DB Is Ready")



if input_text:
    start=time.process_time()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vect.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':input_text})
    st.write(response['answer'])
    st.write("Time taken to fin ans",time.process_time()-start)

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
