import os
import pandas as pd
import glob
from typing import Dict, Any
import streamlit as st
from chromadb.api.models.Collection import Collection
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from _templates import combine_docs_template_customer_service

def clean_prod_workspace(path: str) -> None:
    """Removes all files in a directory.

    Args:
        path: Path to directory.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError("Directory could not exists or could be misspelled")
    
    path = path + "/*" if not path.endswith("/") else path + "*"
    files = glob.glob(path)
    for file in files:
        try:
            os.remove(file)
        except Exception:
            print('File cannot be removed!')

@st.cache_resource
def create_vector_database_from_pdf(pdf_path: str) -> VectorStoreRetriever:
    """Creates a Vector Database on memory based on a pdf file and returns a Vector Store Retriever.

    Args:
        pdf_path: Path to pdf to convert into a Vector Database.

    Returns:
        VectorStoreRetriever with unstructured data.
    """

    # Load pdf and split it
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        length_function=len,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(data)

    # Create Vector Store with Embeddings
    embedding_openai = OpenAIEmbeddings(model = "text-embedding-ada-002")
    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = embedding_openai
    )

    # vector_store.persist()
    retriever = vector_store.as_retriever(search_kwargs = {"k":4})
    return retriever

def create_chatbot(
        retriever: VectorStoreRetriever,
        company_name: str = None
    ) -> BaseConversationalRetrievalChain:
    """Initialize a ConversationalRetrievalChain as a Customer Service Chatbot.

    Args:
        retriever (VectorStoreRetriever): _description_

    Returns:
        BaseConversationalRetrievalChain: _description_
    """
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.0,
        streaming=True
    )
    condense_question_llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.0
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        verbose=True,
        memory=memory,
        return_generated_question=False,
        retriever=retriever,
        condense_question_llm = condense_question_llm,
        #condense_question_prompt=PromptTemplate.from_template(custom_template),
        get_chat_history=lambda h : h
    )
    qa.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(combine_docs_template_customer_service.replace("{company}", company_name))
    return qa