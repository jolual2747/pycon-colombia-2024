import os

import glob
from typing import Dict, Any
import streamlit as st
from chromadb.api.models.Collection import Collection
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.schema import LLMResult
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from _templates import template_customer_service

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
    
    llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.0,
        streaming=True
    )
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    llm2 = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.0,
        streaming=True
    )
    # Retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.invoke,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm2,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory = conversational_memory,
        get_chat_history=lambda h : h,
        return_intermediate_steps=False
    )
    agent.agent.llm_chain.prompt.messages[0].prompt.template = template_customer_service.replace("{company}", company_name)
    return agent

class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False
    
    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""