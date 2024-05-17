import os
import glob
from typing import Any
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import LLMResult
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

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