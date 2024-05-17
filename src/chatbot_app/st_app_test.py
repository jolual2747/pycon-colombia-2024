import streamlit as st
import asyncio
import time
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
#from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any
from utils import (
    clean_prod_workspace, 
    create_vector_database_from_pdf, 
    create_chatbot,
    AsyncCallbackHandler
)


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, streamlit_text, res_box):
        self.streamlit_text = streamlit_text
        self.current_text = ""
        self.res_box = res_box
        self.llm_response_started = False

    def on_llm_new_token(self, token: str, *args, **kwargs):  
        if not self.llm_response_started:  
            self.res_box.empty()  
            self.llm_response_started = True  
        self.current_text += token  
        self.streamlit_text.markdown(self.current_text) 

# Provide empty implementations for the required methods ... 

def start_over_with_new_document() -> None:
    """
    Deletes objects from Streamlit's session.
    """
    st.session_state.text_input = ''
    # delete the vector store, bot and messages from the session state
    del st.session_state.vs
    del st.session_state.bot
    del st.session_state.messages
    # display message to user
    st.info('Please upload new documents to continue after clearing or updating the current ones.')

async def run_call(agent, query: str, stream_it: AsyncCallbackHandler):
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    await agent.acall(inputs={"input": query})

async def create_gen(agent, query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(agent, query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task

def main() -> None:
    """
    Main of the Streamlit app. 
    """
    st.title("Chat with your documents")
    st.write("Upload your document and start chatting!")

    with st.sidebar:
        st.title("Upload a document to interact with")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf"])
        company_name = st.text_input("Enter Company name:")
        if uploaded_file and st.button("Start chatting!"):
            tmp_route = "./src/chatbot_app/tmp"
            clean_prod_workspace(tmp_route)
            file_name = f"{tmp_route}/{uploaded_file.name}"
            # Guardar el documento en el servidor
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("Document uploaded successfully!")

            st.session_state.vs = create_vector_database_from_pdf(file_name)
            st.session_state.bot = create_chatbot(retriever = st.session_state.vs, company_name = company_name)
            clean_prod_workspace(tmp_route)
    
    if uploaded_file and 'vs' in st.session_state:
        # user's question text input widget
        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome_message = f"I am an AI chatbot from {company_name} ready to help you to answer any questions!"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input('Ask one or more questions about the content of the uploaded data:', key='text_input')
        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            #answer = create_gen(st.session_state.bot, prompt, stream_it)
            with st.chat_message("assistant"):
                # st.markdown(answer)
                # stream_it = StreamlitCallbackHandler(st.empty(), st.empty())
                # response = st.session_state.bot.run({"input": prompt}, callbacks=[stream_it])
                text_container = st.empty()
                full_text = ""
                for chunk in st.session_state.bot.stream({"input": prompt}):
                    print(chunk)
                    words = chunk
                    for word in words:
                        full_text += word + " " 
                        text_container.markdown(full_text) 
                        time.sleep(0.05)

                #st.write_stream(st.session_state.bot.stream({"input": prompt}))
            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.text_input or st.session_state.vs:
            st.button('Start again from new context', on_click=start_over_with_new_document, key='new_question_new_context')



if __name__ == "__main__":
    main()