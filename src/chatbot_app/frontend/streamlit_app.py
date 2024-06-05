import streamlit as st
import requests
import time
import base64

company_name = "PyCon Colombia 2024"

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

def response_generator(company_name: str):
    response = f"I am an AI chatbot from {company_name} ready to help you to answer any questions!"
    for word in response.split():
        yield word + " "
        time.sleep(0.005)

def main() -> None:
    """
    Main of the Streamlit app. 
    """
    st.image("pyconco2024.png")
    st.markdown("# PyCon Colombia 2024 😎")
    st.markdown("### Non-official Chabot 🤖")
    st.write("Ask me anything about the event!")
    
    if st.button("Start chat!"):
        st.session_state.start = True

    if "start" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
            welcome_message = f"I am an AI chatbot from {company_name} ready to help you to answer any questions!"
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        prompt = st.chat_input('Ask one or more questions about the content of the uploaded data:', key="text_input")
        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                text_container = st.empty()
                full_text = ""

                s = requests.Session()
                with s.post("http://localhost:8000/chat_stream", stream=True, json={"text": prompt}) as r:
                    for line in r.iter_content():
                        full_text += line.decode("latin-1").replace('`', '') 
                        if full_text.startswith('"'):
                            full_text = full_text[1:]

                        if full_text.endswith('"') or full_text.endswith('}') or full_text.endswith('`'):
                            full_text = full_text[:-1]
                        text_container.write(full_text) 
                        time.sleep(0.005)
            st.session_state.messages.append({"role": "assistant", "content": full_text})

    # if st.session_state.text_input:
    #     st.button('Start again from new context', key='new_question_new_context')



if __name__ == "__main__":
    main()