import asyncio
from typing import Dict, Any
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import AgentExecutor
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from app.services._templates import template_customer_service
from app.services.utils import AsyncCallbackHandler

class ChatBot:
    def __init__(self, retriever: VectorStoreRetriever, company_name: str) -> None:
        self.retriever = retriever
        self.company_name = company_name

    def create_agent(self) -> None:
        """Initialize a ConversationalRetrievalChain as a Customer Service Chatbot.

        Args:
            retriever (VectorStoreRetriever): _description_

        Returns:
            AgentExecutor: _description_
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
        tool = create_retriever_tool(
            self.retriever,
            "knowledge_base",
            'use this tool when answering general knowledge queries to get more information about the topic'
        )
        tools = [tool]

        agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory = conversational_memory,
            get_chat_history=lambda h : h,
            return_intermediate_steps=False
        )
        agent.agent.llm_chain.prompt.messages[0].prompt.template = template_customer_service.replace("{company}", self.company_name)
        self.agent = agent

    # async def run_call(self, query: str, stream_it: AsyncCallbackHandler):
    #     # assign callback handler
    #     self.agent.agent.llm_chain.llm.callbacks = [stream_it]
    #     # now query
    #     await self.agent.acall(inputs={"input": query})

    # async def create_gen(self, query: str, stream_it: AsyncCallbackHandler):
    #     task = asyncio.create_task(self.run_call(self.agent, query, stream_it))
    #     async for token in stream_it.aiter():
    #         yield token
    #     await task


async def run_call(agent: AgentExecutor, query: str, stream_it: AsyncCallbackHandler):
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    await agent.acall(inputs={"input": query})

async def create_gen(agent: AgentExecutor, query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(agent, query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task