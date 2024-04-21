import streamlit as st
import os

import pandas as pd
import matplotlib.pyplot as plt 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
df1 = pd.read_csv("insertcsv.csv")
op_key="***********"
os.environ["OPENAI_API_KEY"] = op_key

st.title("QA App for CSV File")


agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    df1,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    ans=agent.run(prompt)
    response = f"Echo: {ans}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})