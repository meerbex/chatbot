# Step 1: Import necessary libraries
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Step 3: Define the prompt template for the RAG
template = """Ты есть коуч и ментор Бобур Тольбаев основатель Мбулак и ответь таком же стиле и в тоне как в контексте и притащи видео и ссылку на него если есть. Ответьте на вопрос, опираясь только на следующий контекст и если нет ответа то откажи ответить:
{context}

Вопрос: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Step 4: Initialize the OpenAI GPT-4 model
model = ChatOpenAI(
    temperature=0, model_name="gpt-4", openai_api_key=st.secrets["openai_api_key"]
)

# Step 5: Setup the Streamlit interface
st.title("ИИ помощник БТ!")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Я твой коуч и ментор. Как я могу помочь?"
        }
    ]


uploaded_file = open("./test.txt", "r", encoding="utf-8")
string_data = uploaded_file.read()

if uploaded_file is not None:
    if "vectorstore" not in st.session_state:
        splitted_data = string_data.split("\n\n")

        # Step 6: Create and configure the vector store
        embedding = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
        st.session_state.vectorstore = FAISS.from_texts(splitted_data, embedding=embedding)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if question := st.chat_input("Введите ваш вопрос"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get AI response
        retriever = st.session_state.vectorstore.as_retriever()
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            response = chain.invoke(question)
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
