__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import time
import json
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb import chromadb
# DataTransformer - json format
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
from function import DataTransformer
from chatbot_class import Chatbot

# Load environment variables
load_dotenv()

# Initialize necessary data once and cache it
@st.cache_resource
def initialize():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Data loading and chopping
    json_directory = "./dataset/common_senses"
    common_senses = ["TL_배움과 학문"]
    dt = DataTransformer(json_directory=json_directory, common_senses=common_senses)
    json_datas, total_time = dt.load_json_files()

    # Documents: json to text
    documents = [{"text": json.dumps(item)} for item in json_datas]

    # Text splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        for split in tqdm(executor.map(text_splitter.split_text, documents), total=len(documents), desc="Splitting documents"):
            split_docs.extend(split)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for splitting: {total_time:.2f} seconds")

    # Embedding
    chunks = [Document(page_content=doc) for doc in split_docs]
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return vectordb, OPENAI_API_KEY

vectordb, OPENAI_API_KEY = initialize()

# Streamlit UI components
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

def prompt_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def update_prompt(service):
    if service == "지식검색":
        return prompt_load("prompt_common_senses.txt")
    elif service == "퀴즈":
        return prompt_load("prompt_quiz.txt")
    else:
        return '서비스가 선택되지 않았습니다.'

if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "service" not in st.session_state:
    st.session_state["service"] = "지식검색"

if "prompt" not in st.session_state:
    st.session_state["prompt"] = update_prompt(st.session_state["service"])

if "retriever" not in st.session_state:
    print("Retriever Done.")
    st.session_state.retriever = vectordb.as_retriever()

# Streamlit main interface
if __name__ == '__main__':
    st.title("챗-봇")
    with st.expander("챗봇 사용법", expanded=False):
        st.markdown("""
            - 시사 상식을 알려주는 챗봇입니다.
            - 답변 내용은 ai-hub의 지식검색 대화 데이터셋 기반으로 합니다.
        """)

    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
        st.session_state["model"] = st.radio("모델을 선택해주세요.", ["gpt-4o", "gpt-3.5-turbo"])
        st.session_state["service"] = st.radio("답변 카테고리를 선택해주세요.", ["지식 검색", "퀴즈"])

    chatbot = Chatbot(api_key=st.session_state["OPENAI_API"],
                      retriever=st.session_state.retriever,
                      sys_prompt=st.session_state["prompt"],
                      model_name=st.session_state["model"])

    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message'])

    if st.session_state["service"] == "지식검색":
        if prompt := st.chat_input("질문을 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("ai"):
                response = chatbot.generate(str(st.session_state.chat_history[-2:]) + f"\n\n{prompt}")
                for word in stream_data(response):
                    st.markdown(word)
                st.session_state.chat_history.append({"role": "user", "message": prompt})
                st.session_state.chat_history.append({"role": "ai", "message": response})
    elif st.session_state["service"] == "퀴즈":
        if prompt := st.chat_input("정답을 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("ai"):
                response = chatbot.generate(str(st.session_state.chat_history[-2:]) + f"\n\n{prompt}")
                for word in stream_data(response):
                    st.markdown(word)
                st.session_state.chat_history.append({"role": "user", "message": prompt})
                st.session_state.chat_history.append({"role": "ai", "message": response})
