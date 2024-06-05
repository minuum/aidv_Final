import streamlit as st
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
import time
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

def pdf_load(dir):
    input_docs = []
    # Load all PDF files using PyPDFLoader
    input_pdf_files = glob(os.path.join(dir, '*.pdf'))
    for pdf_file in input_pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        input_docs.extend(pdf_documents)
        
    return input_docs
        

documents = []
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pdf_directory = './data'

if "OPENAI_API" not in st.session_state:
    st.session_state["OPENAI_API"] = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
# 기본 모델을 설정합니다.
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"
# 채팅 기록을 초기화합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "sevice" not in st.session_state:
    st.session_state["service"] = "수업"

if "previous" not in st.session_state:
    st.session_state["previous"] = pdf_load('./previous')

if "current" not in st.session_state:
    st.session_state["current"] = pdf_load('./current')

#################################################
if "prompt" not in st.session_state:
    st.session_state["prompt"] = ''' 
    수업 프롬프트는 여기에 입력
    
    ''' if  st.session_state["service"] == "수업" else '''
    
    졸업 프롬프트는 여기에 입력
    '''
#################################################

if "retriever" not in st.session_state:
    pdf_files = glob(os.path.join(pdf_directory, '*.pdf'))

    # Load all PDF files using PyPDFLoader
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)
        
    # 텍스트는 RecursiveCharacterTextSplitter를 사용하여 분할
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunk_splitter.split_documents(documents)
    print("Chunks split Done.")
    # embeddings은 OpenAI의 임베딩을 사용
    # vectordb는 chromadb사용함

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("Retriever Done.")
    st.session_state.retriever = vectordb.as_retriever()
    
# pdf를 사용해서 pdf(논문)을 모두 로드

if __name__ == '__main__':
    
    st.title("챗-봇")
    # Create a sidebar for API key and model selection
    with st.expander("챗봇 사용법", expanded=False):
        st.markdown("""
                    - 강남대학교 학사지원을 위한 챗봇입니다.
                    - 답변 내용은 학사지원 메뉴얼을 기반으로 합니다.
                    """)
    ################# 설정을 위한 사이드바를 생성합니다. 여기서 api키를 받아야 실행됩니다. ##########################################
    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
        #모델을 선택합니다.
        st.session_state["model"] = st.radio("모델을 선택해주세요.", ["gpt-4o", "gpt-3.5-turbo"])
        #라디오 버튼을 사용하여 서비스를 선택합니다.
        st.session_state["service"] = st.radio("학사지원 서비스를 선택해주세요.", ["수업", "졸업"])
    # Chatbot을 생성합니다.
    chatbot = Chatbot(api_key=st.session_state["OPENAI_API"],
                       retriever=st.session_state.retriever,
                       sys_prompt=st.session_state["prompt"],
                       model_name=st.session_state["model"])



    ############################################ 실제 챗봇을 사용하기 위한 Streamlit 코드 ###################################################
    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message'])    
    ### 사용자의 입력을 출력하고 생성된 답변을 출력합니다.
    if prompt := st.chat_input("질문을 입력하세요."):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "message": prompt})

        with st.chat_message("ai"):                
            response = chatbot.generate(prompt)
            st.write_stream(stream_data(response))
            st.session_state.chat_history.append({"role": "ai", "message": response})