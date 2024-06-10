__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
import time
import json
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chromadb import chromadb
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import sys
sys.path.append("")
from function import DataTransformer
from chatbot_class import Chatbot

#==================data loading and embedding==================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
documents = []
# Data loading and chopping
json_directory = "./dataset/common_senses"
common_senses = ["TL_배움과 학문"]
dt = DataTransformer(json_directory=json_directory, common_senses=common_senses)
json_datas, total_time = dt.load_json_files()

# documents : json to text
documents = [{"text": json.dumps(item)} for item in json_datas]

# text_splits
def split_document(doc):
    return text_splitter.split_text(doc["text"])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = []
start_time = time.time()
with ThreadPoolExecutor() as executor:
    for split in tqdm(executor.map(split_document, documents), total=len(documents), desc="Splitting documents"):
        split_docs.extend(split)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken for splitting: {total_time:.2f} seconds")
 
# embedding
# Document 객체로 변환
chunks = [Document(page_content=doc) for doc in split_docs]
print("Chunks split Done.")
# embeddings은 OpenAI의 임베딩을 사용
# vectordb는 chromadb사용함
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever =vectordb.as_retriever()

#=============== 관련 함수들 ====================
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

#============ 프롬프트 업데이트 ===================
def prompt_load(file_path):
    file_content = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content
        
def update_prompt(service):
    if service == "지식검색":
        file_path = "prompt_common_senses.txt"
        return prompt_load(file_path)
    elif service == "퀴즈":
        file_path = "prompt_quiz.txt"
        return prompt_load(file_path)


#=============변수 초기화====================
if "OPENAI_API" not in st.session_state:
    st.session_state["OPENAI_API"] = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
# 기본 모델을 설정합니다.
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"
# 채팅 기록을 초기화합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "service" not in st.session_state:
    st.session_state["service"] = "지식검색"

if "quiz_stage" not in st.session_state:
    st.session_state.quiz_stage = 0

if "correct_answers" not in st.session_state:
    st.session_state.correct_answers = 0

if "retriever" not in st.session_state:    
    st.session_state["retriever"]=retriever
    print("Retriever Done.")
    
if "prompt" not in st.session_state:
    if st.session_state["service"] == "지식검색":
        file_path = "prompt_common_senses.txt"
        st.session_state["prompt"] = prompt_load(file_path)
    elif st.session_state["service"] == "퀴즈":
        file_path = "prompt_quiz.txt"
        st.session_state["prompt"] = prompt_load(file_path)
    else:
        st.session_state["prompt"] = '''
        서비스가 선택되지 않았습니다.
    '''    


# pdf를 사용해서 pdf(논문)을 모두 로드

if __name__ == '__main__':
    
    if st.session_state["service"] == "지식검색":
        st.title("지식검색 챗봇")       
    if st.session_state["service"] == "퀴즈":
        st.title("지식,상식 퀴즈 챗봇")
    # Create a sidebar for API key and model selection
    with st.expander("챗봇 사용법", expanded=False):
        if st.session_state["service"] == "지식검색":
            st.markdown("""
                    - 시사 상식을 알려주는 챗봇입니다.
                    - 답변 내용은 ai-hub의 지식검색 대화 데이터셋 기반으로 합니다.
                    - 사용자의 답변 뿐만 아니라 유사한 주제나 단어, 중요한 단어들에 대한 링크까지 존재합니다.
                    """)
        if st.session_state["service"] == "퀴즈":
            st.markdown("""
                    - 시사 상식을 기반으로 사용자의 답변에 맞는 퀴즈를 제공해주는 챗봇입니다.
                    - 답변 내용은 ai-hub의 지식검색 대화 데이터셋 기반으로 합니다.
                    - 첫번째 입력은 문제의 주제에 대해서, 두번째 입력부터는 문제의 정답을 맞추게 됩니다.
                    """)
    
    ################# 설정을 위한 사이드바를 생성합니다. 여기서 api키를 받아야 실행됩니다. ##########################################
    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
        # 모델을 선택합니다.
        st.session_state["model"] = st.radio("모델을 선택해주세요.", ["gpt-4o", "gpt-3.5-turbo"])
        # 라디오 버튼을 사용하여 서비스를 선택합니다.
        st.session_state["service"] = st.radio("답변 카테고리를 선택해주세요.", ["지식검색", "퀴즈"])
    # Chatbot을 생성합니다.
    chatbot = Chatbot(api_key=st.session_state["OPENAI_API"],
                       retriever=retriever,
                       sys_prompt=st.session_state["prompt"],
                       model_name=st.session_state["model"])

    ############################################ 실제 챗봇을 사용하기 위한 Streamlit 코드 ###################################################
    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message']) 

    if st.session_state["service"] == "지식검색":
        if prompt := st.chat_input("질문을 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("ai"):
                response = chatbot.generate(str(st.session_state.chat_history[-2:]) + f"\n\n{prompt}")
                st.write_stream(stream_data(response))
            st.session_state.chat_history.append({"role": "user", "message": prompt})
            st.session_state.chat_history.append({"role": "ai", "message": response})  
    
    if st.session_state["service"] == "퀴즈":
        if prompt := st.chat_input("문제를 먼저 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.quiz_stage % 2 == 0:  # 문제 주제를 입력하는 단계
                with st.chat_message("ai"):
                    question = chatbot.generate(f"주제: {prompt}\n문제를 만들어 주세요.")
                    for word in stream_data(question):
                        st.markdown(word)
                    st.session_state.chat_history.append({"role": "user", "message": prompt})
                    st.session_state.chat_history.append({"role": "ai", "message": question})
                    st.session_state.quiz_stage += 1
                    st.session_state.current_question = question
            else:  # 정답을 입력하는 단계
                with st.chat_message("ai"):
                    if prompt.lower() in st.session_state.current_question.lower():
                        st.session_state.correct_answers += 1
                        response = f"정답입니다! 현재까지 맞춘 정답 개수: {st.session_state.correct_answers}"
                    else:
                        response = f"틀렸습니다. 현재까지 맞춘 정답 개수: {st.session_state.correct_answers}"
                    for word in stream_data(response):
                        st.markdown(word)
                    st.session_state.chat_history.append({"role": "user", "message": prompt})
                    st.session_state.chat_history.append({"role": "ai", "message": response})
                    st.session_state.quiz_stage += 1
