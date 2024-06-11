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

@st.cache_resource(show_spinner=False)
def load_and_process_data():
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
    return vectordb

vectordb = load_and_process_data()
retriever = vectordb.as_retriever()

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

def pdf_load(dir):
    input_docs = []
    input_pdf_files = glob(os.path.join(dir, '*.pdf'))
    for pdf_file in input_pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        input_docs.extend(pdf_documents)

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

if "OPENAI_API" not in st.session_state:
    st.session_state["OPENAI_API"] = OPENAI_API_KEY

if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"

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

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""

if __name__ == '__main__':
    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
        st.session_state["model"] = st.radio("모델을 선택해주세요.", ["gpt-4o", "gpt-3.5-turbo"])
        st.session_state["service"] = st.radio("답변 카테고리를 선택해주세요.", ["지식검색", "퀴즈"])
        st.session_state["prompt"] = update_prompt(st.session_state["service"])
        st.write()
        if st.session_state["service"] =="퀴즈":
            st.markdown("### 입력 예시")
            with st.expander("지식 검색", expanded=False):
                st.markdown('''
                            #### 문제 입력
                            - 주제만 입력해주세요
                            - 예) 조지아, 그리스로마신화, 아르키메데스, 기묘한이야기....
                            #### 정답 입력
                            - 예) 1.a / 2.b / 3.b / 4.c / 5.c
                            ''')
    chatbot = Chatbot(api_key=st.session_state["OPENAI_API"],
                       retriever=retriever,
                       sys_prompt=st.session_state["prompt"],
                       model_name=st.session_state["model"])

    if st.session_state["service"] == "지식검색":
        st.title("지식검색 챗봇 📚")
        
    if st.session_state["service"] == "퀴즈":
        st.title("🧐 지식,상식 퀴즈 챗봇 🧐")

    with st.expander("사용법", expanded=True):
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
                    - 입력 예시 : 그리스로마신화,인공지능기술,삼국시대,조지아...etc
                    """)

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

            if st.session_state.quiz_stage % 2 == 0:
                with st.chat_message("ai"):
                    question = chatbot.generate(f"주제: {prompt}\n문제를 만들어 주세요.")
                    answers=question.split('=====')
                    st.write_stream(stream_data(question))
                    st.session_state.chat_history.append({"role": "user", "message": prompt})
                    st.session_state.chat_history.append({"role": "ai", "message": question})
                    st.session_state.quiz_stage += 1
                    st.session_state.current_question = question
                    st.session_state.current_answer = answers
                    print(answers)
            else:
                answer_prompt=st.chat_input("정답 입력 : ")
                if "current_answer" in st.session_state:
                    correct_answer = st.session_state.current_answer
                else:
                    correct_answer = ""  # 여기서 correct_answer를 설정해야 합니다.
                while True:
                    if prompt.lower() == correct_answer.lower():
                        with st.chat_message("ai"):
                            st.markdown("정답입니다!")
                        st.session_state.correct_answers += 1
                        st.session_state.quiz_stage += 1
                        st.session_state.chat_history.append({"role": "user", "message": prompt})
                        st.session_state.chat_history.append({"role": "ai", "message": "정답입니다!"})
                    else:
                        with st.chat_message("ai"):
                            st.markdown("틀렸습니다. 다시 시도해보세요.")
                        st.session_state.chat_history.append({"role": "user", "message": prompt})
                        st.session_state.chat_history.append({"role": "ai", "message": "틀렸습니다. 다시 시도해보세요."})

    st.sidebar.write(f"맞춘 정답 개수: {st.session_state.correct_answers}개")
