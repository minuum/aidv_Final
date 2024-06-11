#=============================import===========================
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
import logging


#Define Functions
#==================data loading and embedding==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#첫 로딩시에만 불러오고 cache형태로 저장하게 도와주는 어노테이션
@st.cache_resource(show_spinner=False)
def load_and_process_data():
    json_directory = "./dataset/common_senses"
    common_senses = ["TL_배움과 학문"]
    dt = DataTransformer(json_directory=json_directory, common_senses=common_senses)
    json_datas, total_time = dt.load_json_files()

    documents = [{"text": json.dumps(item)} for item in json_datas]

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

    chunks = [Document(page_content=doc) for doc in split_docs]
    print("Chunks split Done.")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectordb

vectordb = load_and_process_data()
retriever = vectordb.as_retriever()

##느리게 읽어오는 함수
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


#==================== Initializing=======================
# 정답 딕셔너리 선언
answer_dict={}


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
    st.session_state["retriever"] = retriever
    print("Retriever Done.")

if "prompt" not in st.session_state:
    st.session_state["prompt"] = update_prompt(st.session_state["service"])

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""




# Streamlit UI
if __name__ == '__main__':
    #sidebar
    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
        st.session_state["model"] = st.radio("모델을 선택해주세요.", ["gpt-4o", "gpt-3.5-turbo"])
        st.session_state["service"] = st.radio("답변 카테고리를 선택해주세요.", ["지식검색", "퀴즈"])
        st.session_state["prompt"] = update_prompt(st.session_state["service"])
        logging.warning(st.session_state.quiz_stage)
        st.write()
        with st.expander("데이터셋", expanded=True):
            st.markdown('''
                    - 답변 내용은 ai-hub의 지식검색 대화 데이터셋 기반으로 합니다.
                   [![aihub dataset](https://www.aihub.or.kr/web-nas/aihub21/files/public/DATA_SET202304140227079050)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71304)
                    ''')
        if st.session_state["service"] == "지식검색":
            with st.expander("출력에 관하여", expanded=True):
                st.markdown('''
                            원하는 주제에 맞는 답변,
                            관련된 용어 정리,
                            사전과 연결된 링크까지 나온답니다!
                            ''')
        if st.session_state["service"] == "퀴즈":
            with st.expander("입력 예시", expanded=True):
                st.markdown('''
                            #### 문제 입력
                            - 주제만 입력해주세요
                            - 예) 조지아, 그리스로마신화, 아르키메데스, 기묘한이야기....
                            #### 정답 입력
                            - 예) 1.a / 2.b / 3.b / 4.c / 5.c
                            ''')
            st.session_state.correct_answers=0
            st.sidebar.write(f"맞춘 정답 개수: {st.session_state.correct_answers}개")
        
        if st.button("초기화"):
            st.session_state.chat_history = []
            st.session_state["service"] = "수업"
            st.session_state.quiz_stage = 0
            st.session_state.correct_answers = 0
            st.session_state.current_answer = ""
            st.session_state.current_question = ""
            st.rerun()

    #GPT 답변 인스턴스 
    chatbot = Chatbot(api_key=st.session_state["OPENAI_API"],
                       retriever=retriever,
                       sys_prompt=st.session_state["prompt"],
                       model_name=st.session_state["model"])

    #Title
    if st.session_state["service"] == "지식검색":
        st.title("지식검색 챗봇 📚")
        
    if st.session_state["service"] == "퀴즈":
        st.title("🧐 지식,상식 퀴즈 챗봇 🧐")
    #abstract
    with st.expander("개요", expanded=True):
        if st.session_state["service"] == "지식검색":
            st.markdown("""
                    #### 시사 상식을 알려주는 챗봇입니다.
                    - 답변 내용은 ai-hub의 지식검색 대화 데이터셋 기반으로 합니다.
                    - 사용자의 답변 뿐만 아니라 유사한 주제나 단어, 중요한 단어들에 대한 링크까지 존재합니다.
                    """)
        if st.session_state["service"] == "퀴즈":
            st.markdown("""
                    #### 시사 상식을 기반으로 사용자의 답변에 맞는 퀴즈를 제공해주는 챗봇입니다.
                    - 답변 내용은 ai-hub의 지식검색 대화 데이터셋 기반으로 합니다.
                    - 첫번째 입력은 문제의 주제에 대해서, 두번째 입력부터는 문제의 정답을 맞추게 됩니다.
                    - 총 5문제, 객관식으로 출제됩니다!
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
        if st.session_state.quiz_stage % 2 == 0:
            placeholder_text = "*문제를 먼저 입력하세요.*"
        else:
            placeholder_text = "*정답을 입력하세요.*"

        if prompt := st.chat_input(placeholder=placeholder_text,):
            if st.session_state.quiz_stage % 2 == 0:
                with st.chat_message("ai"):
                    question_response = chatbot.generate(f"주제: {prompt}\n문제를 만들어 주세요.")
                    parts = question_response.split('=====')
                    if len(parts) >= 2:
                        question = parts[0]
                        correct_answer = parts[1].strip(' \n').strip('\n').replace("(","").replace(")","").split(' |')
                        correct_answer.remove('')

                        answer_dict = {}
                        for item in correct_answer:
                            key, value = item.split('. ')
                            answer_dict[int(key)] = value

                        st.session_state.current_question = question
                        st.session_state.current_answer = answer_dict
                        st.markdown(question)
                        st.session_state.chat_history.append({"role": "user", "message": prompt})
                        st.session_state.chat_history.append({"role": "ai", "message": question})
                        st.session_state.quiz_stage += 1
                        
                        logging.warning(st.session_state.quiz_stage)
                        logging.warning(st.session_state.current_question)
                        logging.warning(st.session_state.current_answer)
                        st.rerun()
            else:
                user_answer = prompt
                correct_answer = st.session_state.current_answer
                if user_answer:
                    us_list=user_answer.split(' /')
                    us_dict={}
                    for item in us_list:
                        key, value = item.split('.')
                        # 딕셔너리에 새로운 항목을 추가합니다.
                        us_dict[int(key)] = value
                    logging.warning("내가 낸 답변 :" +str(us_dict))
                logging.warning("정답 :" +str(correct_answer))
                
                
                correct_pass = True 
                st.session_state.correct_answers =5
                wrong_answers=[]
                for key in correct_answer.keys():
                # 두 딕셔너리의 특정 키에 대한 값이 같은지 확인합니다.
                    if key in us_dict and correct_answer[key] == us_dict[key]:
                        logging.warning(str(key)+"번 정답!")
                    else:
                        logging.warning(str(key)+"번 오답!")
                        wrong_answers.append(key)
                        correct_pass = False
                        st.session_state.correct_answers -=1

                if correct_pass:
                    with st.chat_message("ai"):
                        st.markdown("정답입니다!")
                    st.session_state.quiz_stage += 1
                    st.session_state.chat_history.append({"role": "user", "message": user_answer})
                    st.session_state.chat_history.append({"role": "ai", "message": "정답입니다!"})
                    st.rerun()
                else:
                    with st.chat_message("ai"):
                        st.markdown(f"""틀렸습니다. 다시 시도해보세요. {st.session_state.correct_answers}개 맞았습니다.
                                      틀린 문제는 {str(wrong_answer+"번" for wrong_answer in wrong_answers)}""")
                        
                    st.session_state.chat_history.append({"role": "user", "message": user_answer})
                    st.session_state.chat_history.append({"role": "ai", "message": "틀렸습니다. 다시 시도해보세요."})

   
