import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from collections import Counter
import gradio as gr

#public url: https://3b20064f5bf66b7688.gradio.live


loader = DirectoryLoader('.',glob="*.txt",loader_cls=TextLoader)
documents = loader.load()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
#chunks
chunks=text_splitter.split_documents(documents)
#print("분할된 택스트의 개수 : ",len(texts))

#ChromaDB를 이용한 검색기 사용 
#임베딩(Embedding)
#- 텍스트 정보를 DB에 저장하기 위해 텍스트를 숫자(벡터)로 바꾸는 과정
embeddings=OpenAIEmbeddings()

#벡터 도구 객체 선언
vectordb=Chroma.from_documents(documents=chunks,embedding=embeddings)
#입력한 텍스트로부터 유사한 텍스트 찾아주는 검색기
retirever=vectordb.as_retriever()
#유사한 문서들 찾아서 반환
docs=retirever.invoke("신혼부부를 위한 정책이 있어?")

qa_chain=RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0),
    chain_type="stuff",
    retriever=retirever,
    return_source_documents=True
)



def get_chatbot_response(chatbot_response):
  print(chatbot_response['result'].strip())
  print('\n문서 출처:')
  for source in chatbot_response['source_documents']:
    print(source.metadata['source'])

input_text="신혼부부의 신혼집 마련을 위한 정책이 있을까?"
chatbot_response=qa_chain.invoke(input_text)
get_chatbot_response(chatbot_response)

print(chatbot_response)

def greet(name):
  return f"Hello {name}!"


#gradio UI 설계
import gradio as gr

#인터페이스를 생성

with gr.Blocks() as demo:

    chatbot= gr.Chatbot(label="청년정책챗봇") #                                                                                                                                                               청년정책챗봇 레이블을 좌측 상단에 구성
    msg = gr.Textbox(label="질문해주세요!") #                                                                                                                                          하단 채팅창의 레이블

    clear= gr.Button("대화 초기화") #                                                                                                                                                                                                               대화 초기화 버튼

    #챗봇의 답변을 처리하는 함수

    def respond (message, chat_history):
        result = qa_chain.invoke(message)
        bot_message = result['result']
        bot_message += '#sources:'
        #  답변의 출처를 표기
        for i, doc in enumerate (result['source_documents']):
            bot_message += '[' + str(i+1) + ']' + doc.metadata['source'] + ''  #채팅 기록에 사용자의 메시지와 봇의 응답을 추가
    
        chat_history.append((message, bot_message))
        return "", chat_history

    # 사용자의 입력을 제출(submit)하면 respond 함수가 호출
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # '초기화' 버튼을 클릭하면 채팅 기록을 초기화
    clear.click(lambda: None, None, chatbot, queue=False)

#인터페이스 실행
demo.launch(debug=True,share=True)
