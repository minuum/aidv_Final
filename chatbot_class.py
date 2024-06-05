from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def _format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
class Chatbot:
    def __init__(self,
                 api_key,
                 retriever,
                 sys_prompt,
                 model_name
                 ):
        self.model = ChatOpenAI(api_key=api_key, model=model_name)
        # 프롬프트 템플릿을 정의합니다.
        # 앞, 중간, 뒤로 나누어서 프롬프트를 정의합니다.
        # 중간은 사용자의 입력을 그대로 전달합니다.
        front_prompt = '''
            {context}를 반드시 충분히 이해하고 여기에서 설명해야해. 잘하면 10달러를 줄게.
        '''
        end_prompt = ''' Question: {question} '''
        
        SYS_PROMPT = front_prompt + sys_prompt + end_prompt
        self.prompt = ChatPromptTemplate.from_template(SYS_PROMPT)
        
        self.chain = (
            {'context': retriever | _format_docs, 'question': RunnablePassthrough()}  # 'context'는 retriever와 format_docs를 통해 설정되고, 'question'은 그대로 전달됩니다.
            | self.prompt  # 프롬프트 템플릿을 적용합니다.
            | self.model  # 모델을 호출합니다.
            | StrOutputParser()  # 출력 파서를 통해 모델의 출력을 문자열로 변환합니다.
            ) 

    def generate(self, input_message):
        return self.chain.invoke(input_message)