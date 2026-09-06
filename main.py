# ===========================================================================================
# 패키지&모듈인지 class인지 비교하는 방법 (파이썬 개발자들은 아래처럼 약속된대로 이름을 짓는다)
# 1. 소문자 + 언더바 -> 패키지, 모듈, 메소드/변수
# ex) langchain_community(패키지), document_loaders(모듈), loader.load() (메소드)
# 
# 2. 단어 첫 글자마다 대문자 -> class
# ex) ChatOpenAI, RecursiveCharacterTextSplitter
# ===========================================================================================


import streamlit as st                # 파이썬으로 간편하게 UI제작 프레임워크
from dotenv import load_dotenv        # API키 보안 정보를 환경변수로 안전하게 불러옴
from langchain_groq import ChatGroq   # Groq API를 렝체인에서 쉽게 쓰도록 연결하는 클래스

# langchain_community -> 최상위 패키지
# document_loaders -> 모듈
# PyPDFLoader -> class
from langchain_community.document_loaders import PyPDFLoader         # pyPDFLoader: PDF문서에서 글자를 추출해 렝체인이 읽을 수 있도록 문서 객체로 듦
from langchain_text_splitters import RecursiveCharacterTextSplitter  # RecursiveCharacterTextSplitter: 긴 글을 문맥이 깨지지 않게 일정 크기(청크)로 자르는 텍스트 분할기
from langchain_huggingface import HuggingFaceEmbeddings              # HuggingFaceEmbeddings: 텍스트를 AI가 이해할 수 있는 벡터로 바꿔주는 임베딩 모델
from langchain_community.vectorstores import FAISS                   # FAISS: Meta가 만든 초고속 벡터 DB로, 유저 질문과 가장 유사한 문서 조각을 검색

from langchain_core.prompts import ChatPromptTemplate                        # ChatPromptTemplate: AI에게 전달할 지시사항과 질문 양식(프롬프트)을 만드는 틀
from langchain_core.runnables import RunnableParallel, RunnablePassthrough   # RunnableParallel / RunnablePassthrough: LCEL 체인 안에서 데이터를 동시에 처리하거나(Parallel) 입력값을 그대로 통과(Passthrough)시켜 주는 파이프라인 제어 도구
from langchain_core.output_parsers import StrOutputParser                    # StrOutputParser: LLM의 복잡한 원본 결과물에서 순수한 글자(String) 답변만 뽑아내는 정제기

# ===========================================================================================
# 1. 문서 로드: langchain_community.document_loaders.PyPDFLoaderPDF 파일에서 글자 추출
# 2. 텍스트 분활: langchain_text_splitters.RecursiveCharacterTextSplitter 긴 문서를 문맥 단위(청크)로 자름
# 3. 임베딩: langchain_huggingface.HuggingFaceEmbeddings 텍스트를 AI용 벡터 데이터로 변환
# 4. 벡터 DB & 검색: langchain_community.vectorstores.FAISS 벡터 저장소 구축 및 검색기(retriever) 생성
# 5. 프롬포트 생성: langchain_core.prompts.ChatPromptTemplate LLM 지시문 양식 구성
# 6. LLM 모델 연결: langchain_groq.ChatGroq roq API 기반 답변 생성 AI 연결
# 7. 체인 파이프 라인: langchain_core.runnables.RunnableParallel, RunnablePassthrough
# 8. 출력 파싱: langchain_core.output_parsers.StrOutputParser AI 응답 객체에서 최종 텍스트만 추출
# ===========================================================================================
load_dotenv()

# 데코레이터: 새로고침할 때마다 PDF를 다시 읽는 것은 바효율적이므로, 함수의 결과물을 메모리에 한 번만 저장해두고, 다음번에는 함수를 다시 실행하지 말고 저장된 결과물을 그대로 재사용하라는 의미
@st.cache_resource                
def init_vector_db():
    
    # PyPDFLoader는 클래스이다. "rag.pdf"라는 재료를 전달하여 실제로 작동할 수 있는 'rag.pdf' 전용 로더기계 (객체)를 실물로 만든다
    loader = PyPDFLoader("rag.pdf") 
    # .load는 loader의 객체, 즉 PyPDFLoader 클래스의 메서드이다 (PDF의 텍스트를 읽어와 파이썬이 다룰 수 있는 리스트 형태로 만든다)
    docs = loader.load()

    # <문서를 청크 단위로 자름>
    # RecursiveCharacterTextSplitter클래스를 통해 text_splitter객체를 만들고, 인자값으로 chunk_size=500, chunk_overlap=50을 전달
    # overlap은 앞 청크와 뒤 청크 사이에 겹치는 부분을 만들어 문맥이 잘리는 것을 방지한다
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # text_splitter 객체를 사용하여 실제로 문서(docs)를 문맥에 맞게 자르는 동작(.split_documents(docs))을 실행
    # splits는 문맥에 맞게 잘린 텍스트가 들어간 리스트이다 (출처정보(metadata)도 포함)
    splits = text_splitter.split_documents(docs)

    # <청크를 벡터 단위로 임베딩>
    # HuggingFaceEmbeddings클래스로 embeddings객체 생성, "jhgan/ko-sroberta-multitask"는 Hugging Face Hub라는 AI모델 공유 플렛폼에 등록된 모델이다. 
    # 첫번째 실행시, HuggingFaceEmbeddings 클래스가 허깅페이스 서버로 접속하여 해당 모델파일을 자동으로 다운로드하여 ~/.cache 폴더에 캐시로 저장하기 때문에 로컬에서 저장된 캐시를 바로 불러온다
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # FAISS: Class, .from_documents(): 클래스 메소드 
    # 데이터를 받아와 알아서 임베딩하고 DB객체까지 한 번에 만들 수 있다
    # splits를 가져와 embeddings로 임베딩한 결과를 vectorstore에 저장
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore

#  문서 조각(List[Document])을 단일 텍스트로 합성해 주는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title('🔍 RAG는 무엇일까?')

vectorstore = init_vector_db()

# 사용자 질문과 의미적으로 가장 유사한 문서 조각을 벡터 DB에서 찾아오는 역할을 한다 `{"k": 3}`은 가장 유사도가 높은 문서 3개를 기져오라는 뜻
# langchain_core.vectorstores.VectorStore 클래스의 as_retriever() 메소드
# from langchain_community.vectorstores import FAISS로 FAISS 하나만 가져오면, 그 안에 as_retriever()가 이미 내장되어 있으므로 VectorStore를 따로 import할 필요가 없다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 프롬포트를 해석하고 답변을 텍스트로 생성하는 AI 모델 (temperature=0은 모델의 창의성을 끄고 가장 확률이 높은(일관된) 답변만 생성하도록 강제하여 환곽을 줄인다)
# langchain_groq.ChatGroq 클래스 (LangChain의 외부 통합 패키지인 langchain-groq에 위치
llm = ChatGroq(model_name="groq/compound-mini", temperature=0)

# LLM에게 내릴 지시문(System Message)과 입력변수({context}, {input}의 템플릿을 정의한다.
# langchain_core.prompts.chat.ChatPromptTemplate 클래스의 from_template() 클래스 메소드
# langchain_core.prompts.chat.ChatPromptTemplate (실제 위치) | langchain_core ➔ prompts 폴더 ➔ chat.py 파일 내부에 정의된 ChatPromptTemplate 클래스의 실제 소스코드 경로
# from langchain_core.prompts import ChatPromptTemplate (지름길) | 개발자가 매번 .chat까지 길게 타핑하는 게 귀찮으니까, 랭체인 제작자가 prompts 상위 폴더(__init__.py)에서 미리 이 클래스를 밖으로 빼놓은 것
prompt = ChatPromptTemplate.from_template(
    """당신은 문서 기반 질의응답을 수행하는 친절한 AI 어시스턴트이다.
아래 제공된 '검색된 문서 내용'만을 바탕으로 질문에 답변할 것.
만약 검색된 내용에 답변이 없다면, "제공된 문서에서는 해당 내용을 찾을 수 없습니다."라고 솔직하게 말할 것.
절대 지어서 답변하지 말 것.

'검색된 문서 내용'
{context}

'질문'
{input}

답변:"""
)

# <답변 생성을 위한 서브 체인>
# 파이브 연산자 (|) 를 통해 데이터를 물 흐르듯 다음 단계로 넘기는 렝체인 문법
chain_answer = (
    # RunnablePassthrough.assign(...): 입력된 딕셔너리 데이터를 그대로 통과시키되, 특정 키(context)의 값만 람다 함수로 변환하여 덮어씌웁니다.
    # 리스트 형태의 문서 조각(List[Document])을 format_docs 함수를 통해 하나의 긴 문자열로 합친다.
    # langchain_core.runnables.passthrough.RunnablePassthrough 클래스의 assign() 메소드
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    # StrOutputParser(): LLM이 반환하는 결과물은 텍스트뿐만 아니라 토큰 사용량 등의 메타데이터가 포함된 복잡한 객체(AIMessage)이다. 여기서 실제 답변 문자열(String)만 깔끔하게 뽑아내는 역할을 한다.
    # langchain_core.output_parsers.string.StrOutputParser 클래스
    | StrOutputParser()
)

# <원문 문서(context)와 답변(answer)을 동시에 딕셔너리로 리턴하는 LCEL 체인 조립>
# 여러 작업을 병렬로 동시에 실행하고, 그 결과를 딕셔너리 형태로 묶어서 반환
# langchain_core.runnables.base.RunnableParallel 클래스
rag_chain = RunnableParallel({
    "context": (lambda x: x["input"]) | retriever,
    "input": lambda x: x["input"]
}).assign(answer=chain_answer)

content = st.text_input('나무위키의 검색증강생성 문서를 바탕으로 답변합니다.')

if st.button('질문하기'):
    if content:
        with st.spinner('문서를 검색하고 답변을 생성하는 중입니다...'):
            try:
                # 조립된 전체 파이프라인(rag_chain)을 작동시킨다.
                # langchain_core.runnables.base.Runnable 인터페이스의 invoke() 메소드
                response = rag_chain.invoke({"input": content})
                
                # response["answer"]와 response["context"]를 안전하게 추출
                st.write(response["answer"])
                
                with st.expander("📚 참고한 문서 원문 보기"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**[조각 {i+1}]**\n{doc.page_content}\n---")
                        
            except Exception as e:
                st.error(f"⚠️ 오류가 발생했습니다: {e}")
    else:
        st.warning('질문을 입력해주세요!')