# models/neural_network.py

from opensearchpy import OpenSearch
import time
import jwt
import requests
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import StuffDocumentsChain

import os
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# Доступ к переменным
HOSTS = (os.getenv('HOSTS'))[2:-2]
PASS = os.getenv('PASS')
CA = os.getenv('CA')
SERVICE_ACCOUNT_ID = os.getenv('SERVICE_ACCOUNT_ID')
PRIVATE_KEY = os.getenv("PRIVATE")
KEY_ID = os.getenv('KEY_ID')
SOURCE_DIR = os.getenv('SOURCE_DIR')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500)) 
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100)) 

# Подключение к OpenSearch
conn = OpenSearch(
    HOSTS,
    http_auth=('admin', PASS),
    use_ssl=True,
    verify_certs=True,
    ca_certs=CA)

# Получение IAM-токена
def get_iam_token():
    now = int(time.time())
    payload = {
        'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
        'iss': SERVICE_ACCOUNT_ID,
        'iat': now,
        'exp': now + 360}
    encoded_token = jwt.encode(
        payload,
        PRIVATE_KEY,
        algorithm='PS256',
        headers={'kid': KEY_ID})
    url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
    x = requests.post(url,
                      headers={'Content-Type': 'application/json'},
                      json={'jwt': encoded_token}).json()
    return x['iamToken']

# Инициализация
token = get_iam_token()
loader = DirectoryLoader(
    SOURCE_DIR,
    glob="*.txt",
    loader_cls=lambda file_path: TextLoader(file_path, encoding="utf-8"),
    silent_errors=True,
    show_progress=True,
    recursive=True
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
docs = text_splitter.split_documents(documents)
embeddings = YandexGPTEmbeddings(iam_token=token, model_uri="emb://b1gnk0qlljh1lhjqekj4/text-search-doc/latest", sleep_interval=0.1, folder_id="b1gnk0qlljh1lhjqekj4")
docsearch = OpenSearchVectorSearch.from_documents(
    docs,
    embeddings,
    opensearch_url='https://rc1a-36otpjvamcgfg3co.mdb.yandexcloud.net:9200',
    http_auth=("admin", PASS),
    use_ssl=True,
    verify_certs=True,
    ca_certs=CA,
    engine='lucene'
)

# Создаём цепочку
llm = YandexGPT(iam_token=token,
                model_uri="gpt://b1gnk0qlljh1lhjqekj4/yandexgpt/latest")

# Промпт для обработки документов
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)

# Промпт для языковой модели
document_variable_name = "context"
stuff_prompt_override = """
    Представь себе, что ты сотрудник Сибирского Федерального университета. Ты отвечаешь от лица мужчины. Твоя задача вежливо отвечать только на вопросы
    Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
    Текст:
    -----
    {context}
    -----
    Вопрос:
    {query}
"""
prompt = PromptTemplate(
    template=stuff_prompt_override,
    input_variables=["context", "query"]
)

# Создаём цепочку
llm_chain = langchain.chains.LLMChain(llm=llm,
                                      prompt=prompt)

chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

async def query_model(query: str):
    docs = docsearch.similarity_search(query, k=2)
    res = chain.invoke({'query': query,
                        'input_documents': docs})
    return res["output_text"]
