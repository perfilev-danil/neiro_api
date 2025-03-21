from opensearchpy import OpenSearch
import time
import jwt
import requests
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import JSONLoader, TextLoader, DirectoryLoader
from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import StuffDocumentsChain

import os
import json
from dotenv import load_dotenv

import magic

import asyncio

# Загрузка переменных из .env
load_dotenv()

# Доступ к переменным
PRIVATE_KEY = os.getenv("KEY")
HOSTS = (os.getenv('HOSTS'))
PASS = os.getenv('PASS')
CA = os.getenv('CA')
SERVICE_ACCOUNT_ID = os.getenv('SERVICE_ACCOUNT_ID')
KEY_ID = os.getenv('KEY_ID')
SOURCE_DIR = os.getenv('SOURCE_DIR')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1500)) 
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 300)) 

# Подключение к OpenSearch
conn = OpenSearch(
    HOSTS,
    http_auth=('admin', PASS),
    use_ssl=True,
    verify_certs=True,
    ca_certs=CA
)

# Считываем документы и разбиваем на фрагменты
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

# Промпт для языковой модели
document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
)
document_variable_name = "context"
stuff_prompt_override = """
    Если спросили кто ты, ответь, что ты сотрудник Сибирского Федерального Университета. Ты отвечаешь от лица мужчины. 
    Ты работаешь на сайте Отчёты губернаторов. Если спросили про сайт, ответь, что это сайт, на котором размещена информация о губернаторах Енисейской губернии и их отчётах.
    Твоя задача вежливо отвечать только на вопросы.
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

# Получение IAM-токена
def get_iam_token():
    now = int(time.time())

    payload = {
        'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
        'iss': SERVICE_ACCOUNT_ID,
        'iat': now,
        'exp': now + 3600
    }
    encoded_token = jwt.encode(
        payload,
        PRIVATE_KEY,
        algorithm='PS256',
        headers={'kid': KEY_ID}
    )
    url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
    response = requests.post(url,
                             headers={'Content-Type': 'application/json'},
                             json={'jwt': encoded_token}).json()
    
    return response['iamToken']

def initialize_models(token):
    embeddings = YandexGPTEmbeddings(iam_token=token, model_uri="emb://b1gnk0qlljh1lhjqekj4/text-search-doc/latest", sleep_interval=0.1, folder_id="b1gnk0qlljh1lhjqekj4")
    docsearch = OpenSearchVectorSearch.from_documents(
        docs,
        embeddings,
        opensearch_url='https://rc1a-36otpjvamcgfg3co.mdb.yandexcloud.net:9200',
        http_auth=("admin", PASS),
        use_ssl=True,
        verify_certs=True,
        ca_certs=CA,
        engine='lucene',
        bulk_size=10000 
    )
    # Создаём цепочку
    llm = YandexGPT(iam_token=token, model_uri="gpt://b1gnk0qlljh1lhjqekj4/yandexgpt/latest")
    llm_chain = LLMChain(llm=llm,
                         prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    return docsearch, llm_chain, chain

def replace_e(text: str) -> str:
    return text.replace('ё', 'е').replace('Ё', 'Е')

global_token = get_iam_token()
global_docsearch, global_llm_chain, global_chain = initialize_models(global_token)

async def update_token_and_models():
    global global_token, global_docsearch, global_llm_chain, global_chain
    while True:
        try:
            global_token = get_iam_token()
            global_docsearch, global_llm_chain, global_chain = initialize_models(global_token)
        except Exception as e:
            print(f"Ошибка при обновлении токена и моделей: {e}")
            print("Перезапуск контейнера...")
            os._exit(1)
        await asyncio.sleep(21600) 

async def query_model(query: str):
    global global_token, global_docsearch, global_llm_chain, global_chain
    try:
        query = replace_e(query)
        try:
            docs = global_docsearch.similarity_search(query, k=7)
            res = global_chain.invoke({'query': query, 'input_documents': docs})
            return res["output_text"]
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}")
            print("Повторное выполнение запроса...")
            try:
                global_token = get_iam_token()
                global_docsearch, global_llm_chain, global_chain = initialize_models(global_token)

                docs = global_docsearch.similarity_search(query, k=7)
                res = global_chain.invoke({'query': query, 'input_documents': docs})
                return res["output_text"]
            except Exception:
                print("Ошибка при повторном выполнении запроса!")
                raise
    except Exception:
        print("Критическая ошибка, перезапускаю контейнер...")
        os._exit(1)

        