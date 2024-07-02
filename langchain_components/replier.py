import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import  ConfigurableFieldSpec
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.vectorstores import Pinecone

import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] =os.getenv("OPENAI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

def create_postgres_chat_message_history(session_id, user_id):
    return PostgresChatMessageHistory(connection_string=POSTGRES_URL,session_id=session_id)

def prepare_prompt_and_chain_with_history():
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert in data that hepls to summerize the text and answer the question based only on the following content, which may include text and tables: if the content is not relevent then return I don't know. "
                ** IMPORTANT INSTRUCTIONS -->
                    1. Reply in 50 words maximum.""",
            ),
            "Here is the context to answer user's questions everytime --> {data}. ",
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
        ]
    )
    runnable = prompt | llm
    with_message_history = RunnableWithMessageHistory(
            runnable,
            create_postgres_chat_message_history,
            input_messages_key="input",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
            verbose=True,
        )
    return with_message_history




def get_vectorstore_from_postgres(collection_name):
    openai_ef = OpenAIEmbeddings()
    vectorstore = PGVector(
        embeddings=openai_ef,
        collection_name=collection_name,
        connection=POSTGRES_URL,
        use_jsonb=True,
    ) 
    return vectorstore      


def get_vectorstore_from_pinecone(index_name):
    openai_ef = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name, openai_ef)
    return vectorstore


def get_context_from_vectorstore(vectorstore,user_query):
    logging.info("Start postgres vector search......")
    relevant_docs = vectorstore.similarity_search(user_query,k=4)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    return context