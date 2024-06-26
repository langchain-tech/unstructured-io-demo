import os
import uuid
import base64
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
POSTGRES_URL_EMBEDDINDS=os.getenv("POSTGRES_URL_EMBEDDINDS")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")



root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.join(root_path, "data/fy2024.pdf")
output_path = os.path.join(root_path, "images")

openai_ef = OpenAIEmbeddings()



text_elements = []
text_summaries = []

table_elements = []
table_summaries = []

image_elements = []
image_summaries = []


def file_reader():
    raw_pdf_elements = partition_pdf(
        filename=filename,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=2000,
        new_after_n_chars=1700,
        extract_image_block_output_dir=output_path,
    )
    return raw_pdf_elements



def text_insert(raw_pdf_elements):
    summary_prompt = """
    Summarize the following {element_type}:
    {element}
    """

    prompt=PromptTemplate.from_template(summary_prompt)
    llm=ChatOpenAI(model="gpt-4o", openai_api_key = openai_api_key, max_tokens=1024)
    runnable = prompt | llm

    for e in raw_pdf_elements:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
            summary = runnable.invoke({'element_type': 'text', 'element': e})
            text_summaries.append(summary.content)

        elif 'Table' in repr(e):
            table_elements.append(e.text)
            summary = runnable.invoke({'element_type': 'table', 'element': e})
            table_summaries.append(summary.content)


def image_insert():

    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def summarize_image(encoded_image):
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images."),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Describe the contents of this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ])
        ]
        response = ChatOpenAI(model="gpt-4-vision-preview", openai_api_key=openai_api_key, max_tokens=1024).invoke(prompt)
        return response.content
    

    for i in os.listdir(output_path):
        if i.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image)
            image_summaries.append(summary)


documents = []
retrieve_contents = []

def get_docummets():
    for e, s in zip(text_elements, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'text',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)
    print("text_element done")

    for e, s in zip(table_elements, table_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'table',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)
    
    print("table_elements done")

    for e, s in zip(image_elements, image_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'image',
                'original_content': e
            }
        )
        retrieve_contents.append((i, s))
        documents.append(doc)

    print("image_elements Done")

def add_docs_to_postgres(collection_name):
    vectorstore = PGVector(embeddings=openai_ef,collection_name=collection_name,connection=POSTGRES_URL_EMBEDDINDS,use_jsonb=True,)
    vectorstore.add_documents(documents)



def add_docs_to_pinecone(index_name):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # we create a new index
    pc.create_index(
            index_name,
            dimension=1536,
            metric='dotproduct',
            spec=spec
        )
    import pdb
    pdb.set_trace()
    n=len(documents)//2
    doc1=documents[:n]
    doc2=documents[n:]

    vectorstore_from_docs = PineconeVectorStore.from_documents(
        doc1,
        index_name=index_name,
        embedding=openai_ef
    )




def main():
    collection_name="fy2024"
    print("started file reader")
    raw_pdf_elements=file_reader()
    print(raw_pdf_elements)
    print()

    text_insert(raw_pdf_elements)
    print("text_insert Done")
    image_insert()
    print("image_insert Done")
    print()
    get_docummets()
    print("get_docummets Done")
    #add_docs_to_postgres(collection_name)
    add_docs_to_pinecone(collection_name)
    print("Done")

if __name__=="__main__":
    main()