import os
import re
import uuid
import base64
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import pdb


from dotenv import load_dotenv
load_dotenv()



openai_api_key = os.getenv("OPENAI_API_KEY")
POSTGRES_URL_EMBEDDINDS=os.getenv("POSTGRES_URL_EMBEDDINDS")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")


parent_path =os.getcwd()
filename=os.path.join(parent_path,"data/fy2024.pdf")
output_path=os.path.join(parent_path,"data_ingest/images")


openai_ef = OpenAIEmbeddings()




text_elements = []
text_summaries = []

table_elements = []
table_summaries = []

image_elements = []
image_and_text_summaries=[]


def file_reader():
    raw_pdf_elements = partition_pdf(
        filename=filename,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=2000,
        new_after_n_chars=1800,
        extract_image_block_output_dir=output_path,
    )
    return raw_pdf_elements



def summary_of_text_by_gpt(element_type, input):
    summary_prompt = """
    Summarize the following {element_type}:
    {element}
    """

    prompt=PromptTemplate.from_template(summary_prompt)
    llm=ChatOpenAI(model="gpt-4o", openai_api_key = openai_api_key, max_tokens=1024)
    runnable = prompt | llm

    summary = runnable.invoke({'element_type': element_type, 'element': input})
    return summary.content




def text_insert(raw_pdf_elements):
    i=0
    for e in raw_pdf_elements:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
            summary = summary_of_text_by_gpt("text",e)
            text_summaries.append(summary)

        elif 'Table' in repr(e):
            table_elements.append(e.text)
            summary =summary_of_text_by_gpt("table",e)
            table_summaries.append(summary)
        print(i)
        i+=1



def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    

def summarize_image(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "Describe the only text written on this image. if there is nothing written on the image then say 'no data found'"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, max_tokens=1024).invoke(prompt)
    return response.content




def get_text_by_page_number(raw_pdf_elements,page_number):
    e = raw_pdf_elements[page_number]
    if 'CompositeElement' in repr(e):
        text_elements.append(e.text)
        summary = summary_of_text_by_gpt("text", e.text)
        return summary
    elif 'Table' in repr(e):
        table_elements.append(e.text)
        summary = summary_of_text_by_gpt("table", e.text)
        return summary
    

def get_last_index_of_page(raw_pdf_elements):
    page_numbers=[]
    last_indices={}
    for i in range(len(raw_pdf_elements)):
        page_number=raw_pdf_elements[i].metadata.page_number
        if page_number in page_numbers:
            last_indices[page_number]=i 
        else:
            page_numbers.append(page_number)
            last_indices[page_number] = i
    return last_indices



def match_text_for_no_data_found(text):
    pattern = r"no data found"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return True
    return False


def image_insert_with_text(raw_pdf_elements,last_indices):
    for i in os.listdir(output_path):
        if i.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, i)
            print(i)
            img_page=i.split("-")[1]
            page_num = int(img_page)
            ind=0
            t=True
            while len(last_indices)>ind and t:
                if page_num in last_indices and page_num>=0:
                    text_page=last_indices[page_num]
                    t=False
                elif page_num>0:
                    page_num=page_num-1
                else:
                    print("not found text for ",img_page)
                    t=False
                    text_page=0
                ind+=1
            if text_page==0:
                continue
            encoded_image = encode_image(image_path)
            img_summary = summarize_image(encoded_image)
            print("img summary:- ",img_summary)
            if not match_text_for_no_data_found(img_summary):
                image_elements.append(encoded_image)
                text_summery=get_text_by_page_number(raw_pdf_elements,text_page)
                print("text sumary:- ",text_summery)
                image_and_text_summaries.append(img_summary+" "+ text_summery)





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

    for e, s in zip(image_elements, image_and_text_summaries):
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

def main():
    collection_name="newfy2024"
    print("started file reader")
    raw_pdf_elements=file_reader()
    print("file reader Done...")
    print()
    print("text_insert started...")
    text_insert(raw_pdf_elements)
    print("text_insert Done")
    pdb.set_trace()
    last_indices=get_last_index_of_page(raw_pdf_elements)
    image_insert_with_text(raw_pdf_elements,last_indices)
    print("image_insert Done")
    print()
    get_docummets()
    
    pdb.set_trace()
    print("get_docummets Done")
    add_docs_to_postgres(collection_name)
    print("Done")

if __name__=="__main__":
    main()