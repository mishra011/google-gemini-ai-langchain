import os
import sys
from dotenv import load_dotenv

global temperature
global max_tokens

temperature = .1
max_tokens = 20




def chat(query, chain, history):
    result = chain({"question": query, "chat_history": history })
    return result



def create_gemini_chain(vector_store):
    from langchain.memory import ConversationBufferMemory
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import ConversationalRetrievalChain

    global temperature
    global max_tokens

    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    #model = "gemini-pro"
    model = "gemini-1.5-pro-latest"
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain


def get_text_chunk_langchain(context):
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.schema.document import Document

    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=500,
        chunk_overlap = 100,
        length_function=len
    )

    text_chunks = [Document(page_content=x) for x in text_splitter.split_text(context)]

    return text_chunks



def train(context):
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    text_chunks = get_text_chunk_langchain(context)

    embed_model = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs= {'device':'cpu'}
    )

    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    
    model = create_gemini_chain(vector_store)

    return model



context = " Hi I am Deepak. I am a developer. How can I help you?"

model = train(context)

import time

while True:
    query = input("TYPE HERE >>>>>>>")
    start = time.time()
    response = chat(query, model, [])
    end = time.time()
    print("RESPONSE :: ", response)
    print("TIME TAKEN :: ", end-start)
    print("------------------------------------------------")

