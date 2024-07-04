import os

import boto3
from botocore.config import Config
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock, verbose=True)

def save_vector_local():

    loader=PyPDFLoader("Containers_com_Docker.pdf")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, 
                                                    chunk_overlap=10,separators=['\n\n','\n'," ",""])

    docs=text_splitter.split_documents(documents)

    vectorstore_faiss=FAISS.from_documents(
    documents = docs,
    embedding = bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def load_vector():

    faiss_vectorsstore = FAISS.load_local(folder_path="faiss_index",embeddings=bedrock_embeddings)

    return faiss_vectorsstore


wrapper_faiss_chat = VectorStoreIndexWrapper(vectorstore=load_vector())

query = "Como criar uma imagem no Linux? Responda sempre em português."

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=load_vector().as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
),
return_source_documents=True,
chain_type_kwargs={"prompt": PROMPT}
)
answer=qa({"query":query})
print(answer)#['result'])