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

query = "Do que se trata as informações disponíveis? Responda sempre em português."



answer = wrapper_faiss_chat.query_with_sources(question = query,llm=llm)
print(answer['answer'])

# Meta 8B
# llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",
#             client=bedrock,
#                 model_kwargs={'max_gen_len':512})

#def create_llm():
# llm=Bedrock(model_id="anthropic.claude-v2",
#             client=bedrock,
#             verbose = True,
#                 model_kwargs={'max_gen_len':512,
#                               'temperature': 0.1,
#                               'top_p': 0.9
#                               })
# # return llm
# prompt_template = """

# Human: Use the following pieces of context to answer the question.
# <context>
# {context}
# </context

# Question: {question}

# Assistant:"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )


# query = "O que é uma imagem?"

# # Realizar a busca por similirdade, retorna a quantidade especificada pelo K
# contexto = db.similarity_search(query,k=3,return_metadata=True)