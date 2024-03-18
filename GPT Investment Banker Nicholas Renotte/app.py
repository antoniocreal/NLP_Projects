import subprocess

# Install required packages
subprocess.run(['pip', 'install', 'langchain', 'langchain-pinecone', 'langchain_openai', 'openai', 'streamlit', 'tiktoken', 'pypdf', 'pycryptodome', 'chromadb', 'pysqlite3-binary', 'sentence-transformers', 'pinecone-client'])

# Import os to set API key
import os 

# Import OpenAI as main LLM services
from langchain_openai import OpenAI, OpenAIEmbeddings

# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...
from langchain_community.document_loaders import PyPDFLoader
# Import Chroma as the vector store
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set API key for OpenAI Service
# It has to really be OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = '-----------------------------'

os.environ['PINECONE_API_KEY'] = '------------------------'
from pinecone import Pinecone

pc = Pinecone(api_key="--------------------------")
index = pc.Index("------")
           
# Create instance of OpenAI LLM
# The higher the temperature, the more randomness the model generates
llm = OpenAI(temperature=0.9)

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf
pages = loader.load_and_split()

from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings()

# Load documents into vector database aka Pinecone or Chroma
# Pinecone.create_index(index_name, dimension=1024, metric="cosine", pods=1, pod_type="starter")
# store = PineconeVectorStore.from_documents(pages, embeddings, index_name=index)
store = Chroma.from_documents(pages, embedding_function, collection_name= 'annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name='annual_report',
    description="A banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC 
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Create a text input box for the user
prompt = st.text_input('Input your prompt here:')

# If the user hits enter
if prompt:
    # Then pass the promt to the LLM
    # response = llm(prompt)

    # Swap out the raw llm for a document agent
    response = agent_executor.run(prompt)

    # ... and write it out to the screen
    st.write(response)

    # With a streamlit expander
    # with st.expander('Document Similarity Search'):
    #     # Find the relevant pages based on similarity
    #     search = store.similarity_search_with_score(prompt)
    #     st.write(search[0][0].page_content)
