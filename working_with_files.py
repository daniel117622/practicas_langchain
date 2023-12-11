import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import *
from langchain.memory import *
from langchain.chains import LLMChain , SequentialChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Redis
import pinecone
import uuid

file = os.open(".env" , os.O_RDONLY)

pinecone.init(api_key=API_PINECONE , environment="gcp-starter")

API_KEY = str(os.read(file,128)).split("=")[1][0:-1]
print(f"KEY: {API_KEY}----------")
chat = ChatOpenAI(openai_api_key=API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# When using a summary we need to add a llm.
memory = ConversationSummaryMemory(
    memory_key="messages", 
    return_messages=True,
    chat_memory=FileChatMessageHistory("messages.json"),
    llm=chat)



text_splitter = CharacterTextSplitter(
    separator='\n', # Then the separator is used to find the nearest newline
    chunk_size=200, # First the chunk size is used
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
pre_docs = loader.load_and_split(text_splitter=text_splitter)
docs = []
for d in pre_docs:
    docs.append(d.page_content)

# This runs only if db is not created
index_name = "langchain-vectorstore"
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

    index = pinecone.Index(index_name)

    embeds = embeddings.embed_documents(docs)
    ids = [str(hash(_)) for _ in range(len(docs))]
    index.upsert(vectors=zip(ids,embeds))

index = pinecone.Index(index_name)
print(index.describe_index_stats())