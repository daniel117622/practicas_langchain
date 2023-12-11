import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import *
from langchain.memory import *
from langchain.chains import LLMChain , SequentialChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

API_KEY = dotenv.get_key(".env","OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=API_KEY)
embeddings = OpenAIEmbeddings()

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
docs = loader.load_and_split(text_splitter=text_splitter)

# Create a small database
db = Chroma.from_documents(
    docs,                   # Calls openai to generate embeddings
    embedding=embeddings,   # Reference to openai
    persist_directory="emb" # Directory of db
)

results = db.similarity_search_with_score("What is an interesting language?")
for result in results:
    print("\n")
    print(result[1])
    print(results[0].page_content)


