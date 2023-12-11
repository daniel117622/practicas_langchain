import dotenv
from langchain.llms import OpenAI


API_KEY = dotenv.get_key(".env","OPENAI_API_KEY")
print(API_KEY)

llm = OpenAI(openai_api_key=API_KEY)
print(llm("hello"))