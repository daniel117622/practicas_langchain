import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import *
from langchain.memory import *
from langchain.chains import LLMChain , SequentialChain

API_KEY = dotenv.get_key(".env","OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=API_KEY)

# When using a summary we need to add a llm.
memory = ConversationSummaryMemory(
    memory_key="messages", 
    return_messages=True,
    chat_memory=FileChatMessageHistory("messages.json"),
    llm=chat,

    )

prompt = ChatPromptTemplate(
    input_variables=["content","messages"],
    messages=[
        # It looks for a variable called messages on the chain
        # Then it expands into Human and AI messages
        MessagesPlaceholder(variable_name='messages'), 
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input("> ")

    # The input variables must be equal to the CPT inputs
    result=chain({"content":content})

    print(result["text"])