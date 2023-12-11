import dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SequentialChain

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--task", default="list of numbers 1 to 10")
parser.add_argument("--lang", default="python")
args = parser.parse_args()

API_KEY = dotenv.get_key(".env","OPENAI_API_KEY")
llm = OpenAI(openai_api_key=API_KEY)

# Prompt template A
code_gen_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language","task"]
)
# Prompt template B
code_ver_prompt = PromptTemplate(
    input_variables=["code", "language"],
    template="Write a test for the following {language} , code : \n {code}"
)

#Chain definition A
ch_code_gen = LLMChain(llm=llm , prompt=code_gen_prompt ,output_key="code")
#Chain definition B
ch_code_ver = LLMChain(llm=llm , prompt=code_ver_prompt ,output_key="test")

#Chain wiring definition
chain = SequentialChain(
    chains=[ch_code_gen , ch_code_ver],
    input_variables=["task","language"],
    output_variables=["test","code"]
)

result = chain({
    "language":args.lang,
    "task":args.task
})

print(result["test"])
print(result["code"])

