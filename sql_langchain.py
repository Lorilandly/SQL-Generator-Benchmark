from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

#############################################
# Create DB
#############################################
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())

#############################################
# Create DB
#############################################
import getpass
import os

#############################################
# Create OpenAI LLM
#############################################
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain import hub

#############################################
# Create Prompt
#############################################
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1

from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(db, state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

print(write_query(db, {"question": "What is the name of the game genre with the most players?"}))
exit()

#############################################
# assemble chain
#############################################

execute_query = QuerySQLDataBaseTool(db=db)

from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate

# Define your custom prompt template
prompt = PromptTemplate(
    input_variables=["instruction", "input", "response"],  # Variables to format
    template="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""
)

# Example Usage in LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define a prompt
prompt = PromptTemplate(
    input_variables=["query"],
    template="Answer the following SQL-related query: {query}",
)

# Create a LangChain with the LLM and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with an input query
response = chain.run("What is the average explainability score in the creative_ai table?")
print(response)

