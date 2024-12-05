import os
import json
import numpy as np
from datasets import load_dataset
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

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

#############################################
# Load inference result
#############################################
with open("outputs/generated_sql_results.json", "r") as f:
    data = np.array(json.load(f))

# Define a function to extract the "Response" section
def extract_response(text):
    return text.split("### Response:\n")[-1].strip()

vectorized_extract_response = np.vectorize(extract_response)

y_predict = vectorized_extract_response(data[:, 0])
y_test = data[:, 1]

#############################################
# Load Dataset
#############################################
dataset = load_dataset("gretelai/synthetic_text_to_sql", split = "test")


# Load the fine-tuned model and tokenizer
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


total = 0
reference_error = 0
ft_match = 0
ft_same = 0
pe_match = 0
pe_same = 0

for i, row in enumerate(dataset):
    if i >= len(y_test):
        break
    print('\n', i, '/', len(y_test))

    if row['sql'][:6] != 'SELECT':
        continue
    os.remove('Chinook.db')
    #############################################
    # Create DB
    #############################################
    print(row['sql_context'])
    engine = create_engine("sqlite:///Chinook.db")
    with engine.connect() as connection:
        for command in row['sql_context'].split(";"):
            if command.strip():
                try:
                    connection.execute(text(command.strip() + ";"))
                except:
                    break
        connection.commit()
    try:
        db = SQLDatabase(engine)
    except:
        continue
    pe_query = write_query(db, {"question": row['sql_prompt']})['query']
    print(f'Test:    {y_test[i]}\nFT Pred: {y_predict[i]}\nPE Pred: {pe_query}')
    total += 1
    if pe_query == y_test[i]:
        pe_same += 1
    try:
        pe_res = db.run(pe_query)
    except Exception as e:
        print(e)
        pe_res = 'error'
    if y_predict[i] == y_test[i]:
        ft_same += 1
    try:
        ft_res = db.run(y_predict[i])
    except Exception as e:
        print(e)
        ft_res = 'error'
    try:
        test_res = db.run(y_test[i])
    except Exception as e:
        print(e)
        reference_error += 1
        test_res = 'error'
    print(f'Test:    {test_res}\nFT Pred: {ft_res}\nPE Pred: {pe_res}')
    if test_res == ft_res:
        ft_match += 1
    if test_res == pe_res:
        pe_match += 1
    continue
    
print(f'ft: {ft_match}/{total}, {ft_match/total*100}%')
print(f'pe: {pe_match}/{total}, {pe_match/total*100}%')
print(f'ft same: {ft_same}, pe same: {pe_same}')
print(f'Ref error: {reference_error}')
