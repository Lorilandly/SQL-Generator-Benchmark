import os
import json
import numpy as np
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities import SQLDatabase

#############################################
# Create DB
#############################################
import os

#############################################
# Load inference result
#############################################
# [prompt, context, pred, y]
with open("outputs/generated_sql_results300.json", "r") as f:
    data = np.array(json.load(f))

# Define a function to extract the "Response" section
def extract_response(text):
    return text.split("### Response:\n")[-1].strip()

total = 0
reference_error = 0
ft_match = 0
ft_same = 0
pe_match = 0
pe_same = 0

for i, row in enumerate(data):
    prompt, context, pred, y = row[0], row[1], extract_response(row[2]), row[3]
    print('\n', i, '/', len(data))

    os.remove('Chinook.db')
    #############################################
    # Create DB
    #############################################
    print(context)
    engine = create_engine("sqlite:///Chinook.db")
    with engine.connect() as connection:
        for command in context.split(";"):
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
    print(f'Test:    {y}\nFT Pred: {pred}')
    total += 1
    if pred == y:
        ft_same += 1
    try:
        ft_res = db.run(pred)
    except Exception as e:
        print(e)
        ft_res = 'error'
    try:
        test_res = db.run(y)
    except Exception as e:
        print(e)
        reference_error += 1
        test_res = 'error'
    print(f'Test:    {test_res}\nFT Pred: {ft_res}')
    if test_res == ft_res:
        ft_match += 1
    
print(f'ft: {ft_match}/{total}, {ft_match/total*100}%')
print(f'ft same: {ft_same}')
print(f'Ref error: {reference_error}')
