import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from src.run import Key_Information_Extraction_BERT, Key_Information_Extraction_RoBERTa
import sys

if __name__ == "__main__":

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You have entered __main__.\n')
    sys.stdout.write('#################################################\n')

    sys.stdout.write('#################################################\n')
    context_txt = open('context.txt','r')
    context = ""
    for line in context_txt:
        context += line
    print('The given context is : ')
    print()
    print(context)
    sys.stdout.write('#################################################\n')

    sys.stdout.write('#################################################\n')
    query = input('What is your query : ')
    print('The given query is : ')
    print()
    print(query)
    sys.stdout.write('#################################################\n')

    sys.stdout.write('#################################################\n')
    model = input('Which model would you use(BERT / RoBERTa) : ')
    sys.stdout.write('#################################################\n')

    if model == 'BERT':
        Key_Information_Extraction_BERT(context, query, False)
    elif model == 'RoBERTa':
        Key_Information_Extraction_RoBERTa(context, query, False)
    else:
        assert "Model is not supported yet!"

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You are exiting __main__.\n')
    sys.stdout.write('#################################################\n')
