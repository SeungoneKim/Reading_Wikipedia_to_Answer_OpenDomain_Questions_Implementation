import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys

def Key_Information_Extraction_BERT(context, query, look_at_internal=False):
    # Check Context and Query given
    print("Context passed to module is : ")
    print(context)
    print()
    print("Query passed to module is : ")
    print(query)
    print()
    
    context_len = len(context)
    query_len = len(query)
    
    # setting up
    if look_at_internal:
        print("Setting up Finetuned tokenizer and model ...")
        print()
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    if look_at_internal:
        print("Tokenizer ready!")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    if look_at_internal:
        print("Model ready!")
    
    # Document Processing & Query Processing
    model_inputs = tokenizer(query,context,return_tensors="pt")
    if look_at_internal:
        print("Progressing Document Processing & Query Processing ...")
        print()
        print("Language Representation for Query is : ")
        print(model_inputs['input_ids'][1:query_len+1])
        print()
        print("Language Representation for Context is : ")
        print(model_inputs['input_ids'][query_len+3:-1])
        print()
    
    # Information Processing
    def Reading_Comprehension_for_Keyword_Extraction(QAModel_input):
        
        # Processing with Language Model
        QAModel_output = model(**model_inputs)
        if look_at_internal:
            print('Output of model is : ')
            print(model_outputs)
            print()
        
        # Keyword Extractor
        start_idx = (QAModel_output['start_logits'])[:,(query_len+3):].argmax(dim=1)+(query_len+3)
        end_idx = (QAModel_output['end_logits'])[:,(query_len+3):].argmax(dim=1)+(query_len+3)
        if start_idx <= end_idx:
            candidate_answer = (QAModel_input['input_ids'])[:,start_idx:end_idx+1]
        
        else:
            tmp_start_idx = (QAModel_output['start_logits'])[:,(query_len+3):end_idx+1].argmax(dim=1)+(query_len+3)
            tmp_end_idx = (QAModel_output['end_logits'])[:,start_idx:].argmax(dim=1)+start_idx
            
            choose_start = (QAModel_output['start_logits'][:,start_idx].item()) + (QAModel_output['end_logits'][:,tmp_end_idx].item())
            choose_end = (QAModel_output['start_logits'][:,tmp_start_idx].item()) + (QAModel_output['end_logits'][:,end_idx].item())
            
            if choose_start >= choose_end:
                candidate_answer = (QAModel_input['input_ids'])[:,start_idx:tmp_end_idx+1]
                end_idx = tmp_end_idx
            else:
                candidate_answer = (QAModel_input['input_ids'][:,tmp_start_idx:end_idx+1])
                start_idx = tmp_start_idx
        
        print('Start position and End position are : ')
        print('Start : ',start_idx.item())
        print('End : ',end_idx.item())
        print()
        
        return candidate_answer, start_idx, end_idx
    
    if look_at_internal:
        print('Processing within model ...')
        print('Progressing Information Processing for Key Information Extraction ...')
    
    # Reading Comprehension for Keyword Extraction
    candidate_answer,si,ei = Reading_Comprehension_for_Keyword_Extraction(model_inputs)
    if look_at_internal:
        print('Candidate answer(Key Information before decoding) is : ')
        print(candidate_answer)
        print()
    
    # Decoding Candidate Answer into Key Information
    def de_tokenize(x):
        # input should be [bs,sl]
        decoded_output = []
        for i in x:
            k = tokenizer.decode(i,skip_special_tokens=True)
            decoded_output.append(k)
        return decoded_output
    if look_at_internal:
        print('Decoding the candidate answer into Key Information ...')
    key_information = de_tokenize(candidate_answer)
    key_information_for_print = key_information[0]
    print('The Key information given the context and query is : ')
    print(key_information_for_print)
    
    return key_information

def Key_Information_Extraction_RoBERTa(context, query, look_at_internal=False):
    # Check Context and Query given
    print("Context passed to module is : ")
    print(context)
    print()
    print("Query passed to module is : ")
    print(query)
    print()
    
    context_len = len(context)
    query_len = len(query)
    
    # setting up
    if look_at_internal:
        print("Setting up Finetuned tokenizer and model ...")
        print()
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    if look_at_internal:
        print("Tokenizer ready!")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    if look_at_internal:
        print("Model ready!")
    
    # Document Processing & Query Processing
    model_inputs = tokenizer(query,context,return_tensors="pt")
    if look_at_internal:
        print("Progressing Document Processing & Query Processing ...")
        print()
        print("Language Representation for Query is : ")
        print(model_inputs['input_ids'][1:query_len+1])
        print()
        print("Language Representation for Context is : ")
        print(model_inputs['input_ids'][query_len+3:-1])
        print()
    
    # Information Processing
    def Reading_Comprehension_for_Keyword_Extraction(QAModel_input):
        
        # Processing with Language Model
        QAModel_output = model(**model_inputs)
        if look_at_internal:
            print('Output of model is : ')
            print(model_outputs)
            print()
        
        # Keyword Extractor
        start_idx = (QAModel_output['start_logits'])[:,(query_len+3):].argmax(dim=1)+(query_len+3)
        end_idx = (QAModel_output['end_logits'])[:,(query_len+3):].argmax(dim=1)+(query_len+3)
        if start_idx <= end_idx:
            candidate_answer = (QAModel_input['input_ids'])[:,start_idx:end_idx+1]
        
        else:
            tmp_start_idx = (QAModel_output['start_logits'])[:,(query_len+3):end_idx+1].argmax(dim=1)+(query_len+3)
            tmp_end_idx = (QAModel_output['end_logits'])[:,start_idx:].argmax(dim=1)+start_idx
            
            choose_start = (QAModel_output['start_logits'][:,start_idx].item()) + (QAModel_output['end_logits'][:,tmp_end_idx].item())
            choose_end = (QAModel_output['start_logits'][:,tmp_start_idx].item()) + (QAModel_output['end_logits'][:,end_idx].item())
            
            if choose_start >= choose_end:
                candidate_answer = (QAModel_input['input_ids'])[:,start_idx:tmp_end_idx+1]
                end_idx = tmp_end_idx
            else:
                candidate_answer = (QAModel_input['input_ids'][:,tmp_start_idx:end_idx+1])
                start_idx = tmp_start_idx
        
        print('Start position and End position are : ')
        print('Start : ',start_idx.item())
        print('End : ',end_idx.item())
        print()
        
        return candidate_answer, start_idx, end_idx
    
    if look_at_internal:
        print('Processing within model ...')
        print('Progressing Information Processing for Key Information Extraction ...')
    
    # Reading Comprehension for Keyword Extraction
    candidate_answer,si,ei = Reading_Comprehension_for_Keyword_Extraction(model_inputs)
    if look_at_internal:
        print('Candidate answer(Key Information before decoding) is : ')
        print(candidate_answer)
        print()
    
    # Decoding Candidate Answer into Key Information
    def de_tokenize(x):
        # input should be [bs,sl]
        decoded_output = []
        for i in x:
            k = tokenizer.decode(i,skip_special_tokens=True)
            decoded_output.append(k)
        return decoded_output
    if look_at_internal:
        print('Decoding the candidate answer into Key Information ...')
    key_information = de_tokenize(candidate_answer)
    key_information_for_print = key_information[0]
    print('The Key information given the context and query is : ')
    print(key_information_for_print)
    
    return key_information