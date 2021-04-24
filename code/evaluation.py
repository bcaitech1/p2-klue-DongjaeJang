from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os
from dataset import *
from train import *
from collections import Counter


def load_valid_dataset(tokenizer, preprocessing_type, model_type):
    dataset = make_dataset("train", preprocessing_type = preprocessing_type)
    data_labels = dataset['labels'].values
    # tokenizing dataset
    tokenized_data = tokenized_dataset(dataset, tokenizer, model_type)

    return tokenized_data, data_labels

def evaluation(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
  
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device), 
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            ) if "token_type_ids" in data.keys() else model(  # xlm roberta large 일때
                input_ids=data['input_ids'].to(device), 
                attention_mask=data['attention_mask'].to(device)
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    # 주의 # flatten할 때는 batch size로 정확히 나누어 떨어지는 경우만 되는 것 같다
    return np.array(output_pred).flatten()

def main(args):
    """
        주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = args.device
    # load tokenizer
    TOK_NAME = args.tokenizer
    tokenizer = define_tokenizer(TOK_NAME)

    # load my model
    result_dir = "/opt/ml/results"

    MODEL_NAME = os.path.join(result_dir, args.result_no, f"checkpoint-{args.checkpoint_no}")   # checkpoint 위치
    model = define_model(TOK_NAME, location = MODEL_NAME)
    # 스페셜 토큰 추가
    model.resize_token_embeddings(tokenizer.vocab_size + 4)

    model.to(device)

    # load test datset
    dataset, labels = load_valid_dataset(tokenizer, TOK_NAME)
    dataset = MyDataset(dataset, labels)

    # predict answer
    pred_answer = evaluation(model, dataset, device)

    acc = ((pred_answer == labels).sum() / len(pred_answer)) * 100
    
    print(f"label 분포 : {Counter(list(labels))}")
    print(f"예측 분포 : {Counter(list(pred_answer))}")
    print(f"Valid_data label과의 정확도 : {acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--tokenizer', required = True, type = str, help = "tokenizer name")   # 토크나이저 설정
    parser.add_argument('--device', required = False, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), help = "device sorts : cuda / cpu") # 디바이스 설정
    parser.add_argument('--result_no', required = True, type = str, help = "result_no in results folder")
    parser.add_argument('--checkpoint_no', required = True, type = str, help = "checkpoint_no")
    args = parser.parse_args()
    print(args)
    main(args)

