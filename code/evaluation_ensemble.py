from torch.utils.data import DataLoader
from dataset import *
from train import *
from evaluation import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os

def evaluation_ensemble(model, tokenized_sent, device):
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

        output_pred.append(logits)
    # 주의 # flatten할 때는 batch size로 정확히 나누어 떨어지는 경우만 되는 것 같다
    return np.array(output_pred)

def main(args):
    """
        주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = args.device
    result_dir = "/opt/ml/results"

    labels = None
    pred_answer_total = []
    # 모델마다 토크나이저 지정
    for i in range(1, args.number + 1) :
        '''
            필요정보 입력
        '''
        TOK_NAME = input(f"{i}번째 모델의 이름 입력 : ")
        tokenizer = define_tokenizer(TOK_NAME)  # 토크나이저
        
        result_no = input(f"{i}번째 모델의 번호 입력 : ")
        checkpoint_no = input(f"{i}번째 모델의 체크포인트 입력 : ")

        MODEL_NAME = os.path.join(result_dir, result_no, f"checkpoint-{checkpoint_no}")
        model = define_model(TOK_NAME, location=MODEL_NAME) # 모델
        model.resize_token_embeddings(tokenizer.vocab_size + 4) # 스페셜토큰 4개 추가

        model.to(device)

        preprocessing_type = (1, 2, 3, 4, 5, 6)
        '''
            데이터셋
        '''
        valid_dataset, valid_labels = load_valid_dataset(tokenizer, preprocessing_type, TOK_NAME)
        if labels == None : labels = valid_labels
        valid_dataset = MyDataset(valid_dataset, valid_labels)

        # predict
        pred_answer = evaluation_ensemble(model, valid_dataset, device)

        pred_answer_total += pred_answer

    # predict answer
    pred_answer = []

    for pred_answers in pred_answer_total :
        result = np.argmax(pred_answers, axis=-1)
        pred_answer += list(result)
    
    acc = ((pred_answer == labels).sum() / len(pred_answer)) * 100
    
    print(f"label 분포 : {Counter(list(labels))}")
    print(f"예측 분포 : {Counter(list(pred_answer))}")
    print(f"Valid_data label과의 정확도 : {acc}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--number', required = True, type = int, help = "ensemble model counts")  # 몇개 앙상블할지
    parser.add_argument('--device', required = False, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), help = "device sorts : cuda / cpu") # 디바이스 설정
    
    args = parser.parse_args()
    print(args)
    main(args)
    
