from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from dataset import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

# def inference(model, tokenized_sent, device):
#   dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
#   model[0].eval()
#   model[1].eval()
#   model[2].eval()
#   # model[3].eval()
#   # model[4].eval()
#   output_pred = []
  
#   for i, data in enumerate(dataloader):
#     with torch.no_grad():
#       outputs1 = model[0](
#           input_ids=data['input_ids'].to(device),
#           attention_mask=data['attention_mask'].to(device),
#           token_type_ids=data['token_type_ids'].to(device)
#           )
#       outputs2 = model[1](
#           input_ids=data['input_ids'].to(device),
#           attention_mask=data['attention_mask'].to(device),
#           token_type_ids=data['token_type_ids'].to(device)
#           )
#       outputs3 = model[2](
#           input_ids=data['input_ids'].to(device),
#           attention_mask=data['attention_mask'].to(device),
#           token_type_ids=data['token_type_ids'].to(device)
#           )
#       # outputs4 = model[3](
#       #     input_ids=data['input_ids'].to(device),
#       #     attention_mask=data['attention_mask'].to(device),
#       #     token_type_ids=data['token_type_ids'].to(device)
#       #     )
#       # outputs5 = model[4](
#       #     input_ids=data['input_ids'].to(device),
#       #     attention_mask=data['attention_mask'].to(device),
#       #     token_type_ids=data['token_type_ids'].to(device)
#       #     )
#     logits1 = outputs1[0]
#     logits2 = outputs2[0]
#     logits3 = outputs3[0]
#     # logits4 = outputs4[0]
#     # logits5 = outputs5[0]
#     logits = logits1.detach().cpu().numpy()+logits2.detach().cpu().numpy()+logits3.detach().cpu().numpy()
#     # +logits4.detach().cpu().numpy()
#     # +logits5.detach().cpu().numpy()
#     result = np.argmax(logits, axis=-1)

#     output_pred.append(result)
  
#   return np.array(output_pred).flatten()

def main(args):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = args.device

    result_dir = "/opt/ml/results"
    logits = []
    
    # logits 값 모두 더하기
    for i in range(1, args.number + 1) :
        '''
            필요정보 입력
        '''
        result_no = input(f"{i}번째 모델의 번호 입력 : ")

        logit_path = os.path.join(result_dir, result_no, "logits.npy")

        npy = np.load(logit_path)

        logits.append(npy)

    logits = sum(logits)
    pred_answer = logits.argmax(axis = -1)

    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv(f'/opt/ml/prediction/submission_{args.no}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--number', required = True, type = int, help = "ensemble model counts")  # 몇개 앙상블할지
    parser.add_argument('--device', required = False, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), help = "device sorts : cuda / cpu") # 디바이스 설정
    parser.add_argument('--no', required = True, type = str, help = "input submission_no")
    args = parser.parse_args()
    print(args)
    main(args)
    
