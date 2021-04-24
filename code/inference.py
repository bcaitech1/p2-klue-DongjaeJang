from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader
from dataset import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from tokenization_kobert import KoBertTokenizer
from train import *

def inference(model, tokenized_sent, device, model_name):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []  # prediction

    output_logits = []  # logits
    
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
        output_logits.append(logits)
    
    return np.array(output_pred).flatten(), np.concatenate(output_logits)

def load_test_dataset(tokenizer, preprocessing_type, model_type):
    test_dataset = make_dataset("test", preprocessing_type= preprocessing_type)
    test_label = test_dataset['labels'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, model_type)
    return tokenized_test, test_label

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

    model.parameters
    model.to(device)

    # 전처리
    preprocessing_type = (1, 2, 3, 4, 5, 6)

    # load test datset
    test_dataset, test_label = load_test_dataset(tokenizer, preprocessing_type, TOK_NAME)
    test_dataset = MyDataset(test_dataset, test_label)

    # predict answer
    pred_answer, pred_logits = inference(model, test_dataset, device, TOK_NAME)
    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(pred_answer, columns=['pred'])
    np.save(os.path.join(os.path.join(result_dir, args.result_no), r'logits.npy'), pred_logits)   # logits 저장 (ensemble용)
    output.to_csv(f'/opt/ml/prediction/submission_{args.result_no}.csv', index=False)   # prediction 저장 (submission용)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--tokenizer', required = True, type = str, help = "tokenizer name")   # 모델 설정
    parser.add_argument('--device', required = False, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), help = "device sorts : cuda / cpu") # 디바이스 설정
    parser.add_argument('--result_no', required = True, type = str, help = "result_no")
    parser.add_argument('--checkpoint_no', required = True, type = str, help = "checkpoint_no")
      
    args = parser.parse_args()
    print(args)
    main(args)
  
