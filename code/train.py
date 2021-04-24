from dataset import *
import os
import pandas as pd
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, AutoTokenizer, BertTokenizer, BertConfig, BertForSequenceClassification, \
                        ElectraTokenizer, ElectraConfig, ElectraForSequenceClassification, \
                        XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, \
                        FunnelTokenizer, FunnelConfig, FunnelForSequenceClassification, EarlyStoppingCallback, get_cosine_with_hard_restarts_schedule_with_warmup
import argparse
from tokenization_kobert import KoBertTokenizer
from glob import glob
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
import wandb
from adamp import AdamP

def define_tokenizer(name) :
    if name in ["bert-base-multilingual-cased", "sangrimlee/bert-base-multilingual-cased-korquad", "kykim/bert-kor-base"] :
        return BertTokenizer.from_pretrained(name)
    elif name in ["monologg/koelectra-base-v3-discriminator", "kykim/electra-kor-base"] :
        return ElectraTokenizer.from_pretrained(name)
    elif name in ["xlm-roberta-large"] :
        return XLMRobertaTokenizer.from_pretrained(name)
    elif name in ["monologg/kobert"] :
        return KoBertTokenizer.from_pretrained(name)
    elif name in ["kykim/funnel-kor-base"] :
        return FunnelTokenizer.from_pretrained(name)

def define_config(name) :
    if name in ["bert-base-multilingual-cased", "sangrimlee/bert-base-multilingual-cased-korquad", "kykim/bert-kor-base", "monologg/kobert"] :
        return BertConfig.from_pretrained(name)
    elif name in ["monologg/koelectra-base-v3-discriminator", "kykim/electra-kor-base"] :
        return ElectraConfig.from_pretrained(name)
    elif name in ["xlm-roberta-large"] :
        return XLMRobertaConfig.from_pretrained(name)
    elif name in ["kykim/funnel-kor-base"] :
        return FunnelConfig.from_pretrained(name)

def define_model(name, config = None, location = None) :
    # config가 있으면 처음 training하는 경우, 없으면 체크포인트 불러오기
    if name in ["bert-base-multilingual-cased", "sangrimlee/bert-base-multilingual-cased-korquad", "kykim/bert-kor-base", "monologg/kobert"] :
        return BertForSequenceClassification.from_pretrained(name, config = config) if config else BertForSequenceClassification.from_pretrained(location)
    elif name in ["monologg/koelectra-base-v3-discriminator", "kykim/electra-kor-base"] :
        return ElectraForSequenceClassification.from_pretrained(name, config = config) if config else ElectraForSequenceClassification.from_pretrained(location)
    elif name in ["xlm-roberta-large"] :
        return XLMRobertaForSequenceClassification.from_pretrained(name, config = config) if config else XLMRobertaForSequenceClassification.from_pretrained(location)
    elif name in ["kykim/funnel-kor-base"] :
        return FunnelForSequenceClassification.from_pretrained(name, config = config) if config else FunnelForSequenceClassification.from_pretrained(location)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

# 시드 고정
def seed_everything(seed) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 모델 저장 어디까지 됐는지 확인
def model_dir() :
    save_dirs = "/opt/ml/results"
    folder_list = glob(f"{save_dirs}/*")
    
    if not folder_list : return '1'
    else :
        num_list = list(map(lambda x : int(x.split('/')[-1]), folder_list))
        last = max(num_list)
        return str(last + 1)

# 학습
def train(args) :
    seed_everything(args.seed)  # 시드 고정

    wandb.init()
    wandb.config.update(args)
    os.environ['Stage 2'] = 'KLUE model finding'    # project name
    os.environ['WANDB_LOG_MODEL'] = 'true'  # save model weights

    '''
        토크나이저 불러오기
    '''
    MODEL_NAME = args.model # 모델명
    tokenizer = define_tokenizer(MODEL_NAME)

    '''
        데이터 불러오기 ## train
    '''
    train_data = make_dataset("train", (1, 2, 3, 4, 5, 6)) # data 불러오기
    train_labels = train_data.labels.values    # 라벨만 따로

    tokenized_train_data = tokenized_dataset(train_data, tokenizer, MODEL_NAME) # 토크나이징
    train_dataset = MyDataset(tokenized_train_data, train_labels)    # 데이터셋 생성

    '''
        데이터 불러오기 ## valid
    '''
    valid_data = make_dataset("valid", (1, 2, 3, 4, 5, 6)) # data 불러오기
    valid_labels = valid_data.labels.values     # 라벨따로

    tokenized_valid_data = tokenized_dataset(valid_data, tokenizer, MODEL_NAME) # 토크나이징
    valid_dataset = MyDataset(tokenized_valid_data, valid_labels)   # 데이터셋 생성

    ''' 
        option 정의
    '''
    # 모델 디렉토리 불러오기
    model_dir_num = model_dir()

    if args.train_type == "single" :

        '''
            config 정의
        '''
        config = define_config(MODEL_NAME)
        config.num_labels = args.class_num
        resize_num = tokenizer.vocab_size + 4   # 스페셜 토큰 4개
        '''
            모델 정의
        '''
        result_dir = "/opt/ml/results"

        model = define_model(MODEL_NAME, config = config)
        
        model.resize_token_embeddings(resize_num) # 스페셜 토큰을 추가해준 만큼 보캡 사이즈 늘리기
        model.to(args.device)

        '''
            optimizer, scheduler, earlystopping
        '''
        # optimizer = AdamP(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=1000, num_cycles=5)
        # early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.00005)

        '''
        other setting
        '''
        train_args = TrainingArguments(
            output_dir=f'/opt/ml/results/{model_dir_num}',
            save_total_limit=args.save_limits,
            save_steps=args.save_steps,
            num_train_epochs=args.train_epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.train_batch_size,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_dir=f'/opt/ml/logs/{model_dir_num}',
            logging_steps=args.logging_steps,
            seed = args.seed,
            evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                                # `no`: No evaluation during training.
                                                # `steps`: Evaluate every `eval_steps`.
                                                # `epoch`: Evaluate every end of epoch.
            eval_steps = args.save_steps,            # evaluation steps
            # load_best_model_at_end=True,
            label_smoothing_factor=0.5,
            dataloader_num_workers=4,
            # fp16=True
        )

        '''
            Trainer
        '''
        trainer = Trainer(
            model = model,
            args = train_args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,             # evaluation dataset
            compute_metrics = compute_metrics,         # define metrics function
            # optimizers=[optimizer, scheduler],
            # callbacks=[early_stopping]
        )
        trainer.train()


    elif args.train_type == "k-fold" :
        '''
            k-fold
        '''
        k = 5
        kf = KFold(k, random_state= args.seed, shuffle=True)

        results_accuracy = []   # Fold 별 Accuracy를 담는 배열
        results_loss = []       # Fold 별 Loss를 담는 배열
        i = 1
        for train_idx, valid_idx in tqdm(kf.split(train_dataset)) :
            split_train_set = Subset(train_dataset, train_idx)
            split_valid_set = Subset(train_dataset, valid_idx)
            '''
            config 정의
            '''
            config = define_config(MODEL_NAME)
            config.num_labels = args.class_num
            resize_num = tokenizer.vocab_size + 4
            '''
                모델 정의
            '''
            model = define_model(MODEL_NAME, config= config)
            model.resize_token_embeddings(resize_num) # 스페셜 토큰을 추가해준 만큼 보캡 사이즈 늘리기
            model.to(args.device)

            '''
            other setting
            '''
            train_args = TrainingArguments(
                output_dir=f'/opt/ml/results/{model_dir_num}/fold{i}',
                save_total_limit=args.save_limits,
                save_steps=args.save_steps,
                num_train_epochs=args.train_epochs,
                learning_rate=args.lr,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.train_batch_size,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                logging_dir=f'/opt/ml/logs/{model_dir_num}',
                logging_steps=args.logging_steps,
                seed = args.seed,
                evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                                # `no`: No evaluation during training.
                                                # `steps`: Evaluate every `eval_steps`.
                                                # `epoch`: Evaluate every end of epoch.
                eval_steps = 500,            # evaluation steps
                load_best_model_at_end=True,
            )
            '''
                Trainer
            '''
            trainer = Trainer(
                model = model,
                args = train_args,
                train_dataset = split_train_set,
                eval_dataset = split_valid_set,             # evaluation dataset
                compute_metrics = compute_metrics,         # define metrics function
            )
            trainer.train()

            # 전체 evaluate 평가
            evaluate_results = trainer.evaluate(eval_dataset = train_dataset)

            temp_loss = 0.0     # i 번째 fold의 loss
            temp_acc = 0.0      # i 번째 fold의 accuracy

            for k in evaluate_results.keys() :
                if "loss" in k : temp_loss = evaluate_results[k]
                elif "accuracy" in k : temp_acc = evaluate_results[k]

            results_loss.append(temp_loss)
            results_accuracy.append(temp_acc)

            print(f"K-fold {i} loss : {temp_loss}, MEAN loss (until now) : {sum(results_loss) / len(results_loss)}")
            print(f"K-fold {i} accuracy : {temp_acc}, MEAN accuracy (until now) : {sum(results_accuracy) / len(results_accuracy)}")
            i += 1
        

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument('--seed', required = False, type = int, default = 42, help = "seed num (default : 42)")    # 시드 설정
    parser.add_argument('--model', required = True, type = str, help = "input model name")   # 모델 설정
    parser.add_argument('--device', required = False, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), help = "device sorts : cuda / cpu") # 디바이스 설정
    parser.add_argument('--class_num', required = False, type = int, default = 42, help = "classify num (default : 42)")
    parser.add_argument('--lr', required = False, type = float, default = 5e-5, help = "learning rate (default : 5e-5)")
    parser.add_argument('--save_limits', required = False, type = int, default = 3, help = "How much do you want to save models ? (default : 3)")
    parser.add_argument('--save_steps', required = False, type = int, default = 300, help = "Save model every ( N ) steps (default : 300)")
    parser.add_argument('--train_epochs', required = False, type = int, default = 10, help = "Train epochs (default : 10)")
    parser.add_argument('--train_batch_size', required = False, type = int, default = 16, help = "Train batch size (default : 16)")
    parser.add_argument('--eval_batch_size', required = False, type = int, default = 16, help = "Eval batch size (default : 16)")
    parser.add_argument('--warmup_steps', required = False, type = int, default = 300, help = "wramup step (default : 300)")
    parser.add_argument('--weight_decay', required = False, type = float, default =0.01, help = "weight decay (default : 0.01)")
    parser.add_argument('--logging_steps', required = False, type = int, default = 450, help = "logging step (default : 450)")
    parser.add_argument('--eval_steps', required = False, type = int, default = 500, help = "eval step (default : 500)")
    parser.add_argument('--train_type', required = True, type = str, help = "choose 'single' or 'k-fold'")
    args = parser.parse_args()
    print(args) # argument 뭐 있는지 터미널에 띄움

    train(args)