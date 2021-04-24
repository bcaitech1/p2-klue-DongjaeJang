import os
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset
import re
from soynlp.normalizer import *
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, ElectraTokenizer


'''
    1. 처음 불러오고
    2. 라벨 붙이고
    3. 전처리 함수 정하기
    4. 전처리 태우기

    그 다음 5. 토크나이징
'''

'''
    [1] 데이터 불러오기
'''
def load_data(data_type) :
    data_path = "/opt/ml/input/data"
    file_name = ''
    if data_type == "train" :       # 기본 학습 데이터 + 피어세션 정제 데이터
        file_name = "gold_train.tsv"
    elif data_type == "test" :
        file_name = "test.tsv"
    elif data_type == "valid" :     # 외부데이터에서 클래스 당 맥시멈 100개씩 총 2900개 정도 검증데이터
        file_name = "valid.tsv"

    data_csv = pd.read_csv(os.path.join(data_path, data_type, file_name), sep = '\t', header = None)
    
    # 문장, entity1, entity2만 필요
    dataset = pd.DataFrame({'sentence' : data_csv[1], 'entity_1' : data_csv[2], 'entity_2' : data_csv[5]})
    
    # 라벨 추가
    labels = give_labels(data_csv, data_path)
    dataset['labels'] = labels

    return dataset

'''
    [2] 라벨 붙이기
'''
def give_labels(data_csv, data_path) :

    # 라벨 불러오기
    with open(os.path.join(data_path, "label_type.pkl"), "rb") as f :
        label_type = pickle.load(f)
        labels = []
        for relation in data_csv[8] :
            if relation == 'blind':
                labels.append(100)
            else :
                labels.append(label_type[relation])

    return labels

'''
    [3] 전처리할 부분들 정하기
    1. "" "" 로 되어있는 경우 바꾸기 -> "" -> '로
    2. [[ ]] 로 되어있는 경우 바꾸기 -> [[ ]] -> 
    3. 반복되는 문자 줄이기
    4. 이상한 문자 통일
    5. 연속된 공백 하나로 바꾸기
    6. 쓸모없는 글자 지우기 -- 실습코드 참조
'''

def replace_double_quotes(sentences) :
    new_sentences = []
    target = '""'
    for sentence in sentences :
        if target in sentence :
            new_sentence = sentence.replace(target, "'")
            new_sentences.append(new_sentence)
        else :
            new_sentences.append(sentence)
    return new_sentences

def replace_bracket(sentences) :
    new_sentences = []

    for sentence in sentences :
        if "[[" in sentence :
            new_sentence = sentence.replace("[[", "")
            new_sentence = new_sentence.replace("]]", "")

            new_sentences.append(new_sentence)
        else :
            new_sentences.append(sentence)
    
    return new_sentences

def remove_repeated_char(sentences) :
    new_sentences = []
    for sentence in sentences :
        new_sentence = repeat_normalize(sentence, num_repeats=2).strip()
        new_sentences.append(new_sentence)

    return new_sentences

def clean_punc(sentences):
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', "《" : "'", "》" : "'", "●" : "", "▲" : "", "〈" : "'", "〉" : "'" }

    new_sentences = []
    for sentence in sentences:
        for p in punct_mapping:
            sentence = sentence.replace(p, punct_mapping[p])
        sentence = sentence.strip()
        if sentence:
            new_sentences.append(sentence)
    return new_sentences


def remove_repeated_space(sentences) :
    new_sentences = []
    for sentence in sentences :
        sentence = re.sub("\s+", " ", sentence).strip()
        new_sentences.append(sentence)
    return new_sentences

def remove_useless_breacket(texts):
    """
    위키피디아 전처리를 위한 함수입니다.
    괄호 내부에 의미가 없는 정보를 제거합니다.
    아무런 정보를 포함하고 있지 않다면, 괄호를 통채로 제거합니다.
    ``수학(,)`` -> ``수학``
    ``수학(數學,) -> ``수학(數學)``
    """
    bracket_pattern = re.compile(r"\((.*?)\)")
    preprocessed_text = []
    for text in texts:
        modi_text = ""
        text = text.replace("()", "")  # 수학() -> 수학
        brackets = bracket_pattern.search(text)
        if not brackets:
            if text:
                preprocessed_text.append(text)
                continue
        replace_brackets = {}
        # key: 원본 문장에서 고쳐야하는 index, value: 고쳐져야 하는 값
        # e.g. {'2,8': '(數學)','34,37': ''}
        while brackets:
            index_key = str(brackets.start()) + "," + str(brackets.end())
            bracket = text[brackets.start() + 1 : brackets.end() - 1]
            infos = bracket.split(",")
            modi_infos = []
            for info in infos:
                info = info.strip()
                if len(info) > 0:
                    modi_infos.append(info)
            if len(modi_infos) > 0:
                replace_brackets[index_key] = "(" + ", ".join(modi_infos) + ")"
            else:
                replace_brackets[index_key] = ""
            brackets = bracket_pattern.search(text, brackets.start() + 1)
        end_index = 0
        for index_key in replace_brackets.keys():
            start_index = int(index_key.split(",")[0])
            modi_text += text[end_index:start_index]
            modi_text += replace_brackets[index_key]
            end_index = int(index_key.split(",")[1])
        modi_text += text[end_index:]
        modi_text = modi_text.strip()
        if modi_text:
            preprocessed_text.append(modi_text)
    return preprocessed_text

'''
    [4] 전처리 태우기
'''
def make_dataset(data_type, preprocessing_type = []) :
    dataset = load_data(data_type)
    # 1 : "" -> '
    if 1 in preprocessing_type :
        dataset.sentence = replace_double_quotes(dataset.sentence)
    # 2 : [[, ]] -> ', '
    if 2 in preprocessing_type :
        dataset.sentence = replace_bracket(dataset.sentence)
    # 3 : ㅋㅋㅋㅋ -> ㅋㅋ
    if 3 in preprocessing_type :
        dataset.sentence = remove_repeated_char(dataset.sentence)
    # 4 : 스페이스 제거
    if 4 in preprocessing_type :
        dataset.sentence = remove_repeated_space(dataset.sentence)
    # 5 : 문자 통일
    if 5 in preprocessing_type :
        dataset.sentence = clean_punc(dataset.sentence)
    # 6 : 쓰레기 문자 버리기
    if 6 in preprocessing_type :
        dataset.sentence = remove_useless_breacket(dataset.sentence)

    return dataset
'''
    [5] 토크나이징
'''
def tokenized_dataset(dataset, tokenizer, model_type) :
    concat_entity = []
    for e1, e2 in zip(dataset.entity_1, dataset.entity_2) :
        if model_type != "xlm-roberta-large" :
            temp = e1 + '[SEP]' + e2  # 기본
            concat_entity.append(temp)
        else :
            temp = e1 + '</s></s>' + e2 # xml-roberta-large 일때 sep 토큰 변경
            concat_entity.append(temp)

    # 토크나이저에 special 토큰 추가해주기
    tokenizer.add_special_tokens({"additional_special_tokens":["[EN1]", "[/EN1]", "[EN2]", "[/EN2]"]})
    # 문장에도 special 토큰 추가
    new_sentences = []
    for i, (e1, e2) in enumerate(zip(dataset.entity_1, dataset.entity_2)) :
        new_sentence = dataset.sentence[i].replace(e1, f"[EN1]{e1}[/EN1]")      # 엔티티 1 앞뒤
        new_sentence = new_sentence.replace(e2, f"[EN2]{e2}[/EN2]")             # 엔티티 2 ㅇ ㅏㅍ뒤
        new_sentences.append(new_sentence)

    dataset.sentence = new_sentences

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length = 100,   # 최대길이가 100으로 잘림
        add_special_tokens = True
    )

    return tokenized_sentences

'''
    데이터셋 만들기
'''
class MyDataset(Dataset) :
    def __init__(self, tokenized_dataset, labels) :
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
    
    def __getitem__(self, index) :
        # item = {key : torch.tensor(val[index]) for key, val in self.tokenized_dataset.items()}    # 에러메세지 뜸 이대로 하면.
        item = {key : val[index].clone().detach() for key, val in self.tokenized_dataset.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self) :
        return len(self.labels)