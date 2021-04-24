KLUE Relation Extraction
========================

[개요]
-----
- 지식 그래프 구축을 위한 핵심 구성 요소인 관계추출은 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리에서 매우 중요합니다.
- 문장, 엔티티, 관계에 대한 정보를 통해 문장과 엔티티 사이의 관계를 추론하는 모델을 학습하고, 평가합니다.

[목표]
-----
- 클래스는 크게
1. 인물
2. 단체
3. 관계없음
- 으로 나뉘고, 세부 클래스는 총 42개입니다.
- 문장 내 엔티티 간의 클래스를 분류할 수 있도록 모델을 학습시킵니다.

[프로세스]
--------
1. 데이터셋 구성
> 과정
>> 1. data.tsv(Train, Valid, Test)를 DataFrame으로 불러옵니다.
>> 2. 클래스에 따른 labeling을 진행합니다.
>> 3. 문장 preprocessing을 진행합니다. (세부 전처리 내용은 '''code/README.md''' 참조)
>> 4. 전처리 진행한 문장에 tokenizing을 진행합니다.
>> 5. 모델 학습을 위한 데이터셋을 생성합니다.
>>

2. Training
> 과정
>> 1. dataset.py를 통해 데이터셋(train / valid dataset)을 생성합니다.
>> 2. Tokenizer 및 Model을 지정합니다. (```argparse.model```)
>> 3. TrainingArgument를 통해 Hyperparameter 지정 및 학습에 필요한 파라미터를 정의합니다.
>> 4. Trainer을 통해 학습을 진행합니다.
>> 5. 학습은 Single model train 또는 K-fold model train으로 진행할 수 있습니다. (```argparse.train_type```)
>>
>> 이외 학습시 지정 가능 arguments
>>> ![image](https://user-images.githubusercontent.com/33143335/115949302-0f9d6d80-a50f-11eb-9ebd-9ac712e9fab3.png)
>>>

3. Inference
> 과정
>> 1. Training을 거쳐 학습된 모델을 불러옵니다.
>> 2. test 문장에 대한 예측값을 저장한 뒤, submission.csv 를 생성합니다.
