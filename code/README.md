# code 간략 설명

## Components

### dataset.py
- 학습 또는 테스트에 필요한 데이터셋을 생성하는 컴포넌트
    - 크게 5가지 단계로 생각했음

        1. 데이터 불러오기
            - load_data(data_type) :
                - data_type : 
                    - train / valid / test
                - train, valid, test tsv를 가져오는 메소드
                - 필요한 정보인 문장, 엔티티1, 엔티티2, 라벨을 데이터프레임 형태로 리턴

        2. 라벨 붙이기
            - give_labels(data_csv, data_path) :
                - data_csv :
                    - 데이터프레임
                - data_path :
                    - label_type.pkl 위치
                - 1번 과정 중에 호출
                - 데이터프레임의 클래스에 맞는 라벨 부착

        3. 전처리 함수 정하기
            - replace_double_qoutes(sentences) :
                - "" "" 로 되어있는 인용문을 ' '로 바꿈
                - sentences를 받아서 전처리
            - replace_bracket(sentences) :
                - [[ ]] 로 되어있는 인용문을 없앰
            - remove_repeated_char(sentences) :
                - 반복되는 문자를 줄임
                - ex) ㅋㅋㅋㅋㅋㅋㅋ -> ㅋㅋ
            - clean_punc(sentences) :
                - 문자 통일
            - remove_repeated_space(sentences) :
                - 공백 중복을 하나로 통일
            - remove_useless_bracket(sentences) :
                - 괄호 내부 의미없는 정보 제거

        4. 전처리 태우기
            - 원하는 번호로 전처리를 선택해서 진행

        5. 토크나이징 진행
            - 엔티티1, 엔티티2에 스페셜 토큰 추가
            - 엔티티1, 엔티티2 사이에 [SEP] 토큰 추가 ("xlm-roberta-large" 모델의 경우 </s></s>)
        
### train.py
- 학습 진행 컴포넌트
    - 모델, 토크나이저, config를 불러오는 함수 지정
    - 평가 지표 함수 지정
    - 시드 고정 함수 지정
    - 모델 저장 폴더를 정하는 함수 지정
    - single 및 k-fold로 나누어 학습 진행할 수 있도록 분할
    - argparse를 통해 터미널에서 간단히 진행할 수 있도록 지정

### evaluation.py
- 검증셋으로 평가하기위한 컴포넌트

    1. load_valid_set :
        - 토크나이저, 전처리 타입, 모델명 등을 받아 검증데이터셋에 전처리 및 토크나이징 진행
    2. evaluation :
        - 검증셋에 대해 해당 모델 정확도 검증

### evaluation_ensemble.py
- 여러 모델을 앙상블할 때, 검증셋에 대해 정확도를 평가하기 위함


### inference.py
- 테스트셋에 예측값을 전달하기 위한 컴포넌트

    1. load_test_set :
        - 토크나이저, 전처리 타입, 모델명 등을 받아 검증데이터셋에 전처리 및 토크나이징 진행
    2. inference :
        - 테스트셋에 대해 예측값을 리턴


### inference_ensemble.py
- 여러 모델을 앙상블할 때, 테스트셋에 대한 예측값을 저장하기 위함
- 모델을 저장해서 사용했었지만, logits 값을 저장한 뒤 soft voting에 이용


### check_submission.py
- submission을 제출하기 전에 분포를 확인하기 위함