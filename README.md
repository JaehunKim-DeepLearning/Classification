# Classification

## KAKAO Classification 대회
    이번 대회를 통하여 자연어처리를 처음 접하게됨
    
    기존 Baseline Model에서 크게 변화하지 않는선에서 모델의 성능을 향상 시킴

    LOGIC1 FOLDER : AVG, DOT 모델 (제출) ---> Data의 Seed값이 다른 모델을 사용하여 제출(제출 실수)
                                            Seed값이 같은 경우의 결과 : ACC DEV: 1.030391   ACC Validation: 1.068 
                                            Seed값이 다른 경우의 결과 : ACC DEV: 1.026281   ACC Validation: 1.064 - 최종 제출된 모델
    LOGIC1 MODEL DOWNLOAD ADDRESS : 
    
    LOGIC2 FOLDER : AVG Ensemble 모델 (미제출)
                                            ACC DEV : 1.029758  ACC Validation: 1.068
    LOGIC2 MODEL DOWNLOAD ADDRESS : 

## LOGIC1 (AVG, DOT Model)

0. 데이터 생성(각각 데이터의 빈도수도 추가 생성)
    - word (product data 단어 분리) : 야구모자
    - shape_word (형태소 단위 단어, 형태) : 야구모자 ---> (야구, 명사), (모자, 명사)
    - noun_word (형태소 단위 단어) : 야구모자 ---> (야구), (모자)
    - ngram (한글자 단위) : 야구모자 ---> 야, 구, 모, 자
    - jamo1 (단어 단위의 자음모음) : 야구모자 ---> (ㅇㅑ,ㄱㅜ,ㅁㅗ,ㅈㅏ)
    - jamo2 (한글자 단위의 자음모음) : 야구모자 ---> (ㅇㅑ),(ㄱㅜ),(ㅁㅗ),(ㅈㅏ)
    - jamo3 (각 자음모음) : 야구모자 ---> ㅇ,ㅑ,ㄱ,ㅜ,ㅁ,ㅗ,ㅈ,ㅏ
    - bmm (brand+model+maker data 사용)
    - image : Resnet을 통한 image 데이터
    
1. Model Network
    - Input : 각각 생성된 데이터를 사용(빈도수 X) ---> word, shape_word, noun_word, ngram, jamo1, jamo2, jamo3, bmm
    
    - Layer1 Input: 사용 데이터 (word, shape_word, noun_word, ngram) - 빈도수 미포함
    - Layer1 Network: Dropout, BatchNormaization, GlobalAveragePooling1D 사용
    
    - Layer2 Input: 사용 데이터 (word, shape_word, noun_word, ngram, jamo1, jamo2, jamo3, bmm) - 빈도수 포함
    - Layer2 Network : Dot, Flatten, Dropout, BatchNormalization 사용
    
    - 각각 아웃풋된 데이터들과 image 데이터를 Concatenate한 후 Dropout 및 BatchNormalization, Activation(Relu) 적용
    - 이후 Dense Layer(sigmoid)를 사용 하여 예측
    
    
## LOGIC2 (Ensemble Model)

0. 데이터 생성(아래 데이터 생성 및 UTF-16, UTF-32 인코딩을 사용하여 추가 데이터 생성 및 사전 배치)
    - word (단어) : 야구모자
    - shape_word (형태소 단위 단어, 형태) : 야구모자 ---> (야구, 명사), (모자, 명사)
    - noun_word (형태소 단위 단어) : 야구모자 ---> (야구), (모자)
    - ngram (한글자 단위) : 야구모자 ---> 야, 구, 모, 자
    - jamo1 (단어 단위의 자음모음) : 야구모자 ---> (ㅇ,ㅑ,ㄱ,ㅜ,ㅁ,ㅗ,ㅈ,ㅏ)
    - jamo2 (한글자 단위의 자음모음) : 야구모자 ---> (ㅇㅑ),(ㄱㅜ),(ㅁㅗ),(ㅈㅏ)
    - image : Resnet을 통한 image 데이터
    
1. Model Network
    - Input : 각각 생성된 데이터를 사용(빈도수 X) ---> word, shape_word, noun_word, ngram, jamo1, jamo2
    - 임베딩 레이어를 거친 값을 Dropout, BatchNormalization, GlobalAveragePooling1D를 통하여 128의 Word 벡터로 정제 
    - 정제 후 L2 Regualarization 0.000001을 통하여 정규화
    - 정규화된 데이터들과 image 데이터를 Concatenate한 후 Dropout 및 BatchNormalization, Activation(Relu) 적용
    - 이후 Dense Layer(sigmoid)를 사용 하여 예측
    
2. Model Ensemble
    - 기존데이터 모델, UTF-16모델, UTF-32모델 (인풋 데이터를 달리하여 3개의 모델 생성)
    - 생성된 3개의 모델의 예측값의 평균을 산출하여 최종 결과 생성


## 실행 방법

0. 데이터의 위치
    - 내려받은 데이터의 위치는 소스가 실행될 디렉토리의 상위 디렉토리로(../) 가정되어 있습니다.
    - 데이터 위치를 바꾸기 위해서는 각 소스의 상단에 정의된 경로를 바꿔주세요.
1. `python data.py make_db train`
    - 학습에 필요한 데이터셋을 생성합니다. (h5py 포맷) dev, test도 동일한 방식으로 생성할 수 있습니다.
    - 위 명령어를 수행하면 `train` 데이터의 80%는 학습, 20%는 평가로 사용되도록 데이터가 나뉩니다.
    - 이 명령어를 실행하기 전에 `python data.py build_y_vocab`으로 데이터 생성이 필요한데, 코드 레파지토리에 생성한 파일이 포함되어 다시 만들지 않아도 됩니다. 
      - Python 2는 `y_vocab.cPickle` 파일을 사용하고, Python 3은 `y_vocab.py3.cPickle` 파일을 사용합니다.
    - `config.json` 파일에 동시에 처리할 프로세스 수를 `num_workers`로 조절할 수 있습니다.
2. `python classifier.py train ./data/train ./model/train`
    - `./data/train`에 생성한 데이터셋으로 학습을 진행합니다.
    - 완성된 모델은 `./model/train`에 위치합니다.
3. `python classifier.py predict ./data/train ./model/train ./data/train/ dev predict.tsv`
    - 단계 1. 에서 `train`의 20%로 생성한 평가 데이터에 대해서 예측한 결과를 `predict.tsv`에 저장합니다.
4. `python evaluate.py evaluate predict.tsv ./data/train/data.h5py dev ./data/y_vocab.cPickle`
    - 예측한 결과에 대해 스코어를 계산합니다.
    - Python 3에서는 `y_vocab.cPickle` 파일 대신 `y_vocab.py3.cPickle` 파일을 사용하여야 합니다.


## 제출하기
1. `python data.py make_db test ./data/test --train_ratio=0.0`
    - `dev` 데이터셋 전체를 예측용도로 사용하기 위해 `train_ratio` 값을 0.0으로 설정합니다.
2. `python classifier.py predict ./data/test ./model/train ./data/test/ dev Model.predict.tsv`
    - 위 실행 방법에서 생성한 모델로 `test` 데이터셋에 대한 예측 결과를 생성합니다.
3. 제출
    - Model.predict.tsv 파일을 zip으로 압축한 후 카카오 아레나 홈페이지에 제출합니다.
    
    
## 기타
- 코드 베이스를 실행하기 위해서는 데이터셋을 포함해 최소 450G 가량의 공간이 필요합니다.

## 테스트 가이드라인
학습데이터의 크기가 100GB 이상이므로 사용하는 장비에 따라서 설정 변경이 필요합니다. `config.json`에서 수정 가능한 설정 중에서 아래 항목들이 장비의 사양에 민감하게 영향을 받습니다.

    - train_data_list
    - chunk_size
    - num_workers
    - num_predict_workers


`train_data_list`는 학습에 사용할 데이터 목록입니다. 전체 9개의 파일이며, 만약 9개의 파일을 모두 사용하여 학습하기 어려운 경우는 이 파일 수를 줄일 경우 시간을 상당히 단축시킬 수 있습니다. 

`chunk_size`는 전처리 단계에서 저장하는 중간 파일의 사이즈에 영향을 줍니다. Out of Memory와 같은 에러가 날 경우 이 옵션을 줄일 경우 해소될 수 있습니다.

`num_workers`는 전처리 수행 시간과 관련이 있습니다. 장비의 코어수에 적합하게 수정하면 수행시간을 줄이는데 도움이 됩니다.

`num_predict_workers`는 예측 수행 시간과 관련이 있습니다. `num_workers`와 마찬가지로 장비의 코어수에 맞춰 적절히 수정하면 수행시간을 단축하는데 도움이 됩니다.
