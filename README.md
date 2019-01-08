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
    - shape_word (형태소 단위 단어, 형태) : 야구모자 ---> (야구, 명사), (모자, 명사) :::Mecab형태소 분석기 사용
    - noun_word (형태소 단위 단어) : 야구모자 ---> (야구), (모자) :::Mecab형태소 분석기 사용
    - ngram (한글자 단위) : 야구모자 ---> 야, 구, 모, 자
    - jamo1 (단어 단위의 자음모음) : 야구모자 ---> (ㅇㅑ,ㄱㅜ,ㅁㅗ,ㅈㅏ)
    - jamo2 (한글자 단위의 자음모음) : 야구모자 ---> (ㅇㅑ),(ㄱㅜ),(ㅁㅗ),(ㅈㅏ)
    - jamo3 (각 자음모음) : 야구모자 ---> ㅇ,ㅑ,ㄱ,ㅜ,ㅁ,ㅗ,ㅈ,ㅏ
    - bmm (brand+model+maker data 사용)
    - image : Resnet을 통한 image 데이터
    
1. Model Network
    - Input : 각각 생성된 데이터를 임베딩(빈도수 X) ---> `word, shape_word, noun_word, ngram, jamo1, jamo2, jamo3, bmm`
    - Layer1 Input: 사용 데이터 (`word, shape_word, noun_word, ngram`) - 빈도수 미포함
    - Layer1 Network: `Dropout, BatchNormaization, GlobalAveragePooling1D`
    - Layer2 Input: 사용 데이터 (`word, shape_word, noun_word, ngram, jamo1, jamo2, jamo3, bmm`) - 빈도수 포함
    - Layer2 Network : `Dot, Flatten, Dropout, BatchNormalization`
    - 각각 아웃풋된 데이터들과 `image` 데이터를 `Concatenate`한 후 `Dropout` 및 `BatchNormalization, Activation(Relu)` 적용
    - 이후 `Dense Layer(sigmoid)`를 사용 하여 예측
    
    
## LOGIC2 (Ensemble Model)

0. 데이터 생성(아래 데이터 생성 및 `UTF-16`, `UTF-32` 인코딩을 사용하여 추가 데이터 생성 및 사전 배치)
    - word (단어) : 야구모자
    - shape_word (형태소 단위 단어, 형태) : 야구모자 ---> (야구, 명사), (모자, 명사)  :::Mecab형태소 분석기 사용
    - noun_word (형태소 단위 단어) : 야구모자 ---> (야구), (모자)  :::Mecab형태소 분석기 사용
    - ngram (한글자 단위) : 야구모자 ---> 야, 구, 모, 자
    - jamo1 (단어 단위의 자음모음) : 야구모자 ---> (ㅇ,ㅑ,ㄱ,ㅜ,ㅁ,ㅗ,ㅈ,ㅏ)
    - jamo2 (한글자 단위의 자음모음) : 야구모자 ---> (ㅇㅑ),(ㄱㅜ),(ㅁㅗ),(ㅈㅏ)
    - image : Resnet을 통한 image 데이터
    
1. Model Network
    - Input : 각각 생성된 데이터를 임베딩(빈도수 X) ---> `word, shape_word, noun_word, ngram, jamo1, jamo2`
    - 임베딩된 값을 `Dropout, BatchNormalization, GlobalAveragePooling1D`를 사용
    - 정제 후 `L2 Regualarization 0.000001`을 통하여 정규화
    - 정규화된 데이터들과 `image` 데이터를 `Concatenate`한 후 `Dropout 및 BatchNormalization, Activation(Relu)` 적용
    - 이후 `Dense Layer(sigmoid)`를 사용 하여 예측
    
2. Model Ensemble
    - `기존데이터 Input 모델, UTF-16 Input 모델, UTF-32 Input 모델` (인풋 데이터를 달리하여 3개의 모델 생성)
    - 생성된 3개의 모델의 예측값의 평균을 산출하여 최종 결과 생성


## 실행 방법

0. 데이터 위치
    - 데이터의 위치는 소스가 실행될 디렉토리의 상위 디렉토리로(../) 가정되어.
    - 데이터 위치를 바꾸기 위해서는 각 소스의 상단에 정의된 경로를 바꿔주세요.
1. 학습 데이터 생성(train 80%, validation 20%)
    - `python data.py make_db train`(동시에 처리할 프로세스 수를 `config.json`파일의 `num_workers`로 조절할 수 있습니다.)
2. 학습 진행 및 모델 생성
    - `python classifier.py train ./data/train ./model/train`(`./data/train`:데이터 위치, `./model/train`:생성모델 위치)
3. 테스트 데이터 생성 (홈페이지 최종 제출용)
    - `python data.py make_db test ./data/test --train_ratio=0.0` 
4. 생성된 모델을 활용한 테스트 데이터 에측 (홈페이지 최종제출용 - zip압축 후 홈페이지 제출)
    - `python classifier.py predict ./data/test ./model/train ./data/test/ dev Model.predict.tsv`
5. 생성된 모델을 활용한 Validation 데이터 예측
    - `python classifier.py predict ./data/train ./model/train ./data/train/ dev predict.tsv` 
6. Validation 데이터 평가 및 스코어 계산
    - `python evaluate.py evaluate predict.tsv ./data/train/data.h5py dev ./data/y_vocab.cPickle`
    - 예측한 결과에 대해 스코어를 계산합니다.
    - Python 3에서는 `y_vocab.cPickle` 파일 대신 `y_vocab.py3.cPickle` 파일을 사용하여야 합니다.

    
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
