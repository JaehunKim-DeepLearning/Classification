## RESULT

### 1. L2 Regulaization 비교 결과 (UTF 인코딩 미적용 기준 모델)
  - L2 0.0001     ---> DEV ACC : 1.009365
  - L2 0.00001    ---> DEV ACC : 1.018614
  - L2 0.000001   ---> DEV ACC : 1.026462 
  - L2 0.0000001  ---> DEV ACC : 1.025966


2. L2 0.000001 각 Data 인코딩 INPUT 결과
  - UTF NO  ---> DEV ACC : 1.026462 
  - UTF 16  ---> DEV ACC : 1.025795
  - UTF 32  ---> DEV ACC : 1.025966


3. Ensemble Result
  - Three Model Ensemble  ---> DEV ACC : 1.029758


4. 각 인코딩의 단어 사전 배치 (size10 기준)
  - product ='하프클럽 뉴욕 시티 볼캡야구모자'
  - UTF NO : [(2, 2), (8, 1), (1, 1)]
  - UTF 16 : [(6, 2), (1, 1), (10, 1)]
  - UTF 32 : [(4, 2), (9, 1), (6, 1)]
  - 각각 사전의 배치가 다른 결과를 보임
