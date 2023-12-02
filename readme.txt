코드 진행 설명

1. data_preprocessing.ipynb
병합된 데이터(concatenated_dataset_1003_raw.csv)로 전처리 -> concatenated_dataset_1003_v6.csv

2. similarity_method.ipynb
BERT 모델 기반 increase, decrease와 유사도 top 100단어 각각 추출, 마스킹 리스트인 LSTWORD.csv는 이것을 이용해서 따로 만듦, concatenated_dataset_1003_v6.csv을 불러와 LSTWORD.csv 안의 단어들을 포함하는 문장만 추출 -> filtered_df_small_ver1106.csv

3. revised_news_and_EDA.ipynb
중복 제거 등 전처리 조금 더 수행 및 분포 확인, EDA 및 학습에 필요한 문장 수만 추출(10만개, 20만개, ...), 추출된 문장.txt 파일은 data_postTraining의 custom_postdata.txt로 하나씩 이름 바꿔가면서 학습 진행

4. further_pre-train_models_Num.ipynb
증감 관련 문장들로 제안한 방법으로의 post-training 시작, 모델은 따로 허깅페이스에 저장

5. Fine_tuning_sentiment_analysis_textmining.ipynb
허깅페이스에 저장한 모델을 가져와 fine-tuning해 성능 비교

데이터셋

1. LSTWORD_finbert
Araci의 FinBERT 모델을 대상으로 similarity_method.ipynb 을 실행시킨 것으로, BERT 모델과 비교했을 때 증감 관련 단어가 잘 추출되지 않은 모습을 직관적으로 알 수 있음.

2. result_incorporated.csv
실험 1에서의 전체 fine-tuning 결과표로, result_visualization.ipynb에 쓰임. 실험2에 대한 결과데이터는 따로 첨부 하지 않음.