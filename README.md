# Enhancing Financial Sentiment Analysis through Targeted Numerical Change reltaed Masking in LM
paper

## 작성 기간
2023. 8.20 ~ 2023.11.28

# About dataset
## Here, it's a matter of capacity, but if you put the news articles you've collected into a name like this, you can run the code
+ concatenated_dataset_1003_raw.csv : Collected raw data
+ concatenated_dataset_1003_v6.csv : Data from concatenated_dataset_1003_raw.csv with simple preprocessing
+ filtered_df_small_ver1106.csv : Data extracted from concatenated_dataset_1003_v6.csv to only sentences containing words in LSTWORD.csv
+ data_postTraining/custom_postdata.txt : Data extracted from filtered_df_small_ver1106.csv with the appropriate number of sentences for the experimental environment, which is directly used for post-training.
