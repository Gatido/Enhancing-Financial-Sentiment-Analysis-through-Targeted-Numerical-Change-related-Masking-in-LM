import os
import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

from custom_data_collator import DataCollatorForWholeWordMask as DataCollatorForNM

company_filepath = os.path.join('data', 'LSTWORD.csv') # LSTWORD.csv 는 numerical change-related word list

def train(tokenizer, model, dataset, save_dir, method_name='NM'):
    print("Method Name", method_name)
    mlm_prob=0.15
    model.train()
    
    training_args = TrainingArguments(
        output_dir = save_dir,
        num_train_epochs = 5,   
        per_device_train_batch_size = 8,
        save_steps = 1000,
        save_total_limit = 1,
    )
    
    if method_name == 'NM':
        masking_list = [item.lower() for item in list(pd.read_csv(os.path.join('data', 'LSTWORD.csv')).Name.unique())]
        print('The Number of numerical change-related words', len(masking_list))
        data_collator = DataCollatorForNM(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob, masking_list=masking_list)
    
    elif method_name == 'SM':
        print('Subword Masking')
        data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = mlm_prob)

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
    )
    
    print('Post-training...')
    trainer.train()
    
    tokenizer.save_pretrained(save_dir)
    trainer.save_model(save_dir)
    print('Post-training saved trained model at {}'.format(save_dir))