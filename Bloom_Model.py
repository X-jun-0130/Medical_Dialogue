# deepspeed --master_addr 172.16.0.126 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  ./Model_Bloom.py
import os
os.chdir('/Nlp_2023/Dialogue_Bloom/')

import torch
import json
import numpy as np
from torch.utils.data import random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM


model_name = "/workspace/Xuxiangjun/Model_TH/Bloom_3B/"
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
training_args = TrainingArguments(output_dir='./results', 
                                 overwrite_output_dir=True,
                                 num_train_epochs=2.5, 
                                 logging_steps=200, 
                                 save_strategy='steps',
                                 save_steps = 5000,
                                 evaluation_strategy = 'epoch',
                                 per_device_train_batch_size=8, 
                                 per_device_eval_batch_size=8, 
                                 lr_scheduler_type="linear",
                                 warmup_steps=200,
                                 weight_decay=0.01,
                                 fp16=True,
                                 deepspeed='./ds_config.json')
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
# model.resize_token_embeddings(len(tokenizer))


kg_list = json.load(open('./data/dia_data.json', 'r', encoding='utf-8')) 

kg_dataset = [['<s>' + key +'</s>' , j] for j,key in enumerate(kg_list)][:160000]


dataset = []
for line in kg_dataset:
    k = tokenizer.encode(line[0])
    if len(k) <= 512:
        dataset.append(line)

print(dataset[0:2])
train_size = int(0.99 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
print(len(kg_list), len(train_dataset))


'''
model training
'''
def the_collate_fn(batch):  
    r = tokenizer([b[0] for b in batch], padding=True)
    input_ids = torch.LongTensor(r['input_ids'])
    attention_mask = torch.LongTensor(r['attention_mask'])
    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':input_ids}



class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

trainer = Mytrainer(model=model, args=training_args, train_dataset=train_dataset,eval_dataset=val_dataset, data_collator=the_collate_fn)
trainer.train()
