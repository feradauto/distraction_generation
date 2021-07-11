# Importing libraries
import json
import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from configuration import Configuration
from configuration import CONSTANTS as C
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer
from models.model_t5 import T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from rich.table import Column, Table
from rich import box
from rich.console import Console
from tensorboardX import SummaryWriter
import time
import utils.utils as U
from torch import cuda
import glob
import sys

class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and 
    loading it into the dataloader to pass it to the neural network for finetuning the model
    """    
    def __init__(self, dataframe, tokenizer, source_len, target_len,answer_len, source_text, target_text,answer_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.ans_len = answer_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]
        self.answer_text = self.data[answer_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        answer_text = str(self.answer_text[index])
        #cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())
        answer_text = ' '.join(answer_text.split())
        source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        answer = self.tokenizer.batch_encode_plus([answer_text], max_length= self.ans_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        answer_ids = answer['input_ids'].squeeze()
        answer_mask = answer['attention_mask'].squeeze()
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'answer_ids': answer_ids.to(dtype=torch.long),
            'answer_mask': answer_mask.to(dtype=torch.long)
        }


def create_model_dir(experiment_main_dir, experiment_id, model_summary):
    """
    Create a new model directory.
    :param experiment_main_dir: Where all experiments are stored.
    :param experiment_id: The ID of this experiment.
    :param model_summary: A summary string of the model.
    :return: A directory where we can store model logs. Raises an exception if the model directory already exists.
    """
    model_name = "{}-{}".format(experiment_id, model_summary)
    model_dir = os.path.join(experiment_main_dir, model_name)
    if os.path.exists(model_dir):
        raise ValueError("Model directory already exists {}".format(model_dir))
    os.makedirs(model_dir)
    return model_dir

def train(epoch, tokenizer, model, device, loader, optimizer,writer,global_step,records,model_dir,model_params):

    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    c=0
    for _,data in enumerate(loader, 0):
        print("mem",torch.cuda.memory_allocated(device=C.DEVICE))
        c=c+1
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        
        ans_str = data['answer_ids'].to(device, dtype = torch.long)
        ans_mask = data['answer_mask'].to(device, dtype = torch.long)
        
        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids,
                        labels=lm_labels,answer_str=ans_str,answer_mask=ans_mask,tokenizer=tokenizer,c=c,param=model_params['LAMBDA'])
        loss = outputs[0]

        #print("preds",outputs[1])
        #preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs[1]]
        #print(preds)
        #print("ans",outputs["ans_ids"])
        #an = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs[2]]
        #print(an)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss, global_step)
        
        
        ### measure bleu
        if c%200==0:
            model.eval()
            predictions = []
            actuals = []
            num_dist=[]
            ## generate 3 distractors
            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=3,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True,
                num_return_sequences=3,
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)for t in y]
            predictions.extend(preds)
            for tt in target:
                actuals.extend([tt,tt,tt])
                num_dist.extend([1,2,3])

            temp_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals,'Num distractor':num_dist})
            temp_df=temp_df.drop_duplicates()
            val=records.rename(columns={'distractor':'Actual Text'})

            gen_dist=val.merge(temp_df,on=['Actual Text']).loc[:,['text','Generated Text','Num distractor']]
            gen_dist=gen_dist.drop_duplicates()
            distractors=val.groupby(['text']).agg({ 'Actual Text': lambda x: list(x.str.split())}).reset_index()

            dist_compare=distractors.merge(gen_dist,on=['text'])
            dist_compare['Generated Text']=dist_compare['Generated Text'].str.split()
            ## compute bleu scores
            dist_compare=dist_compare.assign(bleu1=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(1, 0, 0, 0),smoothing_function=SmoothingFunction().method1),axis=1))
            dist_compare=dist_compare.assign(bleu2=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(0, 1, 0, 0),smoothing_function=SmoothingFunction().method1),axis=1))
            dist_compare=dist_compare.assign(bleu3=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(0, 0, 1, 0),smoothing_function=SmoothingFunction().method1),axis=1))
            dist_compare=dist_compare.assign(bleu4=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(0, 0, 0, 1),smoothing_function=SmoothingFunction().method1),axis=1))
            
            for i in range(1,4):
                bleu_1=dist_compare.loc[dist_compare['Num distractor']==i].bleu1.mean()
                bleu_2=dist_compare.loc[dist_compare['Num distractor']==i].bleu2.mean()
                bleu_3=dist_compare.loc[dist_compare['Num distractor']==i].bleu3.mean()
                bleu_4=dist_compare.loc[dist_compare['Num distractor']==i].bleu4.mean()
                writer.add_scalar('bleu/distractor_{}/bleu_1'.format(i), bleu_1, global_step)
                writer.add_scalar('bleu/distractor_{}/bleu_2'.format(i), bleu_2, global_step)
                writer.add_scalar('bleu/distractor_{}/bleu_3'.format(i), bleu_3, global_step)
                writer.add_scalar('bleu/distractor_{}/bleu_4'.format(i), bleu_4, global_step)
            
            
            bleu_1=dist_compare.bleu1.mean()
            bleu_2=dist_compare.bleu2.mean()
            bleu_3=dist_compare.bleu3.mean()
            bleu_4=dist_compare.bleu4.mean()
            writer.add_scalar("bleu/distractor_gen/bleu_1", bleu_1, global_step)
            writer.add_scalar("bleu/distractor_gen/bleu_2", bleu_2, global_step)
            writer.add_scalar("bleu/distractor_gen/bleu_3", bleu_3, global_step)
            writer.add_scalar("bleu/distractor_gen/bleu_4", bleu_4, global_step)
            
            if c%1000==0:
                path = os.path.join(model_dir, "model_files")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)

            model.train()
        
        
        global_step += 1
    return global_step


def validate(epoch, tokenizer, model, device, loader,writer,records_test):

    """
    Function to evaluate model for predictions

    """
    global_step = 0
    model.eval()
    predictions = []
    actuals = []
    num_dist=[]
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            ## generate distractors
            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=3,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True,
                num_return_sequences=3,
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)for t in y]
            predictions.extend(preds)
            for tt in target:
                actuals.extend([tt,tt,tt])
                num_dist.extend([1,2,3])

        temp_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals,'Num distractor':num_dist})
        temp_df=temp_df.drop_duplicates()
        val=records_test.rename(columns={'distractor':'Actual Text'})
        
        gen_dist=val.merge(temp_df,on=['Actual Text']).loc[:,['text','Generated Text','Num distractor']]
        gen_dist=gen_dist.drop_duplicates()
        distractors=val.groupby(['text']).agg({ 'Actual Text': lambda x: list(x.str.split())}).reset_index()

        dist_compare=distractors.merge(gen_dist,on=['text'])
        dist_compare=dist_compare.drop_duplicates()
        dist_compare['Generated Text']=dist_compare['Generated Text'].str.split()
        ## compute bleu scores
        dist_compare=dist_compare.assign(bleu1=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(1, 0, 0, 0),smoothing_function=SmoothingFunction().method1),axis=1))
        dist_compare=dist_compare.assign(bleu2=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(0, 1, 0, 0),smoothing_function=SmoothingFunction().method1),axis=1))
        dist_compare=dist_compare.assign(bleu3=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(0, 0, 1, 0),smoothing_function=SmoothingFunction().method1),axis=1))
        dist_compare=dist_compare.assign(bleu4=dist_compare.apply(lambda x:sentence_bleu(x['Actual Text'],x['Generated Text'],weights=(0, 0, 0, 1),smoothing_function=SmoothingFunction().method1),axis=1))

        for i in range(1,4):
            bleu_1=dist_compare.loc[dist_compare['Num distractor']==i].bleu1.mean()
            bleu_2=dist_compare.loc[dist_compare['Num distractor']==i].bleu2.mean()
            bleu_3=dist_compare.loc[dist_compare['Num distractor']==i].bleu3.mean()
            bleu_4=dist_compare.loc[dist_compare['Num distractor']==i].bleu4.mean()
            writer.add_scalar('val/bleu/distractor_{}/bleu_1'.format(i), bleu_1, global_step)
            writer.add_scalar('val/bleu/distractor_{}/bleu_2'.format(i), bleu_2, global_step)
            writer.add_scalar('val/bleu/distractor_{}/bleu_3'.format(i), bleu_3, global_step)
            writer.add_scalar('val/bleu/distractor_{}/bleu_4'.format(i), bleu_4, global_step)


        bleu_1=dist_compare.bleu1.mean()
        bleu_2=dist_compare.bleu2.mean()
        bleu_3=dist_compare.bleu3.mean()
        bleu_4=dist_compare.bleu4.mean()
        writer.add_scalar("val/bleu/distractor_gen/bleu_1", bleu_1, global_step)
        writer.add_scalar("val/bleu/distractor_gen/bleu_2", bleu_2, global_step)
        writer.add_scalar("val/bleu/distractor_gen/bleu_3", bleu_3, global_step)
        writer.add_scalar("val/bleu/distractor_gen/bleu_4", bleu_4, global_step)
            
    return predictions, actuals


def main(config):
    '''
    model_params={
        "MODEL":"t5-small",             # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE":2,          # training batch size
        "VALID_BATCH_SIZE":2,          # validation batch size
        "TRAIN_EPOCHS":10,              # number of training epochs
        "VAL_EPOCHS":1,                # number of validation epochs
        "LEARNING_RATE":1e-4,          # learning rate
        "MAX_SOURCE_TEXT_LENGTH":900,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH":901,   # max length of target text
        "MAX_ANSWER_LENGTH":900,   # max length of answer text
        "SEED": 42                     # set seed for reproducibility 

    }
    '''
    model_params=vars(config)
    
    source_text='text'
    target_text='distractor'
    answer_text='answer_text'
    model_params=model_params

    with open(os.path.join(C.DATA_DIR, "distractor/race_train_original.json"), 'r') as content_file:
        content = content_file.read()
    content=content.replace('\n',',')
    content='['+content[:-1]+']'
    records = json.loads(content)
    records=pd.DataFrame(records)
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"]) # pytorch random seed
    np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True


    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(C.DEVICE)

    ## format the input
    records=records.assign(question=records.question.str.join(' '))
    records=records.assign(distractor=records.distractor.str.join(' '))
    records=records.assign(article=records.article.str.join(' '))
    records=records.assign(answer_text=records.answer_text.str.join(' '))
    records=records.loc[:,['article','question','answer_text','distractor']]
    records=records.assign(text="dist q: "+records.question+" a: "+records.answer_text+" p: "+records.article)
    records=records.loc[:,['text','distractor','answer_text']]

    with open(os.path.join(C.DATA_DIR, "distractor/race_dev_original.json"), 'r') as content_file:
        content = content_file.read()
    content=content.replace('\n',',')
    content='['+content[:-1]+']'
    records_test = json.loads(content)
    records_test=pd.DataFrame(records_test)

    ## format the input
    records_test=records_test.assign(question=records_test.question.str.join(' '))
    records_test=records_test.assign(distractor=records_test.distractor.str.join(' '))
    records_test=records_test.assign(article=records_test.article.str.join(' '))
    records_test=records_test.assign(answer_text=records_test.answer_text.str.join(' '))
    records_test=records_test.loc[:,['article','question','answer_text','distractor']]
    records_test=records_test.assign(text="dist q: "+records_test.question+" a: "+records_test.answer_text+" p: "+records_test.article)
    records_test=records_test.loc[:,['text','distractor','answer_text']]
    #records_test=records_test.loc[:,['text','answer_text']].drop_duplicates()
    #records_test=records_test.assign(distractor='')
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation. 
    val_dataset=records_test
    train_dataset = records


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"],model_params["MAX_ANSWER_LENGTH"], source_text, target_text,answer_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"],model_params["MAX_ANSWER_LENGTH"], source_text, target_text,answer_text)



    # Defining the parameters for creation of dataloaders
    train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


    val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])
    
    # Create Tensorboard logger.
    experiment_id = int(time.time())
    experiment_name = "name"
    model_dir = create_model_dir(os.path.join(C.DATA_DIR, "experiments/"), experiment_id, experiment_name)
        
    ## save code
    code_files = glob.glob('./*.py', recursive=False)
    U.export_code(code_files, os.path.join(model_dir, 'code.zip'))
    config.to_json(os.path.join(model_dir, 'config.json'))

    # Save the command line that was used.
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(cmd)
        
    global_step = 0
    writer = SummaryWriter(os.path.join(model_dir, 'logs'))
    print("mem1",torch.cuda.memory_allocated(device=C.DEVICE))
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        global_step=train(epoch, tokenizer, model, C.DEVICE, training_loader, optimizer,writer,global_step,records,model_dir,model_params)
        print("epoch",epoch)

    #Saving the model after training
    path = os.path.join(model_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


    # evaluating test dataset
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, C.DEVICE, val_loader,writer,records_test)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv(os.path.join(model_dir, 'predictions.csv'),index=False)


if __name__ == '__main__':
    main(Configuration.parse_cmd())