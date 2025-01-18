import huggingface_hub
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
huggingface_hub.login(os.getenv('HUGGINGFACE_TOKEN'))

import pandas as pd
from pathlib import Path
import os
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor, AutoProcessor
from transformers import BlipConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datasets import load_dataset

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, data, segment, text_processor, image_processor):
        self.data = data
        self.questions = data['question']
        self.answers = data['answer']
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_length = 32
        self.image_height = 128
        self.image_width = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get image + text
        answers = self.answers[idx]
        questions = self.questions[idx]
        image = self.data[idx]['image'].convert('RGB')
        text = self.questions[idx]

        image_encoding = self.image_processor(image,
                                  do_resize=True,
                                  size=(self.image_height,self.image_width),
                                  return_tensors="pt")

        encoding = self.text_processor(
                                  None,
                                  text,
                                  padding="max_length",
                                  truncation=True,
                                  max_length = self.max_length,
                                  return_tensors="pt"
                                  )
        # # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        encoding["pixel_values"] = image_encoding["pixel_values"][0]
        # # add labels
        labels = self.text_processor.tokenizer.encode(
            answers,
            max_length= self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )[0]
        encoding["labels"] = labels

        return encoding
    
version = 0
out_dir = '/N/scratch/tnn3/BTL/AI/DucDo/output/result'
while True:
    if os.path.isdir(out_dir + f'_v{version}'):
        version += 1
    else:
        # version = 4
        out_dir = out_dir + f'_v{version}'
        break
Path(os.path.join(out_dir, 'model_save')).mkdir(parents=True, exist_ok=True)

# Load dataset từ Hugging Face
dataset = load_dataset("flaviagiammarino/path-vqa", cache_dir='./cache/dataset')
config = BlipConfig.from_pretrained("Salesforce/blip-vqa-base")

checkpoint = "Salesforce/blip-vqa-base"
text_processor = BlipProcessor.from_pretrained(checkpoint, cache_dir=f'./cache/processor/{checkpoint}')
image_processor = BlipImageProcessor.from_pretrained(checkpoint, cache_dir=f'./cache/Iprocessor/{checkpoint}')
print("Checkpoint loaded")

def filter_yes_no(sample):
    return sample['answer'].lower() in ['yes', 'no']
train_data = dataset['train'].filter(filter_yes_no)
val_data = dataset['validation'].filter(filter_yes_no)
test_data = dataset['test'].filter(filter_yes_no)
print("Data filtered")

train_vqa_dataset = VQADataset(data=train_data,
                               segment='train',
                               text_processor = text_processor,
                               image_processor = image_processor)

val_vqa_dataset = VQADataset(data=val_data,
                             segment='validation',
                             text_processor = text_processor,
                             image_processor = image_processor)

test_vqa_dataset = VQADataset(data=test_data,
                             segment='test',
                             text_processor = text_processor,
                             image_processor = image_processor)

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['pixel_values'] = torch.stack(pixel_values)
    batch['labels'] = torch.stack(labels)

    return batch

train_dataloader = DataLoader(train_vqa_dataset,
                              collate_fn=collate_fn,
                              batch_size=32,
                              shuffle=True)
val_dataloader = DataLoader(val_vqa_dataset,
                            collate_fn=collate_fn,
                            batch_size=32,
                            shuffle=False)
test_dataloader = DataLoader(test_vqa_dataset,
                            collate_fn=collate_fn,
                            batch_size=32,
                            shuffle=False)



batch = next(iter(train_dataloader))

model = BlipForQuestionAnswering.from_pretrained(checkpoint, cache_dir=f'./cache/model/{checkpoint}')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
image_mean = image_processor.image_mean
image_std = image_processor.image_std


batch_idx = 1

unnormalized_image = (batch["pixel_values"][batch_idx].cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)

log = {
        'train_loss': [],
        'val_loss': [],
    }

mode = 'train'

if mode == 'test':
    model.load_state_dict(torch.load(os.path.join(out_dir, 'model_save', f'model_{30}.pth')))

if mode == 'train':
    for epoch in range(30):
        print(f"Epoch: {epoch}")
        model.train()
        total_loss = []
        for batch in tqdm(train_dataloader):
            # get the inputs;
            batch = {k:v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        total_loss = sum(total_loss)
        print("Train Loss:", total_loss)
        log['train_loss'].append(total_loss)
        
        total_loss = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader):
                # get the inputs;
                batch = {k:v.to(device) for k,v in batch.items()}

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(**batch)
                loss = outputs.loss
                total_loss.append(loss.item())
        
        total_loss = sum(total_loss)
        print("Val Loss:", total_loss)
        log['val_loss'].append(total_loss)
        
        
        df = pd.DataFrame(log)
        df.to_excel(os.path.join(out_dir, 'train_log.xlsx'))
        
        torch.save(model.state_dict(), os.path.join(out_dir, 'model_save', f'model_{epoch}.pth'))

df = []
for x in range(test_vqa_dataset.__len__()):
    sample = test_vqa_dataset[x]
    arr = []
    arr.append(text_processor.decode(sample['input_ids'], skip_special_tokens=True))
    sample = {k: v.unsqueeze(0).to(device) for k,v in sample.items()}

    # forward pass
    outputs = model.generate(pixel_values=sample['pixel_values'],
                            input_ids=sample['input_ids'])
    arr.append(text_processor.decode(outputs[0],skip_special_tokens=True))
    arr.append(text_processor.decode(sample['labels'][0], skip_special_tokens=True))
    #########################################################################
    # unnormalized_image = (sample["pixel_values"][0].cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
    # unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    # unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    # display(Image.fromarray(unnormalized_image))
    #########################################################################
    df.append(arr)
    
df = pd.DataFrame(df, columns=['Question', 'Predicted', 'Actual'])
df.to_excel(os.path.join(out_dir, 'test_pred.xlsx'))