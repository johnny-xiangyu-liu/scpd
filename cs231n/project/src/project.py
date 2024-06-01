#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision as tv
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import tv_tensors  # we'll describe this a bit later, bare with us

import torchvision.datasets as datasets
from pathlib import Path

from torchview import draw_graph

import constants
import dataset
import util
import json
import pandas as pd
import models 
from models import VQANet
import matplotlib.pyplot as plt
import numpy as np
import time

from transformers import AutoTokenizer
import traceback
import gc

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():      
    device = 'mps'                         
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)
    



# In[2]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[3]:


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# In[4]:


# with open(constants.CAPTION_TRAIN, 'r') as f:
#     data = json.load(f)
#     print(data.keys())
#     print(data["annotations"][0])

# with open(constants.VQA_OPEN_ENDED_QUESTION_TRAIN, 'r') as f:
#     data = json.load(f)
#     print(data.keys())
#     print(data["questions"][0])

# with open(constants.VQA_OPEN_ENDED_ANSWER_TRAIN, 'r') as f:
#     data = json.load(f)
#     print(data.keys())
#     print(data["annotations"][0])
    
# with open(constants.CAPTION_VAL, 'r') as f:
#     data = json.load(f)
#     print(data.keys())

# with open(constants.VQA_OPEN_ENDED_QUESTION_VAL, 'r') as f:
#     data = json.load(f)
#     print(data.keys())

# with open(constants.VQA_OPEN_ENDED_ANSWER_VAL, 'r') as f:
#     data = json.load(f)
#     print(data.keys())

#dataset.load(constants.VQA_OPEN_ENDED_QUESTION_TRAIN, ['image_id', 'id', 'caption'])


# In[5]:


train = dataset.Coco()
# val = dataset.Coco("validation")
# test = dataset.Coco("test")


# In[6]:


print(len(train))
print(len(train.captions))
#img = train.__getitem__(1)


# In[7]:


# print(img)
# print(img.image_id)
# print(img.image_path)

# print(">>>>")
# print(img.captions())

# print(">>>>")
# print(img.qa())

# show([img.image_tensor()])

# #plt.imshow(  img.image_tensor().permute(1, 2, 0)  )


# In[8]:


tokenizer  = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# Add the Q and A token as special token
tokenizer.add_special_tokens(constants.QA_TOKEN_DICT)


# In[9]:


def annotate_qa(q, a):
    return constants.QUESTION_TOKEN + ' ' + q.lower() + ' ' + constants.ANSWER_TOKEN + ' ' + a.lower() + ' ' + constants.END_TOKEN
    
def collate_fn2(batch):
    result = {}
    
    result['image_ids'] = []
    result['images'] = []
    raw_captions = []  # plain text 
    raw_qa = []   # plain text
    result['c2i'] = [] # index for images for a given caption. same len as 'caption'
    result['qa2i'] = [] # index of corresponding image for a given qa. same len as 'qa'
    target  = [] # the corresponding target for the qa.
    result['images']
    for idx, data in enumerate(batch):
        result['image_ids'].append(data.image_id)
        result['images'].append(data.image_tensor())
        caption_list = data.captions()
        if caption_list is not None:
            raw_captions += caption_list
            for c in range(len(caption_list)):
                result['c2i'].append(idx)
        
        qa_list = data.qa()
        if qa_list is not None:
            raw_qa += qa_list
            for c in range(len(qa_list)):
                result['qa2i'].append(idx)
    #print("raw_cap", len(raw_captions))
    #print("raw_qa", len(raw_qa))
    
    result['raw_cap'] = raw_captions
    result['captions'] = None if len(raw_captions) == 0 else \
                                tokenizer(raw_captions, padding=True , return_tensors="pt").to(device)
    result['raw_qa'] = raw_qa
    if len(raw_qa) != 0:
        result['qa'] =  tokenizer(raw_qa, padding=True , return_tensors="pt")['input_ids'].to(device)
        end_padding = torch.broadcast_to(torch.zeros(1), (result['qa'].shape[0], 1)).to(device)
        #print(end_padding.shape)
        # return a shape {seq, batch}
        target = torch.column_stack((result['qa'][:, 1:], end_padding)).transpose(0, 1)
    else:
        result['qa'] = None
        target = None
    return result, target


# In[10]:


from torch.utils.data import DataLoader
batch_size = 2
#fn = collate_fn
fn = collate_fn2 
shuffle = False  # True
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, collate_fn=fn)
#val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=shuffle, collate_fn=fn)
#test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, collate_fn=fn)


# In[11]:


#x = next(iter(train_dataloader))


# In[12]:


ce_fn = nn.CrossEntropyLoss( reduction='none')
cos_fn = nn.CosineSimilarity(dim=1)
model = VQANet(tokenizer).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


# In[13]:


# out = model(result, device)
# image_embedding, captions_embeddings, output_logits = out
# print(captions_embeddings.shape)
# a = output_logits.reshape(-1, len(tokenizer))
# b = target.reshape(-1)
# print(a.shape)
# print(b.shape)
# ce_loss = ce_fn(a, b)
# print(ce_loss.shape)
# N = len(result['images'])
# M = len(result['qa2i'])
# ce = ce_loss.reshape(-1, M).transpose(0, 1)
# print(ce.shape)
# print(ce)
# per_qa  = torch.mean(ce, axis = 1)
# print(per_qa.shape)


# In[ ]:





# In[14]:


def cal_average(size, blown_loss, replicas):
    result= torch.zeros(size).to(device)
    counts = torch.zeros(size).to(device)
    for index, val in enumerate(replicas):
        result[val] += blown_loss[index]
        counts[val] += 1
        
    for index in range(size):
        if counts[index] == 0:
            counts[index] = 1  # so that result / counts still makes sense.
    #print("result", result)
    #print("counts:", counts)
    result /= counts
    return result


# In[15]:


# blown = models.blow_to(image_embedding, result['c2i'])
# print(image_embedding.shape)
# print(image_embedding)
# print(blown.shape)
# print(blown)
# print("captions_embedding:", captions_embeddings.shape)
# print(result['c2i'])


# In[16]:


# print(blown)
# print(captions_embeddings)
# cos= nn.CosineSimilarity(dim = 0)
# print(cos(blown[1], captions_embeddings[1]))

# per_caption_loss = cos_fn(blown, captions_embeddings)
# print(per_caption_loss)
# per_image_caption_loss = cal_average(len(result['images']), per_caption_loss, result['c2i'])
# print(per_image_caption_loss.shape)

# print(per_image_caption_loss)


# In[17]:


# hyperparam to tune the caption loss v.s. qa loss
gamma = 0.9
DEBUG = False
def do_train(idx, x, target):
        N = len(x['images'])
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        if DEBUG:
            image_embedding, captions_embedding, output_logits = None, None, None
        else:
            image_embedding, captions_embedding, output_logits  = model(x, device)

#        image_embedding, captions_embedding, output_logits = None, None, None
        per_image_qa_loss = torch.zeros(N).to(device)
        per_image_caption_loss = torch.zeros(N).to(device)
        
        if output_logits is not None:
            a = output_logits.reshape(-1, len(tokenizer))
            b = target.reshape(-1)

            K = len(x['qa2i'])
            # back to (K, seq)
            qa_loss = ce_fn(a, b).reshape(-1, K).transpose(0, 1)
            #print("qa_loss", qa_loss.shape)
            # qa loss, shape of (K) (different images can have diff counts of qas)
            per_qa_loss = torch.mean(qa_loss, axis = 1)

            # per image qa loss, shape of (N)
            per_image_qa_loss = cal_average(N, per_qa_loss, x['qa2i'])
            #print("per_qa_loss", per_qa_loss.shape)
            #print("per_image_qa_loss", per_image_qa_loss.shape)

        if captions_embedding is not None:
            blown_embedding = models.blow_to(image_embedding, x['c2i'])
            # loss per caption, shape of (M) (different images can have diff counts of captions)
            per_caption_loss = cos_fn(blown_embedding, captions_embedding)
            # per image loss on the caption scale. shape of (N)
            per_image_caption_loss = cal_average(N, per_caption_loss, x['c2i'])

        #print("per_caption_loss", per_caption_loss.shape)
        #print("per_image_caption_loss", per_image_caption_loss.shape)
        total_loss = gamma * per_image_caption_loss + per_image_qa_loss

        loss = torch.sum(total_loss) / N
        #print("total_loss", total_loss.shape)
        if not DEBUG:
            print("backwards")
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


        del per_image_caption_loss
        del per_image_qa_loss
        del x
        del loss
            
def training(early_terminate = None, empty_catch_after_every_n = 200, gc_every = 20, print_every = 100):
    start_time = time.time()
    model.train()
    for idx, (x, target) in enumerate(train_dataloader):
        if (idx + 1) % print_every == 0:
            print(">>>> Batch # ", idx,  x['image_ids'] )

        try:
            do_train(idx, x, target)
        except Exception as e:
            print(">>>> FAILED! Batch # ", idx,  x['image_ids'])
            traceback.print_exc()
            return;

        if empty_catch_after_every_n is not None and (idx + 1) % empty_catch_after_every_n == 0:
            print(">>>empty torch mps cache")
            torch.mps.empty_cache()
        if early_terminate is not None:
            if idx > early_terminate:
                print("early terminating.", early_terminate)
                break;
        if gc_every is not None and (idx + 1) % gc_every == 0:
                print("explictly calling GC:")
                gc.collect()
        print("--- %s Per batch time ---" % (time.time() - start_time))

    print("---DONE: %s seconds ---" % (time.time() - start_time))

    


# In[18]:


# from torch.profiler import profile, record_function, ProfilerActivity

# try:
#     torch.mps.profiler.start(wait_until_completed=True)
#     training(5, empty_catch_after_every_n=5);
#     torch.mps.profiler.stop()

# except Exception as e:
#     print(f"Failed: {e}")


# In[19]:
training()
# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("training"):
#         training(empty_catch_after_every_n=None, early_terminate=50)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

