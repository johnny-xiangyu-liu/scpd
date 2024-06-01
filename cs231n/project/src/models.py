import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

import math

from transformers import AutoTokenizer, BertModel
#from transformers.models.bert import BertEmbeddings

import constants
from itertools import chain

# # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

class TextEncoder(nn.Module):
    
    @torch.no_grad()
    def __init__(self, tokenizer):
        super().__init__()
        self.model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        self.model.resize_token_embeddings(len(tokenizer))

        
    @torch.no_grad()
    def forward(self, x):
        """
        x: tokenizer output
        """
        out = self.model(**x)
        return out.pooler_output
        
    


class MaskRNNBackbone(nn.Module):
        
    @torch.no_grad()
    def __init__(self):
        super().__init__()

        # micmicking the maskrnn behavior in https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/models/detection/generalized_rcnn.py#L46-L103
        # use maskrnn backbone as the pretrained image backbone
        maskrnn = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        
        self.preprocess_transforms = weights.transforms()
        # always set it to eval so that it's frozen.
        self.transform = maskrnn.transform.eval()       
        self.image_backbone = maskrnn.backbone.eval()
        
        
    @torch.no_grad()
    def forward(self, x):
        """
        x: (N, C, H, W) a batch of images.
        """
        images, _ = self.transform(self.preprocess_transforms(x), None)
        features = self.image_backbone(images.tensors)
        return features


def flatten(results):
    ids = []
    images = []
    captions = []
    qas = []
    for (image_id, tensor, c, qa) in results:
        ids.append(image_id)
        images.append(tensor)
        captions.append(c)
        qas.append(qa)
    return ids, torch.stack(images, dim=0), captions, qas

def blow_to(images, replicas):
    return images[replicas]

    

class VQANet(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.image_backbone = MaskRNNBackbone()
        self.text_encoder = TextEncoder(tokenizer)
        self.embedding = self.text_encoder.model.embeddings
        text_embedding_size = self.text_encoder.model.config.hidden_size
        
        hidden_size = 2048
        self.proj_to_text_embedding = nn.Sequential(
            nn.Linear(constants.IMAGE_BACKBONE_POOL_DIM, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, text_embedding_size),
            nn.LeakyReLU()
        )
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=text_embedding_size, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output = nn.Linear(text_embedding_size, len(tokenizer))
        
    def forward(self, x, device):
        """
        returns:
         image embedding: shape of (N, 768)
         captions embeddings: shape of (M, 768) where M = sum( # of captions of image for image in images)
         output_logits: (seq, K, WORD_SIZE) where K = sum(# of qas of image for image in images),
                         seq: the padded sequence length output by the tokenizer
                         WORD_SIZE: all the worlds that the tokenizer knows about (constant). (~30k)
        """
        
        ids = x['image_ids']
        images = torch.stack(x['images'], dim = 0).to(device)
        cap_tokens = x['captions']
        qa_input_ids = x['qa']
        c2i = x['c2i']
        qa2i = x['qa2i']
        N = images.shape[0]

        feature_map = self.image_backbone(images)
        #print("feature_map", feature_map.keys())
        image_embedding = self.get_image_embed(feature_map)
        
        captions_embedding = None
        image_embedding_for_captions = None
        if cap_tokens is not None:
            captions_embedding = self.text_encoder(cap_tokens)
#            print("captions_embedding", captions_embedding.shape)
            image_embedding_for_captions = blow_to(image_embedding, c2i)
#            print("image_embedding_for_captions", image_embedding_for_captions.shape)


        out_logits = None
        if qa_input_ids is not None:
            print("qa_input_ids", qa_input_ids)
            qa_embed = self.embedding(input_ids = qa_input_ids)

            # (seq, batch, embedding)
            qa_embed = qa_embed.transpose(0,1)
#            print("qa_embed", qa_embed.shape)

            image_embed_for_qa = blow_to(image_embedding, qa2i)
#            print("image_embed_for_qa", image_embed_for_qa.shape)

            blown_image_embed_for_qa = torch.broadcast_to(image_embed_for_qa, qa_embed.shape)
#            print("blown_image_embed_for_qa", blown_image_embed_for_qa.shape)

            qa_mask = nn.Transformer.generate_square_subsequent_mask(qa_embed.shape[0]).to(device)

#            print("qa_mask", qa_mask.shape)
            output_embedding = self.decoder(tgt = qa_embed, memory = blown_image_embed_for_qa, tgt_mask=qa_mask )
#            print("output_embedding", output_embedding.shape)
            out_logits = self.output(output_embedding)
        
        return image_embedding_for_captions, captions_embedding, out_logits
    
    def get_image_embed(self, backbone_output):
        pool = backbone_output['pool']
        N = pool.shape[0]
        pool = pool.view(N, -1)
        return self.proj_to_text_embedding(pool)

    def parameters(self):
        tunable = [self.proj_to_text_embedding, self.decoder, self.output]
        return chain.from_iterable([m.parameters() for m in tunable])
    
    
    def answer(self, x, device, max_length=30):
        """
        """
        with torch.no_grad():
            for t in range(max_length):
                print(f">>>>>{t}")
                print(x)
                indices = (x["qa"] == 102).nonzero() # 102 is the [SEP] token in bert.
                last_word_indices = indices -torch.tensor([0, 1]) # get the token right before 102

                print("indices", indices)
                # Predict the next token (ignoring all other time steps).
                image_embedding_for_captions, captions_embedding, output_logits = self.forward(x, device)
                # (seq, K, WORD_SIZE) - > (K, seq, WORD_SIZE)
                output_logits = output_logits.transpose(0,1)
                print("output_logits", output_logits.shape)
                # Choose the most likely word ID from the vocabulary.
                word = torch.argmax(output_logits, axis=2)
                print("word", word.shape, word)
                word = word[last_word_indices[:,0], last_word_indices[:, 1]]
                
                print("after word", word.shape, word)
                # (k, seq)
                qa = x["qa"]
                # (k, seq) -> (k, seq+1)
                new_qa = torch.cat((qa, torch.zeros((qa.shape[0], 1), dtype=torch.int64)), dim = 1)
                print("new_qa, 1", new_qa)
                # put the predicted word at the indices
                new_qa = new_qa.index_put(tuple(indices.t()), word) 
                print("new_qa, 2", new_qa)
                # move the indices by 1
                next_indices = indices + torch.tensor([0, 1])
                # put the "102", i.e. [SEP] after the predicted word
                new_qa = new_qa.index_put(tuple(next_indices.t()),
                                          102 * torch.ones(indices.shape[0], dtype=torch.int64))
                print("new_qa",new_qa)
                
                # put the qa back.
                x["qa"] = new_qa

            return x
    
    
    
    
    
    
    
    
    
    
    
    
    