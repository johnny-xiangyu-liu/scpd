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

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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

def make_qa(q, a):
    return constants.QUESTION_TOKEN +' ' + q +' ' + constants.ANSWER_TOKEN + ' ' + a + ' ' + constants.END_TOKEN


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

class VQANet(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer  = tokenizer
        self.image_backbone = MaskRNNBackbone()
        self.text_encoder = TextEncoder(self.tokenizer)
        self.embedding = self.text_encoder.model.embeddings
        text_embedding_size = self.text_encoder.model.config.hidden_size
        self.linear = nn.Linear(constants.IMAGE_BACKBONE_POOL_DIM, text_embedding_size)
        
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=text_embedding_size, nhead=8, batch_first = True)
        self.image_encoder =  nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        
        self.position_encoding = PositionalEncoding(d_model=text_embedding_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=text_embedding_size, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.output = nn.Linear(text_embedding_size, len(tokenizer))
        
    def forward(self, x, device):
        """
        x: List of ImageData
        """

        ids, images, captions, qas = flatten(x)
        N = images.shape[0]
        print(images.shape)

        feature_map = self.image_backbone(images)
        print("feature_map", feature_map.keys())
        pool = feature_map['pool']
        projection = self.linear(pool.view(N, -1))
        print("projection", projection.shape)
        
        image_embedding = self.image_encoder(projection)
        print("image_embedding", image_embedding.shape)
        
        cap_tokens = self.tokenizer(captions, padding=True , return_tensors="pt").to(device)
        captions_embedding = self.text_encoder(cap_tokens)
        print("captions_embedding", captions_embedding.shape)

        
        qa_tokens = self.tokenizer(qas, padding=True , return_tensors="pt").to(device)        
        print("qa_tokens keys", qa_tokens.keys())
        embed = self.embedding(input_ids = qa_tokens['input_ids'])
        
        # (seq, batch, embedding)
        embed = embed.transpose(0,1)
        print("embedding", embed.shape)
        
        image_embedding = torch.broadcast_to(image_embedding, embed.shape)
        qa_mask = nn.Transformer.generate_square_subsequent_mask(embed.shape[0]).to(device)
        
        output_embedding = self.decoder(tgt = embed, memory = image_embedding, tgt_mask=qa_mask )
        
        out_logits = self.output(output_embedding)
        
        print("out_logits", out_logits)
        return out_logits
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    