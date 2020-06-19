import torch 
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F

# self attention
class self_attention(nn.Module):
    '''
    Module to apply self attention to an input sequence of vectors
    
    parameters:
    
    emb_dim = dimension of the embedding vector
    h = number of self attention heads
    
    '''
    def __init__(self, emb_dim, h):
        super().__init__()
        self.emb_dim = emb_dim
        self.h = h
        self.red_vec_size = emb_dim//h
        
        # Querry vector
        self.WQ = nn.Linear(emb_dim, self.red_vec_size, bias = False)
        self.WK = nn.Linear(emb_dim, self.red_vec_size, bias = False)
        self.WV = nn.Linear(emb_dim, self.red_vec_size, bias = False)
        
    def forward(self, x):
        # x has shape (batch_size, seq_len, emb_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        querries = self.WQ(x)
        keys = self.WK(x)
        values = self.WV(x)
        att_scores = F.softmax(querries@keys.permute(0,2,1) \
                               /np.sqrt(self.red_vec_size), dim = 2)
        ctx_vecs = att_scores @ values 
        assert ctx_vecs.shape == (batch_size, seq_len, self.red_vec_size ) 
        return querries, keys, values, att_scores, ctx_vecs
    
    
    

# multi-head self-attention

class multi_head_attn(nn.Module):
    '''
    Module to create multiple attention heads
    
    parameters:
    
    emb_dim = dimension of the embedding vectors
    h = number of attention heads
    parallelize = parallelize the computations for differnt heads 
    
    '''
    def __init__(self, emb_dim, h, p_drop = 0.1, parallelize = 'False'):
        super().__init__()
        self.emb_dim = emb_dim
        self.h = h
        self.red_vec_size = emb_dim // h 
        
        self.heads = nn.ModuleList([self_attention(emb_dim, h) for i in range(h)])
        # need to wrap the heads with nn.ModuleList to make sure they are properly registered
        # without doing so the parameters of the modules in the list do not get registered
        # for e.g. see the following stack exchange post
        # https://stackoverflow.com/questions/50463975/pytorch-how-to-properly-create-a-list-of-nn-linear
        # and here
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
        # and here
        # https://pytorch.org/docs/master/generated/torch.nn.ModuleList.html
        
        # transform the contatenated context vectors to have same size as emb_sim
        # this is to be able to enable implement a skip-connection between the input and output
        self.Wo = nn.Linear(self.red_vec_size*h, emb_dim, bias = False) 
        
        # layer norm
        # should we apply 
        self.LNorm = nn.LayerNorm(emb_dim)
        
        self.drop = nn.Dropout(p_drop)
        
    def forward(self, x):
        ctx_vecs = torch.cat([head(x)[4] for head in self.heads], dim = 2)
        transformed = self.drop(self.Wo(ctx_vecs))
        
        return self.LNorm(x + transformed)
    

# single encoder layer    
class encoder(nn.Module):
    '''
    The complete encoder module.
    
    parameters:
    
    emb_dim = dimension of the embedding vectors
    h = number of attention heads
    parallelize = parallelize the computations for differnt heads 
    ffn_l1_out_fts = number of out_features of 1st layer in feed forward NN. Default is 2048 a suggested in the original paper
    
    
    '''
    
    def __init__(self, emb_dim, h, p_drop = 0.1, parallelize = False, ffn_l1_out_fts = 2048 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.h = h
        self.red_vec_size = emb_dim//h
        
        # multi_head_attention sub-layer
        self.mul_h_attn = multi_head_attn(emb_dim, h, p_drop, parallelize)
        
        # feedforward sublayers
        self.l1 = nn.Linear(emb_dim, ffn_l1_out_fts)
        self.l2 = nn.Linear(ffn_l1_out_fts, emb_dim)
        
        # layer norm
        self.LNorm = nn.LayerNorm(emb_dim) 
        
        self.drop = nn.Dropout(p_drop)
        
    def forward(self, x):
        ctx_vecs = self.mul_h_attn(x)
        out = torch.relu(self.l1(ctx_vecs))
        out = self.drop(self.l2(out))
        
        return self.LNorm(out + ctx_vecs)    
    
    
# encoder decoder attention
class encoder_decoder_attention(nn.Module):
    '''
    Module to implement the encoder_decoder attention layer. 
    This is same as the self_attention layer except that it takes two input vectors: 
                 1)encoder's final output 
                 2) output from previous decoder layer
    The querries are generated from the previous decoder layer's output
    The keys and the values are generated from the encoder's output 
         
    '''
    def __init__(self, emb_dim, h):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.h = h
        self.red_vec_size = emb_dim//h
        
        # Querry vector
        self.WQ = nn.Linear(emb_dim, self.red_vec_size, bias = False)
        # Key vector
        self.WK = nn.Linear(emb_dim, self.red_vec_size, bias = False)
        # Value vector
        self.WV = nn.Linear(emb_dim, self.red_vec_size, bias = False)
        
    def forward(self, enc_out, dec_out):
        # x has shape (batch_size, seq_len, emb_dim)
        batch_size = enc_out.shape[0]
        seq_len = dec_out.shape[1] 
        querries = self.WQ(dec_out)
        keys = self.WK(enc_out)
        values = self.WV(enc_out)
        att_scores = F.softmax((querries@keys.permute(0,2,1))\
                               /np.sqrt(self.red_vec_size), dim = 2)
        ctx_vecs = att_scores @ values 
        assert ctx_vecs.shape == (batch_size, seq_len, self.red_vec_size ) 
        return querries, keys, values, att_scores, ctx_vecs
    

# multi-head encoder decoder attention   
class multi_head_enc_dec_attn(nn.Module):
    def __init__(self, emb_dim, h, p_drop = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.h = h
        self.red_vec_size = emb_dim // h 
        
        self.heads = nn.ModuleList([encoder_decoder_attention(emb_dim, h) for i in range(h)])
        
        # transform the contatenated context vectors to have same size as emb_sim
        # this is to be able to enable implement a skip-connection between the input and output
        self.Wo = nn.Linear(self.red_vec_size*h, emb_dim, bias = False) 
        
        # layer norm
        # should we apply 
        self.LNorm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(p_drop)
        
    def forward(self, enc_out, dec_out):
        ctx_vecs = torch.cat([head(enc_out, dec_out)[4] for head in self.heads], dim = 2)
        transformed = self.drop(self.Wo(ctx_vecs))
        
        return self.LNorm(dec_out + transformed)    
    
# single decoder layer    
class decoder(nn.Module):
    '''
    The complete decoder module. 
    
    parameters:
    
    emb_dim = dimension of the embedding vectors
    h = number of attention heads
    parallelize = parallelize the computations for differnt heads 
    ffn_l1_out_fts = number of out_features of 1st layer in feed forward NN. Default is 2048 a suggested in the original paper
    
    '''
    def __init__(self, emb_dim, h, p_drop = 0.1, parallelize = False, ffn_l1_out_fts = 2048):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.h = h
        self.red_vec_size = emb_dim//h
        
        # multi_head_attention sub-layer
        self.mul_h_attn = multi_head_attn(emb_dim, h, p_drop, parallelize)
        
        # multi head encoder decoder attention sublayer
        self.mul_h_enc_dec_attn = multi_head_enc_dec_attn(emb_dim, h, p_drop)
        
        # feedforward sublayers
        self.l1 = nn.Linear(emb_dim, ffn_l1_out_fts)
        self.l2 = nn.Linear(ffn_l1_out_fts, emb_dim)
        
        # layer norm
        self.LNorm = nn.LayerNorm(emb_dim) 
        
        self.drop = nn.Dropout(p_drop)
        
    def forward(self, enc_vecs, dec_vecs):
        dec_vecs = self.mul_h_attn(dec_vecs)
        ff_in = self.mul_h_enc_dec_attn(enc_vecs, dec_vecs)
        out = torch.relu(self.l1(ff_in))
        out = self.drop(self.l2(out))
        
        return self.LNorm(out + ff_in)  
    
    
# positional encoding function
def positional_encoding(emb_dim, seq_len):
    posts = torch.arange(seq_len).unsqueeze(1)
    pows = 10000**(torch.arange(emb_dim//2)/float(emb_dim))
    mat = posts/pows # rows = position in the sequence , # col = index along the embedding space
    first_half = torch.sin(mat)
    second_half = torch.cos(mat)
    out = torch.cat((first_half, second_half), dim = 1)
    return out    

# The full encoder stack
class encoder_stack(nn.Module):
    def __init__(self, emb_dim, h, p_drop = 0.1, parallelize = False, 
                 ffn_l1_out_fts = 2048,n_encoders = 3):
        super().__init__()
        
        self.encoders = [encoder(emb_dim, h, p_drop , 
                                 parallelize, 
                                 ffn_l1_out_fts)  for _ in range(n_encoders)]
        
    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        
        return x 
    
# The full decoder stack
class decoder_stack(nn.Module):
    def __init__(self, emb_dim, h, p_drop = 0.1, parallelize = False, 
                 ffn_l1_out_fts = 2048, n_decoders = 3 ):
        super().__init__()
        
        self.decoders = [decoder(emb_dim, h, p_drop = 0.1, 
                                 parallelize = False, 
                                 ffn_l1_out_fts = 2048) for _ in range(n_decoders)]
        
    def forward(self, enc_vecs, dec_vecs):
        for decoder in self.decoders:
            dec_vecs = decoder(enc_vecs, dec_vecs)
        
        # the decoder stack returns only the feature vector corresponding to 
        # the last step in decoder's input seq
        # this is then expected to be passed to through a fully-connected network which will 
        # output the logits corresponding to the next possible word
        return dec_vecs[:, -1]
    
# The full transformer
class Transformer(nn.Module):
    '''
    Transformer implementation by Prarit Agarwal
    Based on https://arxiv.org/abs/1706.03762
    
    Imp: Though the Transformer is parallelizable by design. We haven't yet implemented the parallelization yet.
    
    Parameters:
    enc_emb = pretrained embedding matrix to apply to encoder's input
    dec_emb = pretrained embedding matrix to apply to decoder's input/output
    num_heads = number of attention heads
    n_encoders = number of encoder layers in the encoder stack; default is 3
    n_decoders = number of decoder layers in the decoder stack; default is 3
    p_drop = probability of dropout 
    ffn_l1_out_fts = number of output features of the 1st feed-forward sublayer layer in encoder/decoder layers
    pad_idx = idx for the padding token
    bos_idx = idx for begining of string token
    max_sq_len = maximum length of the output sequence
    positional_encoding_func = function to generate positional encodings
    parallelize = to parallelize or not; parallelization yet to be implemented!
    
    
    
    '''
    
    def __init__(self, enc_emb, dec_emb, num_heads, 
                 n_encoders = 3, n_decoders = 3,
                 p_drop = 0.1, ffn_l1_out_fts = 2048, 
                 pad_idx = 1, bos_idx = 0, max_sq_len = 30,
                 positional_encoding_func = positional_encoding, 
                 parallelize = False ):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.max_sq_len = max_sq_len
        
        # encoder 
        self.enc_emb = nn.Embedding.from_pretrained(enc_emb, freeze = False,
                                                     padding_idx = pad_idx)
        
        self.emb_dim = self.enc_emb.embedding_dim # dimension of embedding vectors
        
        self.h = num_heads
        self.p_drop = p_drop
        
        self.positional_encoding = positional_encoding_func
        
        self.drop_input = nn.Dropout(p_drop)
        
        self.encoder = encoder_stack(self.emb_dim, num_heads, p_drop = p_drop, 
                                     parallelize = parallelize,
                                     ffn_l1_out_fts = ffn_l1_out_fts,
                                     n_encoders = n_encoders)
        
        # decoder 
        
        self.dec_emb = nn.Embedding.from_pretrained(dec_emb, freeze = False, 
                                                    padding_idx = pad_idx)
        self.num_words = self.dec_emb.num_embeddings 
        self.decoder = decoder_stack(self.emb_dim, num_heads, p_drop = p_drop, 
                                     parallelize = parallelize,
                                     ffn_l1_out_fts = ffn_l1_out_fts, 
                                     n_decoders = n_decoders)
        
        # logits from decoder output
        self.logits = nn.Linear(self.emb_dim, self.num_words)
        # we will tie the weights of the logits layer to that of the dec_embeddings
        # that this improves translation was suggested in the following paper
        # https://arxiv.org/abs/1608.05859
        # this was also done in the transformer paper
        self.logits.weight = self.dec_emb.weight
        # The above weight tying is identical to the one done in the following pytorch example
        # https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L28
        
        
    def forward(self, x):
        # has shape (batch_size, seq_length); entries in a sequence correspond to word indices
        batch_size, enc_seq_len = x.shape
        enc_embeddings = self.enc_emb(x) # embeddings to input to the encoder
        # add positional encodings to the encoder's input embeddings
        enc_pe = self.positional_encoding(self.emb_dim, enc_seq_len).to(device = x.device)
        enc_in = self.drop_input(enc_embeddings + enc_pe)
        enc_out = self.encoder(enc_in)
        dec_in_seq = torch.tensor([[self.bos_idx]]*batch_size).to(device = x.device)
        # dec_in_seq contains word indices obtained  from the decoder; Initialised to bos_idx
        out_seq_logits = []
        for itr in range(self.max_sq_len):
            dec_embeddings = self.dec_emb(dec_in_seq) 
            dec_seq_len = dec_in_seq.shape[1]
            # add positional encodings to dec_embeddings
            dec_pe = self.positional_encoding(self.emb_dim, dec_seq_len).to(device = x.device)
            dec_in = self.drop_input(dec_embeddings + dec_pe)
            dec_out = self.decoder(enc_out, dec_in)
            # recall that the decoder_stack always returns the feature vector corresponding
            # to the last word in the decoder input sequence. Therefore, dec_out will be of
            # shape (batch_size, emb_dim)
            # we need to pass this through a dense layer to convert it to 
            # logits for the output words; probs are obtained by applying softmax activation
            # recall that pytorch cross entropy loss combines log_softmax and NLLLoss
            # so here we will not apply softmax and pass logits to the loss function
            logits = self.logits(dec_out)
            #probs = F.softmax(self.logits(dec_out), dim = 1)
            # logits as well as probs, have shape = batch_size x num_words
            out_seq_logits.append(logits)
            #next_word = probs.max(dim = 1)[1]
            next_word = logits.argmax(dim = 1)
            if all(next_word == self.pad_idx): break
            dec_in_seq = torch.cat((dec_in_seq, next_word.unsqueeze(1)), axis = 1)
        
        return torch.stack(out_seq_logits, dim = 1)