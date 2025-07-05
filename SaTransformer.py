import torch
import torch.nn as nn
from HIL import Squeeze, DilatedConvBlockB
import torch.nn.functional as F
import math
import copy
from self_attention import EncoderLayer

class SaTransformerNetwork(nn.Module):
    def __init__(self, hidden_feat_size):
        super(SaTransformerNetwork, self).__init__()
        self.max_d = 50
        self.max_p = 256
        self.input_dim_drug = 23532
        self.input_dim_targe = 16693
        self.emb_size = hidden_feat_size//2
        self.dropout_rate = 0.1
        self.hidden_size = hidden_feat_size//2
        self.intermediate_size = 256
        self.num_attention_heads = 4
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.n_layer = 2
        smi_embed_size = 128
        pkt_embed_size = 128
        PT_FEATURE_SIZE = 40

        #transformer
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_targe, self.emb_size, self.max_p, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.smi_attention_poc= EncoderLayer(128, 128, 0.1, 0.1, 4)
        self.poc_attention_smi= EncoderLayer(128, 128, 0.1, 0.1, 4)

        conv_smi_trans = []
        ic = smi_embed_size
        for oc in [32, 64, 128]:
            conv_smi_trans.append(DilatedConvBlockB(ic, oc))
            ic = oc
        conv_smi_trans.append(nn.AdaptiveMaxPool1d(1))
        conv_smi_trans.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi_trans)

        conv_pkt_trans = []
        ic = pkt_embed_size
        for oc in [32, 64, 64]:
            conv_pkt_trans.append(nn.Conv1d(ic, oc, 3))
            conv_pkt_trans.append(nn.BatchNorm1d(oc))
            conv_pkt_trans.append(nn.PReLU())
            ic = oc
        conv_pkt_trans.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt_trans.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt_trans)

        self.pkt_embed = nn.Linear(PT_FEATURE_SIZE, pkt_embed_size)

    def forward(self, prot_trans, lig_trans, pkt_trans, prot_mask, lig_mask):
        ex_l_mask = lig_mask.unsqueeze(1).unsqueeze(2)
        ex_l_mask = (1.0 - ex_l_mask) * -10000.0
        l_emb = self.demb(lig_trans)
        lt = self.d_encoder(l_emb.float(), ex_l_mask.float())# [B,L,D]

        ex_p_mask = prot_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        p_emb = self.pemb(prot_trans)
        p_mask1 = drop_tokens(p_emb, 0.1)
        pt = self.p_encoder(p_mask1.float(), ex_p_mask.float())# [B,L,D]

        pkt = self.pkt_embed(pkt_trans)# [B,L,D]

        smi_embed = self.smi_attention_poc(lt, pkt)
        pkt_embed = self.poc_attention_smi(pkt, pt)

        smi_embed = torch.transpose(smi_embed, 1, 2)# [B,D,L]
        smi_embed = self.conv_smi(smi_embed)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_embed= self.conv_pkt(pkt_embed)

        return smi_embed, pkt_embed

def drop_tokens(embeddings, word_dropout):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings

class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b = torch.LongTensor(1, 2)
        b = b.cuda()
        input_ids = input_ids.type_as(b)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DilatedConv(nn.Module):  # Dilated Convolution
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
            super().__init__()
            padding = int((kSize - 1) / 2) * d
            self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
            output = self.conv(input)
            return output