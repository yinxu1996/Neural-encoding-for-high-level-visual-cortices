import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import math
import torch
import torch.nn as nn

class FMRIEmbedding(nn.Module):
    def __init__(self, brain_roi, enc_in, e_model, dropout):
        super(FMRIEmbedding, self).__init__()
        self.dorsal_linear = nn.Linear(enc_in*5, e_model)
        self.ventral_linear = nn.Linear(enc_in*9, e_model)
        self.visualrois_linear = nn.Linear(enc_in*7, e_model)
        self.faces_linear = nn.Linear(enc_in*2, e_model)
        self.words_linear = nn.Linear(enc_in*2, e_model)
        self.places_linear = nn.Linear(enc_in, e_model)
        self.bodies_linear = nn.Linear(enc_in*2, e_model)
        self.small_linear = nn.ModuleList([nn.Linear(enc_in, e_model) for _ in range(brain_roi)])
        self.layernorm = nn.LayerNorm(e_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dorsal_src, ventral_src, 
                visualrois_src, faces_src, words_src, places_src, bodies_src,
                src):
        dorsal_x = torch.unsqueeze(self.dorsal_linear(dorsal_src), dim=1)
        ventral_x = torch.unsqueeze(self.ventral_linear(ventral_src), dim=1)
        
        visualrois_x = torch.unsqueeze(self.visualrois_linear(visualrois_src), dim=1)
        faces_x = torch.unsqueeze(self.faces_linear(faces_src), dim=1)
        words_x = torch.unsqueeze(self.words_linear(words_src), dim=1)
        places_x = torch.unsqueeze(self.places_linear(places_src), dim=1)
        bodies_x = torch.unsqueeze(self.bodies_linear(bodies_src), dim=1)
        rois_x = []
        for i, linear_layer in enumerate(self.small_linear):
            roi_x = linear_layer(src[:,i,:])
            rois_x.append(roi_x)
        rois_x = torch.stack(rois_x, dim=1)

        # multi-scale
        x = torch.concatenate((self.dropout(dorsal_x), self.dropout(ventral_x),
                                self.dropout(visualrois_x), self.dropout(faces_x), self.dropout(words_x), self.dropout(places_x), self.dropout(bodies_x),
                                self.dropout(rois_x)), dim=1)
        
        x = self.layernorm(x)
        return x
    
class WordEmbedding(nn.Module):
    def __init__(self, dec_in, d_model):
        super(WordEmbedding, self).__init__()
        self.embed = nn.Embedding(dec_in, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.embed(x)
        x = self.layernorm(x)
        return x

class Cate(nn.Module):
    def __init__(self, e_model, supercategories):
        super(Cate, self).__init__()
        self.cf1 = nn.Linear(e_model, e_model*2)
        self.cf2 = nn.Linear(e_model*2, supercategories)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.cf1(x))
        x_feature = self.cf2(x)
        x = self.softmax(x_feature)
        return x_feature, x
    
class Name(nn.Module):
    def __init__(self, e_model, names):
        super(Name, self).__init__()
        self.cf1 = nn.Linear(e_model, e_model*2)
        self.cf2 = nn.Linear(e_model*2, names)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.cf1(x))
        x = self.sigmoid(self.cf2(x))
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob)
            for _ in range(4)
        ])
        self.dropout = nn.Dropout(dropout_prob)

        self.scale_range = [
            [0,3,4],  # large scale
            [1,5,6,7,8,9],  # midle scale
            [2,10,11,12,13,14,15,16,17,18,19,20,21,22,23]  # small scale
        ]
        self.inter_scale_range = [0, 1, 2] 

    def forward(self, x):
        attn_weights_list = {}
        for i, scale_range in enumerate(self.scale_range):
            scale_x = x[scale_range, :, :] 
            attn_output, attn_weights = self.attn_layers[i](scale_x, scale_x, scale_x)
            x[scale_range,:,:] = attn_output
            attn_weights_list[f"intra_scale_{i}"] = attn_weights[:,1:,1:]
        
        inter_scale_x = x[self.inter_scale_range]
        inter_scale_output, attn_weights = self.attn_layers[3](inter_scale_x, inter_scale_x, inter_scale_x)
        x[self.inter_scale_range,:,:] = inter_scale_output
        attn_weights_list["inter_scale"] = attn_weights
        
        x = self.dropout(x)
        return x, attn_weights_list

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feed_forward_dim, dropout_prob=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.attention = MultiScaleAttention(hidden_size, num_heads, dropout_prob)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, hidden_size)
        )
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x)
        x = self.layernorm1(x + attn_output)
        x = self.dropout(x)

        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + ff_output)
        x = self.dropout(x)
        
        return x, attn_weights

class CustomTransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, feed_forward_dim, num_layers, dropout_prob=0.1):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(hidden_size, num_heads, feed_forward_dim, dropout_prob)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x, _ = layer(x)
        return x.permute(1,0,2)

class Transformer_multitasks(nn.Module):
    def __init__(self, configs):
        super(Transformer_multitasks, self).__init__()

        self.device = configs.gpu
        self.maxlen = configs.cap_len
        self.beam_size = configs.beam_size
        self.embed = nn.Embedding(configs.dec_in, configs.e_model)

        self.a_fmri = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

        self.softmax_weight = nn.Parameter(torch.ones(configs.d_model, configs.dec_in))
        self.softmax_bias = nn.Parameter(torch.zeros(configs.dec_in))

        self.fmri_embed = FMRIEmbedding(
            configs.brain_roi,
            configs.enc_in,
            configs.e_model,
            configs.dropout
        )
        
        self.encoder = CustomTransformerEncoder(configs.d_model, configs.n_heads, configs.d_ff, 
                                                configs.e_layers, configs.dropout)

        # Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(
            d_model=configs.d_model, 
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=configs.d_layers)

        # classification
        self.cate = Cate(configs.e_model, configs.supercategories)
        self.name = Name(configs.e_model, configs.labels)

    def positional_encoding(self, sequence, max_len):
        batch_size, seq_len, embedding_dim = sequence.size()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)) 
        
        pos_encoding = torch.zeros(max_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.repeat(batch_size, 1, 1)  

        pos_encoding = pos_encoding[:, :seq_len, :]
        return pos_encoding

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def create_mask(self, tgt, device):
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        tgt_padding_mask = (tgt == 0)
        return tgt_mask, tgt_padding_mask

    def encode(self, src):
        # large-scale
        # dorsal stream
        dorsal_src = src[:,:5,:].reshape(src.shape[0], -1)
        # ventral stream
        ventral_src = src[:,5:,:].reshape(src.shape[0], -1)
        # middle-scale
        # prf-visualrois
        visualrois_src = src[:,[5,0,6,1,7,2,8],:].reshape(src.shape[0], -1)
        # floc-faces
        faces_src = src[:,[9,10],:].reshape(src.shape[0], -1)
        # floc-words
        words_src = src[:,[11,12],:].reshape(src.shape[0], -1)
        # floc-places
        places_src = src[:,3,:].reshape(src.shape[0], -1)
        # floc-bodies
        bodies_src = src[:,[4,13],:].reshape(src.shape[0], -1)
              
        src = self.fmri_embed(dorsal_src, ventral_src, 
                              visualrois_src, faces_src, words_src, places_src, bodies_src,
                              src)
        return src
    
    def decode(self, memory, cap_input, tgt_mask, tgt_key_padding_mask):
        dec = self.embed(cap_input)
        dec *=  dec.shape[-1] ** 0.5
        dec += self.b * self.positional_encoding(dec, self.maxlen).to(self.device)
        output = self.decoder(dec, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.view(-1, memory.shape[-1])
        logits = torch.matmul(output, self.softmax_weight) + self.softmax_bias
        return logits
    
    def voc_ap(slef, rec, prec, true_num):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def mAP(self, predict, target):
        class_num = target.shape[1]
        seg = np.concatenate((predict, target), axis=1)
        gt_label = seg[:, class_num:].astype(np.int32)
        num_target = np.sum(gt_label, axis=1, keepdims=True)
        threshold = 1 / (num_target + 1e-6)
        predict_result = seg[:, 0:class_num] > threshold
        sample_num = len(gt_label)
        tp = np.zeros(sample_num)
        fp = np.zeros(sample_num)
        aps = []
        recall = []
        precise = []
        for class_id in range(class_num):
            confidence = seg[:, class_id]
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            sorted_label = [gt_label[x][class_id] for x in sorted_ind]
            for i in range(sample_num):
                tp[i] = (sorted_label[i] > 0)
                fp[i] = (sorted_label[i] <= 0)
            true_num = sum(tp)
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            if true_num == 0:
                rec = tp / 1000000
            else:
                rec = tp / float(true_num)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            recall += [rec]
            precise += [prec]
            ap = self.voc_ap(rec, prec, true_num)
            aps += [ap]
        np.set_printoptions(precision=3, suppress=True)

        mAPvalue = np.mean(aps)
        return mAPvalue
    
    def cate_loss(self, memory, cate):
        feature, pre_cate_label = self.cate(memory)
        acc_cate = (pre_cate_label.argmax(dim=1)==cate).float().mean()
        loss_cate = nn.functional.cross_entropy(feature, cate)
        return pre_cate_label, acc_cate, loss_cate
    
    def name_loss(self, memory, name):
        pre_name_label = self.name(memory)
        mAP = self.mAP(pre_name_label.cpu().detach().numpy(), name.cpu().detach().numpy())
        loss_name = nn.functional.binary_cross_entropy(pre_name_label, name)
        loss_name1 = nn.functional.mse_loss(pre_name_label, name)
        return pre_name_label, loss_name+loss_name1, mAP
    
    def cap_loss(self, memory, cap_input, cap_output, tgt_mask, tgt_padding_mask):
        pre_cap = self.decode(memory, cap_input, tgt_mask, tgt_padding_mask)
        loss_cap = nn.functional.cross_entropy(pre_cap, cap_output.reshape(-1), reduction='none')
        label_weigths = ~tgt_padding_mask.reshape(-1)
        loss_cap = torch.sum(loss_cap * label_weigths) / len(memory)
        return loss_cap

    def forward(self, src, mt, cate, name, cap_input, cap_output, tgt_mask, tgt_padding_mask):
        mt_emb = self.embed(mt)
        src = self.encode(src)
        src = torch.cat((mt_emb, src), dim=1)
        src *= 2 * src.shape[-1] ** 0.5
        src += self.a_fmri * self.positional_encoding(src, src.shape[1]).to(self.device)
        memory = self.encoder(src)
        token = torch.mean(memory[:,:3,:], axis=1)
        """  category  """
        pre_cate_label, acc_cate, loss_cate = self.cate_loss(token, cate)
        """ semantic  """
        pre_name_label, loss_name, mAP = self.name_loss(token, name)
        """ language  """
        loss_cap = self.cap_loss(token.unsqueeze(1), cap_input, cap_output, tgt_mask, tgt_padding_mask)
        
        return pre_cate_label, acc_cate, loss_cate, \
            pre_name_label, loss_name, mAP, \
            loss_cap
        
    


