# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:13:32 2021
@author: wangxu
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_DEVICE_OEDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# 3 grapgh sparse-attention
#channel sparse attention

from .sagcn import GCN

import scipy.sparse as sp



# 3 GCN sparse-attention




# 3 GCN sparse-attention
#channel sparse attention
class GCN_CSA_Block(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(GCN_CSA_Block, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.gcn_channel = GCN(nfeat=64, nhid=8, nclass=64,  dropout=0)
        self.acti = nn.Tanh()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.thres = 0.5

    def normalize(self,mx):
        rowsum = torch.sum(mx,2)  # 对每一行求和
        r_inv = torch.pow(rowsum, -1)  # 求倒数
        r_inv[torch.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
        r_mat_inv = torch.stack([torch.diag(e) for e in r_inv])
        mx = torch.bmm(r_mat_inv,mx)
        return mx

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K = K.shape
        _, _, L_Q = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-2).expand(B, H, L_Q, L_K)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None],  M_top] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B,H, L_Q,V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V = V.shape

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None],
                    index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, x):
        m_batchsize, C, height = x.size()
        query = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        similarity_query = torch.cosine_similarity(query.unsqueeze(2), query.unsqueeze(1), dim=-1)
        adj_query = (similarity_query> self.thres).long()
        
        adj_query = adj_query + adj_query.transpose(1,2).multiply(adj_query.transpose(1,2) > adj_query)- adj_query.multiply(adj_query.transpose(1,2) > adj_query)
        query_features = self.normalize(query)
        eye_mat = torch.stack([torch.eye(adj_query.shape[1]) for _ in range(adj_query.shape[0])]).to(device)
        adj_query = self.normalize(adj_query + eye_mat)   
        
        queries = self.gcn_channel(query_features,adj_query)
        
        keys = queries
        values = queries
    
        
        U_part = 2 * np.ceil(np.log(keys.shape[-1])).astype('int').item() # c*ln(L_k)
        u = 2 * np.ceil(np.log(queries.shape[-1])).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<keys.shape[-1] else keys.shape[-1]
        u = u if u<queries.shape[-1] else queries.shape[-1]
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./np.sqrt(keys.shape[-2])
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, keys.shape[-1])
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, keys.shape[-1])
        
        # return self.acti(context.transpose(2,1).contiguous())
        return self.gamma*context.transpose(2,1).contiguous() + x
    
    


#Spatial sparse attention
class GCN_SSA_Block(nn.Module):
    def __init__(self, in_dim=64, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(GCN_SSA_Block, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.query_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.feature_len = 128
        self.gcn_query = GCN(nfeat=self.feature_len, nhid=8, nclass=self.feature_len,  dropout=0)
        self.gcn_key = GCN(nfeat=self.feature_len, nhid=8, nclass=self.feature_len,  dropout=0)
        self.gcn_value = GCN(nfeat=self.feature_len, nhid=8, nclass=self.feature_len,  dropout=0)
        self.acti = nn.Tanh()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.thres = 0.5

    def normalize(self,mx):
        rowsum = torch.sum(mx,2)  # 对每一行求和
        r_inv = torch.pow(rowsum, -1)  # 求倒数
        r_inv[torch.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
        r_mat_inv = torch.stack([torch.diag(e) for e in r_inv])
        mx = torch.bmm(r_mat_inv,mx)
        return mx        

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K = K.shape
        _, _, L_Q = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-2).expand(B, H, L_Q, L_K)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None],M_top] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, L_V, H = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, L_Q,H, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V = V.shape

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None],
                    index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, x):
        
        m_batchsize, C, height = x.size()
        queries = self.query_conv(x).view(m_batchsize, -1, height)
        keys = self.key_conv(x).view(m_batchsize, -1,height)
        values = self.value_conv(x).view(m_batchsize, -1,height)
        
        

        query = queries
        similarity_query = torch.cosine_similarity(query.unsqueeze(2), query.unsqueeze(1), dim=-1)
        adj_query = (similarity_query> self.thres).long()  
        adj_query = adj_query + adj_query.transpose(1,2).multiply(adj_query.transpose(1,2) > adj_query)- adj_query.multiply(adj_query.transpose(1,2) > adj_query)
        query_features = self.normalize(query)
        eye_query = torch.stack([torch.eye(adj_query.shape[1]) for _ in range(adj_query.shape[0])]).to(device)
        adj_query = self.normalize(adj_query + eye_query)       
        queries = self.gcn_query(query_features,adj_query)
        

        
        key = keys
        similarity_key = torch.cosine_similarity(key.unsqueeze(2), key.unsqueeze(1), dim=-1)
        adj_key = (similarity_key> self.thres).long()  
        adj_key = adj_key + adj_key.transpose(1,2).multiply(adj_key.transpose(1,2) > adj_key)- adj_key.multiply(adj_key.transpose(1,2) > adj_key)
        key_features = self.normalize(key)
        eye_key = torch.stack([torch.eye(adj_key.shape[1]) for _ in range(adj_key.shape[0])]).to(device)
        adj_key = self.normalize(adj_key + eye_key)       
        keys = self.gcn_query(key_features,adj_key)       
        
        value = values
        similarity_value = torch.cosine_similarity(value.unsqueeze(2), value.unsqueeze(1), dim=-1)
        adj_value = (similarity_value> self.thres).long()  
        adj_value = adj_value + adj_value.transpose(1,2).multiply(adj_value.transpose(1,2) > adj_value)- adj_value.multiply(adj_value.transpose(1,2) > adj_value)
        value_features = self.normalize(value)
        eye_value = torch.stack([torch.eye(adj_value.shape[1]) for _ in range(adj_value.shape[0])]).to(device)
        adj_value = self.normalize(adj_value + eye_value)       
        values = self.gcn_query(value_features,adj_value)        
        
        B, L_Q, H = queries.shape
        _, L_K, _ = keys.shape

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./np.sqrt(H)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        
        # return self.acti(context.contiguous())
        return self.gamma*context.contiguous() + x
    






    


class GCN_Sparse_Att_Block(nn.Module):
    def __init__(self, in_dim):
        super(GCN_Sparse_Att_Block, self).__init__()
        self.CSA = GCN_CSA_Block(in_dim)
        self.SSA = GCN_SSA_Block(in_dim)
        self.MAP = nn.Sequential(nn.Conv1d(in_dim, in_dim, 3, 1, 1))

    def forward(self, x):
        csa = self.CSA(x)
        ssa = self.SSA(csa)
        map = self.MAP(ssa)
        return map










class FECGNet(nn.Module):  
    def __init__(self, input_size=1, output_size=128):
        super().__init__()
        self.attention = GCN_Sparse_Att_Block(64)

        
        
        self.norm = nn.BatchNorm1d(1)
        # self.norm64 = nn.BatchNorm1d(64)
        # self.norm = nn.InstanceNorm1d(1)
        
        self.acti = nn.Tanh()
        self.classifier = nn.Linear(output_size, output_size)
        
        
        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh(),
                                  nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh())
        
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh(),
                                  nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh())
                
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh(),
                                  nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh())
        
        self.conv4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh(),
                                  nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh())
        
        self.conv5 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.Tanh(),
                                  nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1, bias=False))
 

    def forward(self, x):    
        x = self.conv1(x.to(torch.float32))
        
        x = self.attention(x)
        

        
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('out1')
        # plt.show()
        
        residual = x
        x = self.conv2(x)
        # x = self.attention(x)
        x = x + residual
        
        
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('out2')
        # plt.show()
   
        residual = x     
        x = self.conv3(x)
        x = x + residual
        
        
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('out3')
        # plt.show()
        
        residual = x
        x = self.conv4(x)
        x = x + residual
        
        
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('out4')
        # plt.show()
        
        
        x = self.conv5(x)
        
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('out5')
        # plt.show()
        
        # x = self.norm(x)

        # print(x.shape)
        # x = self.norm1(x)
        # x = self.classifier(x)
        # # plt.plot(in1[0].squeeze(0).detach().cpu().numpy(),'b')
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('out')
        # plt.show()
        
        # print(x.shape)
        # x = self.classifier(x)
        # plt.plot(x[0].squeeze(0).detach().cpu().numpy())
        # plt.title('linear')
        # plt.show()
        
        return x
       
    

def fecg(output_size=1000):
    model = FECGNet(output_size=output_size)
    return model


def main():
    x = torch.rand([1,128])
    model = FECGNet()
    print(model(x))

if __name__ == "__main__":
    main()
