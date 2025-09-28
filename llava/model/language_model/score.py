'''
Copyright (2024) Peking University. 
Developers: Yuan Zhang, Chun-Kai Fan

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def attn_postprocess_rank(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, scale, bias, layer_idx, keep_percent=-1, accumulated_attn_weights=None, mode='fixed', image_shape=144, use_abs=False):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    # if accumulated_attn_weights is not None and len(accumulated_attn_weights) > 0:
    #     assert accumulated_attn_weights[0].shape == self_attn_weights.shape
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K] [1, 32, 664, 664]

    t_token_idx = t_token_idx[1] + text_token_start # [21] 重要的文本token数量
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start+v_token_num] # B, L2, L1 [1, 21, 576]

    # entropy = compute_entropy(relation_vis_text)
    # rank = torch.linalg.matrix_rank(relation_vis_text.float()) # rank 21
    # if (accumulated_attn_weights is not None and len(accumulated_attn_weights) > 0):
    #     bef_token_num = accumulated_attn_weights[0][1].shape[1] # 21
    #     pre_relation_vis_text_sum = relation_vis_text.new_zeros(accumulated_attn_weights[0][1].shape)
    #     for idx, pre_self_attn_weights in enumerate(accumulated_attn_weights):
    #         pre_relation_vis_text = pre_self_attn_weights[1]
    #         pre_relation_vis_text_sum += pre_relation_vis_text
    #     relation_vis_text = relation_vis_text.mean(1) 
    #     relation_vis_text[:, :bef_token_num] = (relation_vis_text[:, :bef_token_num] * 1 + pre_relation_vis_text_sum *0.5) / 1.5
    # else:
    relation_vis_text = relation_vis_text.mean(1) # B, L1 [1, 576]
    
    # relation_vis_text = relation_vis_text * entropy

    s_flag = True # layer needs sparsification or not
    assert keep_percent >= 0, "keep_percent should be set in fixed mode"
    mask = torch.zeros_like(relation_vis_text, dtype=bool) 
    if use_abs:
        _, indices = torch.topk(relation_vis_text, min(math.ceil((image_shape - 1) * keep_percent), relation_vis_text.shape[1]), dim=1)
    else:
        _, indices = torch.topk(relation_vis_text, math.ceil((v_token_num - 1) * keep_percent), dim=1)
    mask[0][indices] = 1

    return mask, s_flag, relation_vis_text


def compute_entropy(att_map, dim=1):
    """
    计算 N × M Attention Map 每行（或每列）的 Shannon 熵
    :param att_map: (N × M) Tensor
    :param dim: 计算维度，1 = 按行计算, 0 = 按列计算
    :return: 熵 (N) 或 (M) Tensor
    """
    # 归一化，转换为概率分布
    att_map = att_map / torch.sum(att_map, dim=dim, keepdim=True)
    
    # 避免 log(0)
    att_map = torch.clamp(att_map, min=1e-10)

    # 计算 Shannon 熵
    entropy = -torch.sum(att_map * torch.log2(att_map), dim=dim)
    entropy /= entropy.max()
    
    return entropy