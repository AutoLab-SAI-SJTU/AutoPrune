#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # FasterVLM
    def encode_images(self, images, masks=None, use_image_text_blance=False):
        image_features, image_attentions = self.get_model().get_vision_tower()(images) # (B, N, C), (B, M, N) = (1, 576, 1024), (1, 16, 576)

        # image_attentions = image_attentions.max(dim=1)[0] # (B, N) = (1, 576)
        image_attentions = image_attentions.mean(dim=1) # (B, N) = (1, 576)

        B, N = image_features.shape[:2]
        visual_token_num = self.get_visual_token_num() # T

        # prune visual tokens by random scores
        # token_weights = torch.rand(B, N, device=image_features.device) # (B, N)
        # token_indices = torch.topk(token_weights, k=visual_token_num, dim=1)[1] # (B, T)
        # token_indices = torch.sort(token_indices, dim=1)[0] # (B, T)
        
        if masks is not None:
            masks = masks.reshape(B, N)
            image_attentions[masks.bool()] *= 0.0
        # prune visual tokens by attention scores
        
        token_indices = torch.topk(image_attentions, k=visual_token_num, dim=1)[1] # (B, T)
        vis = False
        if vis:
            import matplotlib.pyplot as plt
            from torchvision.transforms import ToPILImage
            from PIL import Image, ImageDraw
            
            # 逆归一化图像
            denormalized_image = denormalize(images[0])  # 假设 denormalize 函数已定义
            to_pil = ToPILImage()
            denormalized_image_clipped = torch.clamp(denormalized_image, 0, 1)
            pil_image = to_pil(denormalized_image_clipped).convert("RGBA")  # 转换为 RGBA 以支持透明度
            
            # 创建一个用于绘制半透明方框的图层
            overlay = Image.new('RGBA', pil_image.size, (0,0,0,0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            patch_size = 14  # 每个 patch 的尺寸（14x14 像素）
            grid_size = 24    # 每行和每列的 patch 数量（24x24）
            image_size = 336  # 图像的尺寸（336x336 像素）
            
            # 获取保留的 token 索引，并转换为集合以便快速查找
            kept_indices = token_indices[0].cpu().numpy()  # [T]
            kept_set = set(kept_indices)
            
            for idx in range(N):
                if idx not in kept_set:
                    # 计算当前 patch 的行和列
                    row = idx // grid_size
                    col = idx % grid_size
                    # 计算 patch 的左上角和右下角坐标
                    x0 = col * patch_size
                    y0 = row * patch_size
                    x1 = x0 + patch_size
                    y1 = y0 + patch_size
                    # 在被裁剪的 patch 位置绘制半透明灰色方框
                    draw_overlay.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 150))  # 灰色，透明度 100/255
            
            # 将覆盖图层与原始图像合并
            pil_image_with_overlay = Image.alpha_composite(pil_image, overlay)
            
            # 在合并后的图像上绘制网格线
            draw_grid = ImageDraw.Draw(pil_image_with_overlay)
            for i in range(grid_size + 1):
                # 绘制垂直线
                x = i * patch_size
                draw_grid.line([(x, 0), (x, image_size)], fill=(255, 255, 255, 255), width=1)
                # 绘制水平线
                y = i * patch_size
                draw_grid.line([(0, y), (image_size, y)], fill=(255, 255, 255, 255), width=1)
            
            # 将图像转换回 RGB 模式，并保存为 JPEG 文件
            pil_image_with_overlay = pil_image_with_overlay.convert("RGB")  # 移除 alpha 通道
            pil_image_with_overlay.save('/home/hswang/paper/llm/FasterVLM/vis_jpg/vis4.jpg')


        token_indices = torch.sort(token_indices, dim=1)[0] # (B, T)

        # generate index mask
        index_mask = torch.zeros(B, N, dtype=torch.bool, device=image_features.device) # (B, N)
        index_mask.scatter_(1, token_indices, True) # (B, N)

        image_features = self.get_model().mm_projector(image_features) # (B, N, D) [1, 576, 1024] -> [1, 576, 4096]
        
        return image_features, index_mask, image_attentions

    # FasterVLM
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, mask_tensors=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks, image_attns = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            # mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = 'spatial' # ! only support 'spatial' and 'spatial_unpad'
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        base_index_mask = index_mask[0]
                        image_feature = image_feature[1:]
                        index_mask = index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, index_masks, image_attns = self.encode_images(images, masks=mask_tensors)
            new_image_features = []
            for image_feature, index_mask in zip(image_features, index_masks):
                image_feature = image_feature[index_mask] # [N, 4096]
                new_image_features.append(image_feature)
            image_features = torch.stack(new_image_features, dim=0) # [B, N, 4096]

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME 没有用到这个，没区别
        _input_ids = input_ids # 这是一个batch的input_ids，每个input_ids是一个序列，来自于数据集
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0: # 这里应该是判断有没有图像输入
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # 遍历 image_token_indices，提取图像和文本的token
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] # 存储每个段落的长度
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) # 将输入的词语索引张量转换为对应的嵌入向量张量 [N, 4096]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx] # 取出每个batch的图像信息
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) # 文本token映射的特征和图像特征进行拼接
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None: # None
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them 将序列填充到统一长度
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        #                None           None             None       [1, 376, 4096]      None           288                     [1, 576]
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features[0].shape[0], image_attns

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
    def prepare_sparse_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, mask_tensors=None, use_image_text_blance=False, use_image_text_blance2=True
    ):
        vision_tower = self.get_vision_tower()      # CLIPVisionTower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            self.model.generate_process_count += 1 # ! 这里不同
            return input_ids, position_ids, attention_mask, past_key_values, None, labels,self.image_shape,self.token_length_list,self.pre_prompt_length_list

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks, image_attns = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_attns = torch.split(image_attns, split_sizes, dim=0)
            # mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = 'spatial' # ! only support 'spatial' and 'spatial_unpad'
            # assert mm_patch_merge_type == 'spatial', f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                new_image_attns = []
                for image_idx, (image_feature, image_attn) in enumerate(zip(image_features, image_attns)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        base_image_attn = image_attn[0]
                        image_feature = image_feature[1:]
                        image_attn = image_attn[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            image_attn = image_attn.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            raise NotImplementedError
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_attn = image_attn.permute(0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            image_attn = image_attn.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        image_attn = torch.cat((base_image_attn, image_attn), dim=0)
                    else:
                        image_feature = image_feature[0]
                        image_attn = image_attn[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            image_attn = torch.cat((
                                image_attn,
                                torch.ones(1, dtype=torch.bool).to(image_attn.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                    new_image_attns.append(image_attn)
                image_features = new_image_features
                image_attns = new_image_attns
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # image_features = self.encode_images(images) # 从[1, 3, 336, 336]变成([1, 576, 4096])
            if use_image_text_blance:
                image_features, index_masks, image_attns = self.encode_images(images, masks=mask_tensors, use_image_text_blance=use_image_text_blance)
            else:
                image_features, index_masks, image_attns = self.encode_images(images, masks=mask_tensors)
                new_image_features = []
                for image_feature, index_mask in zip(image_features, index_masks):
                    image_feature = image_feature[index_mask] # [N, 4096]
                    new_image_features.append(image_feature)
                image_features = torch.stack(new_image_features, dim=0) # [B, N, 4096]


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:    #  用input_ids的长度来填充position_ids
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # 决定哪些部分需要被mask掉
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []   # 下面那个大的for循环，也就是把pre_prompt_embedding , image_embedding , question_embedding给拼接起来
        new_labels = []
        cur_image_idx = 0
        pre_prompt_length_list = []      # 记录哪些token是预先的prompt，这些token不进行稀疏化
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # 1
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # 因为他input的格式是<pre_prompt><image><question>，<image>是一个分隔符，下面这行代码是为了找到<image>的位置,如[-1, 35, 93]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] 
            pre_prompt_length_list.append(image_token_indices[1])
            cur_input_ids_noim = [] # 以列表的方式存入input，将<image>分隔符删除，也即[ <pre_prompt> , <question> ]
            cur_labels = labels[batch_idx]
            cur_labels_noim = []    # 以列表的方式存入label，将<image>分隔符删除，也即[ <pre_prompt> , <question> ]
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] # [35, 57]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) # 相当于不对<image>这个标识符进行embedding，torch.Size([92, 4096])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)  # 再将embedding分成<pre_prompt>、<question>两部分
            cur_new_input_embeds = [] # 在for循环后变成[ pre_prompt_embedding , image_embedding , question_embedding ]，[35,576,57]
            cur_new_labels = []     # 在for循环后变成[ pre_prompt_embedding , image_embedding , question_embedding ]，[35,576,57]
            if use_image_text_blance:
                new_blance_image_features = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    if use_image_text_blance:
                        def cosine_similarity(a, b):
                            dot_product = torch.matmul(a, b)
                            
                            norm_a = torch.norm(a)
                            norm_b = torch.norm(b)
                            
                            similarity = dot_product / (norm_a * norm_b)
                            return similarity
                        cls_token_text = cur_input_embeds_no_im[i][0] # [4096]
                        # att_map = (cls_token_text @ image_features[cur_image_idx].T) / len(cls_token_text) # [4096] @ [4096, 576] = [576]
                        # print(cls_token_text)
                        att_map = cosine_similarity(cls_token_text, image_features[cur_image_idx].T) # [576]
                        if att_map.mean() > image_attns[cur_image_idx].mean():
                            att_map_new = att_map + image_attns[cur_image_idx] # [576]
                            # print("att_map_new.mean():", att_map_new.mean())
                            # print('both')
                        else:
                            att_map_new = image_attns[cur_image_idx]
                            # print('vis only')
                        token_indices = torch.topk(att_map_new, k=self.get_visual_token_num(), dim=0)[1]
                        token_indices_mask = torch.zeros(image_features[cur_image_idx].shape[0], dtype=torch.bool, device=image_features[cur_image_idx].device)
                        token_indices_mask[token_indices] = True
                        cur_image_features = image_features[cur_image_idx][token_indices]
                        assert num_images == 1, "Only support one image per input (Just for now 2025-04-24)"
                        new_blance_image_features.append(cur_image_features)
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features) # 之前不是没把那个<image>标识符给embedding嘛，其实就是为了这一步，把<image>标识符换成真正的image_features
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    # 把这个image_features对应的label全部设为IGNORE_INDEX
            if use_image_text_blance:
                image_features = torch.stack(new_blance_image_features, dim=0) # [B, N, 4096]
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]    

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)  # 把他们拼接成一个tensor
            cur_new_labels = torch.cat(cur_new_labels) # 把他们拼接成一个tensor

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them      下面这部分代码主要是，把一个batch内的input填充到相同的维度，比如有的input是[1,4096],有的是[668,4096],那么就将他们都填充到[668,4096]
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        #  用IGNORE_INDEX填充label，用0填充attention_mask和position_ids
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        token_length_list = []      # 这个list是为了后续把填充的token给mask掉
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            token_length_list.append(cur_len)
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left": # False
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed, # [664, 4096]
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device) # [0, 4096]
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)  # 把列表里的元素拼接

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if type(image_features) is list:
            self.image_shape = image_features[0].shape[0]
        else:
            self.image_shape = image_features.shape[1]
        self.token_length_list = token_length_list
        self.pre_prompt_length_list = pre_prompt_length_list
        self.model.init_token_total_shape = max_len       # 这个参数用于初始化policy
# attention_mask.shape：torch.Size([1, 668])，new_input_embeds.shape：torch.Size([1, 668, 4096])，new_labels.shape：torch.Size([1, 668])
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels,self.image_shape,token_length_list,pre_prompt_length_list, image_attns

def denormalize(image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    """
    对图像张量进行逆归一化。
    
    Args:
        image (torch.Tensor): 归一化后的图像张量，形状为 [C, H, W] 或 [B, C, H, W]。
        mean (list or tuple): 每个通道的均值。
        std (list or tuple): 每个通道的标准差。
    
    Returns:
        torch.Tensor: 逆归一化后的图像张量。
    """
    if image.ndimension() == 3:
        # 单张图像 [C, H, W]
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    elif image.ndimension() == 4:
        # 批量图像 [B, C, H, W]
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
    else:
        raise ValueError("Unsupported image dimensions")
    
    # 确保均值和标准差与图像在同一设备上
    mean = mean.to(image.device)
    std = std.to(image.device)
    
    return image * std + mean