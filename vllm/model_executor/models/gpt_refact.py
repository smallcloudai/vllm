# coding=utf-8
"""Inference-only Refact model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
import math
from typing import List, Optional, Tuple, Iterable, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig, CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QuantizationConfig,
                                               RowParallelLinear,
                                               QKVParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsLoRA
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2,
                                  total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(start=1,
                                    end=1 + 2 * num_remaining_heads,
                                    step=2,
                                    dtype=torch.int32)
        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class LayerNormWithoutBias(nn.LayerNorm):

    def __init__(
            self,
            normalized_shape,
            eps: float = 1e-5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps, elementwise_affine=True, **factory_kwargs)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)


class RefactMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            mult: float,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        multiple_of = 256
        intermediate_size = int(2 * (hidden_size * mult) / 3)
        self.intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * self.intermediate_size,
            bias=False,
            quant_config=quant_config
        )
        self.c_proj = RowParallelLinear(
            self.intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class RefactAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        tp_rank = get_tensor_model_parallel_rank()
        self.num_heads = (self.total_num_heads //
                          self.tensor_model_parallel_world_size)
        assert self.num_heads % self.tensor_model_parallel_world_size == 0
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5
        self.num_kv_heads = 1
        self.kv_dim = self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.num_heads)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()
        self.sa = Attention(self.num_heads,
                            self.head_dim,
                            self.scaling,
                            alibi_slopes=alibi_slopes,
                            num_kv_heads=self.num_kv_heads,
                            cache_config=cache_config,
                            quant_config=quant_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.sa.forward(q, k, v, kv_cache, attn_metadata)
        output, _ = self.c_proj(attn_output)
        return output


class RefactDecoderLayer(nn.Module):

    def __init__(
            self,
            config: LlamaConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = RefactAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.mlp = RefactMLP(
            hidden_size=self.hidden_size,
            mult=4.0,
            quant_config=quant_config,
        )
        self.ln_1 = LayerNormWithoutBias(
            self.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.ln_2 = LayerNormWithoutBias(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class RefactModel(nn.Module):

    def __init__(
            self,
            config: LlamaConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(vocab_size, config.hidden_size)
        self.h = nn.ModuleList([
            RefactDecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = self.wte(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        return hidden_states


class GPTRefactForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q",
            "kv",
        ]
    }

    supported_lora_modules = [
        "q",
        "kv",
        "c_proj",
        "gate_up_proj",
        "wte",
        "lm_head",
    ]
    embedding_modules = {
        "wte": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = [
        "lm_head"
    ]

    def __init__(
            self,
            config: LlamaConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.transformer = RefactModel(config, cache_config, quant_config, lora_config=lora_config)
        self.vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.ln_f = LayerNormWithoutBias(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            bias=False,
            org_num_embeddings=self.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.vocab_size)
        self.sampler = Sampler()

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, intermediate_tensors)
        return self.ln_f(hidden_states)

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q", "q"),
            ("qkv_proj", "kv", "k"),
            ("qkv_proj", "kv", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if weight_name == "kv":
                    k_weight, v_weight = loaded_weight.split(loaded_weight.shape[0] // 2)
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(param, k_weight, "k")
                    param.weight_loader(param, v_weight, "v")
                else:
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
