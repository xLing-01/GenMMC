"""条件 GPT 模型。"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .transformer_unet_encoder import (
    ScalarConditionMLPEncoder,
    ScalarConditionWithSampleIdEncoder,
    TopOptDOFEncoder as SeqEncoder,
)


def slice_structure_logits_for_targets(logits, targets, condition_len=0):
    """Return logits aligned so slice[:, i] predicts targets[:, i]."""
    target_len = int(targets.size(1))
    end = int(condition_len) + target_len
    if logits.size(1) < end:
        raise ValueError(
            f"logits length {logits.size(1)} is shorter than condition_len + target_len = {end}"
        )
    return logits[:, int(condition_len) : end, :].contiguous()


def oh_mmc_structure_ids(idx, bos_token=513, sep_token=512, eos_token=514, pad_token=515):
    """Return parameter-type, serialized-component, and token-role ids."""
    batch_size, seq_len = idx.shape
    positions = torch.arange(seq_len, device=idx.device)
    slots = torch.remainder(torch.clamp(positions - 1, min=0), 10)
    parameter_type = torch.where(slots < 9, slots, torch.full_like(slots, 9))
    parameter_type = parameter_type.unsqueeze(0).expand(batch_size, -1).clone()

    component_index = torch.div(torch.clamp(positions - 1, min=0), 10, rounding_mode="floor") + 1
    component_index = component_index.unsqueeze(0).expand(batch_size, -1).clone()

    token_role = torch.full_like(idx, 5)
    token_role[idx.eq(int(bos_token))] = 0
    token_role[idx.ge(0) & idx.lt(int(sep_token))] = 1
    token_role[idx.eq(int(sep_token))] = 2
    token_role[idx.eq(int(eos_token))] = 3
    token_role[idx.eq(int(pad_token))] = 4

    parameter_type[token_role.ne(1)] = 9
    component_index[token_role.eq(0) | token_role.eq(3) | token_role.eq(4)] = 0
    return parameter_type.long(), component_index.long(), token_role.long()


def apply_oh_mmc_generation_grammar(
    logits,
    prefixes,
    bos_token=513,
    sep_token=512,
    eos_token=514,
    pad_token=515,
    num_parameter_bins=512,
):
    """Mask next-token logits using BOS + (9 parameters + EOC)*N + EOS grammar."""
    if logits.ndim != 2 or prefixes.ndim != 2 or logits.size(0) != prefixes.size(0):
        raise ValueError("logits and prefixes must be rank-2 tensors with equal batch size")
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    for row_index in range(prefixes.size(0)):
        row = prefixes[row_index]
        non_pad = row.ne(int(pad_token)).nonzero().flatten()
        if non_pad.numel() == 0:
            raise ValueError("OH-MMC generation prefix cannot be empty")
        row = row[: int(non_pad[-1]) + 1]
        if row[0].item() != int(bos_token):
            raise ValueError("OH-MMC generation prefix must start with BOS")
        if row.eq(int(eos_token)).any():
            allowed[row_index, int(pad_token)] = True
            continue
        slot = int(row.numel() - 1) % 10
        if slot == 9:
            allowed[row_index, int(sep_token)] = True
        else:
            allowed[row_index, : int(num_parameter_bins)] = True
            if slot == 0:
                allowed[row_index, int(eos_token)] = True
    return logits.masked_fill(~allowed, -float("inf"))


def apply_oh_mmc_teacher_grammar(
    logits,
    targets,
    condition_len=0,
    sep_token=512,
    eos_token=514,
    pad_token=515,
    num_parameter_bins=512,
):
    """Apply the OH-MMC grammar to teacher-forcing structure logits."""
    structure_logits = slice_structure_logits_for_targets(logits, targets, condition_len)
    allowed = torch.zeros_like(structure_logits, dtype=torch.bool)
    for row_index in range(targets.size(0)):
        row_targets = targets[row_index]
        separator_positions = row_targets.eq(int(sep_token)).nonzero().flatten()
        eos_positions = row_targets.eq(int(eos_token)).nonzero().flatten()
        if separator_positions.numel() > 0:
            phase_offset = (9 - int(separator_positions[0])) % 10
        elif eos_positions.numel() > 0:
            phase_offset = (-int(eos_positions[0])) % 10
        else:
            phase_offset = 0
        after_eos = False
        for position in range(targets.size(1)):
            if after_eos:
                allowed[row_index, position, int(pad_token)] = True
            else:
                slot = (position + phase_offset) % 10
                if slot == 9:
                    allowed[row_index, position, int(sep_token)] = True
                else:
                    allowed[row_index, position, : int(num_parameter_bins)] = True
                    if slot == 0:
                        allowed[row_index, position, int(eos_token)] = True
            if targets[row_index, position].item() == int(eos_token):
                after_eos = True
    masked_structure = structure_logits.masked_fill(~allowed, -float("inf"))
    prefix = logits[:, : int(condition_len), :]
    suffix_start = int(condition_len) + targets.size(1)
    suffix = logits[:, suffix_start:, :]
    return torch.cat([prefix, masked_structure, suffix], dim=1)


@torch.no_grad()
def MAE_loss(logits, targets, vocab_size=512):
    """计算物理量反归一化 MAE。"""
    device = logits.device
    preds = torch.argmax(logits, dim=-1)
    seq_len = targets.size(1)
    seq_ids = torch.arange(seq_len, device=device)

    # 每 8 个 token 为一组，第 8 个是组件终止符
    pos_mask = (seq_ids % 10) != 9
    phys_mask = (targets < vocab_size) & (targets >= 0) & pos_mask.unsqueeze(0)
    if not phys_mask.any():
        return 0.0

    v_preds = preds[phys_mask].float()
    v_targets = targets[phys_mask].float()
    norm_err = torch.abs(v_preds - v_targets) / (vocab_size - 1)
    return norm_err.mean().item()


@dataclass
class Model_args:
    """模型配置。"""

    architecture: str = "prefix_gpt"
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    label_smoothing: float = 0.0
    bias: bool = True
    UnetEncoder: bool = True
    condition_encoder_type: str = "dof"
    condition_dim: int = 4
    condition_len: int = 64
    condition_injection: str = "prefix"
    condition_dropout: float = 0.0
    cross_attention_every: int = 1
    share_cross_attention: bool = False
    use_parameter_type_embedding: bool = False
    use_component_index_embedding: bool = False
    use_token_role_embedding: bool = False
    use_grammar_mask: bool = False
    max_sample_id: int = 1024
    sample_id_embed_dim: int = 64
    memory_factor_rank: int = 128
    memory_latent_count: int = 32
    loss_ignore_index: int = -1


class RMS_Norm(nn.Module):
    """RMSNorm 层。"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        sqrt_pow_mean = torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True))
        return self.weight * hidden_states / (sqrt_pow_mean + self.eps)


class flash_att(nn.Module):
    """因果自注意力层。"""

    def __init__(self, args):
        super().__init__()
        self.qkv_atten = nn.Linear(args.n_embed, 3 * args.n_embed, bias=args.bias)
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        assert args.n_embed % args.n_head == 0
        self.head_size = args.n_embed // args.n_head
        self.dropout = args.dropout
        self.att_dropout = nn.Dropout(self.dropout)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=args.bias)

    def forward(self, x):
        bsz, seq_len, channels = x.shape
        q, k, v = self.qkv_atten(x).split(self.n_embed, dim=2)

        q = q.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.att_dropout(self.c_proj(y))

    def forward_with_kv_cache(self, x, past_key_value=None):
        """Run causal attention while returning an append-only key/value cache."""
        bsz, seq_len, channels = x.shape
        q, new_k, new_v = self.qkv_atten(x).split(self.n_embed, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        new_k = new_k.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        new_v = new_v.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        if past_key_value is None:
            key, value = new_k, new_v
            is_causal = True
        else:
            if seq_len != 1:
                raise ValueError("cached attention accepts exactly one new token")
            past_key, past_value = past_key_value
            expected = (bsz, self.n_head, past_key.size(2), self.head_size)
            if tuple(past_key.shape) != expected or tuple(past_value.shape) != expected:
                raise ValueError("past key/value cache shape is invalid")
            key = torch.cat((past_key, new_k), dim=2)
            value = torch.cat((past_value, new_v), dim=2)
            is_causal = False
        y = F.scaled_dot_product_attention(
            q, key, value, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.att_dropout(self.c_proj(y)), (key, value)

    def forward_with_preallocated_kv(
        self,
        x,
        key_buffer=None,
        value_buffer=None,
        *,
        current_length=0,
        capacity=None,
    ):
        """Append projected K/V in-place to a fixed-capacity inference buffer."""
        bsz, seq_len, channels = x.shape
        q, new_k, new_v = self.qkv_atten(x).split(self.n_embed, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        new_k = new_k.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        new_v = new_v.view(bsz, seq_len, self.n_head, self.head_size).transpose(1, 2)
        if key_buffer is None or value_buffer is None:
            if current_length != 0 or capacity is None or capacity < seq_len:
                raise ValueError("invalid initial preallocated KV contract")
            shape = (bsz, self.n_head, int(capacity), self.head_size)
            key_buffer = torch.empty(shape, dtype=new_k.dtype, device=new_k.device)
            value_buffer = torch.empty(shape, dtype=new_v.dtype, device=new_v.device)
            is_causal = True
        else:
            if seq_len != 1:
                raise ValueError("preallocated cached attention accepts one new token")
            if key_buffer.shape != value_buffer.shape or key_buffer.size(0) != bsz:
                raise ValueError("preallocated KV buffer shape mismatch")
            capacity = int(key_buffer.size(2))
            is_causal = False
        end = int(current_length) + seq_len
        if end > int(capacity):
            raise ValueError("transformer_context_exhausted")
        key_buffer[:, :, current_length:end, :].copy_(new_k)
        value_buffer[:, :, current_length:end, :].copy_(new_v)
        key = key_buffer[:, :, :end, :]
        value = value_buffer[:, :, :end, :]
        y = F.scaled_dot_product_attention(
            q, key, value, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.att_dropout(self.c_proj(y)), (key_buffer, value_buffer)


class MLP(nn.Module):
    """带门控的前馈网络。"""

    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.up_proj = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)
        self.down_c_proj = nn.Linear(4 * args.n_embed, args.n_embed, bias=args.bias)
        self.act_func = nn.functional.relu
        self.gate = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)

    def forward(self, x):
        gate_proj = self.gate(x)
        x = self.up_proj(x)
        x = self.act_func(gate_proj) * x
        x = self.down_c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer 残差块。"""

    def __init__(self, args):
        super().__init__()
        self.norm = RMS_Norm(args.n_embed)
        self.attn = flash_att(args)
        self.mlp = MLP(args)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x + self.mlp(self.norm(x))

    def forward_with_kv_cache(self, x, past_key_value=None):
        attention, present = self.attn.forward_with_kv_cache(
            self.norm(x), past_key_value
        )
        x = x + attention
        return x + self.mlp(self.norm(x)), present

    def forward_with_preallocated_kv(
        self, x, key_buffer=None, value_buffer=None, *, current_length=0, capacity=None
    ):
        attention, present = self.attn.forward_with_preallocated_kv(
            self.norm(x), key_buffer, value_buffer,
            current_length=current_length, capacity=capacity,
        )
        x = x + attention
        return x + self.mlp(self.norm(x)), present


class PropertyTokenEncoder(nn.Module):
    """Encode each scalar condition independently and preserve its property identity."""

    def __init__(self, condition_dim, d_model, dropout):
        super().__init__()
        hidden_dim = max(16, d_model // 4)
        self.property_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, d_model),
                )
                for _ in range(condition_dim)
            ]
        )
        self.property_type_embedding = nn.Embedding(condition_dim, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(condition_dim * d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, condition):
        if condition.ndim != 2 or condition.size(1) != len(self.property_encoders):
            raise ValueError(
                f"condition must have shape (batch, {len(self.property_encoders)}), "
                f"got {tuple(condition.shape)}"
            )
        encoded = [encoder(condition[:, index : index + 1]) for index, encoder in enumerate(self.property_encoders)]
        tokens = torch.stack(encoded, dim=1)
        property_ids = torch.arange(condition.size(1), device=condition.device)
        tokens = self.token_norm(tokens + self.property_type_embedding(property_ids)[None, :, :])
        return self.dropout(tokens)

    def fused_state(self, property_tokens):
        return self.fusion(property_tokens.flatten(start_dim=1))


class PropertyConditionedBlock(nn.Module):
    """Pre-norm causal block with optional property cross-attention and AdaLN."""

    def __init__(self, args, use_cross_attention, use_adaln):
        super().__init__()
        self.norm_attn = RMS_Norm(args.n_embed)
        self.norm_cross = RMS_Norm(args.n_embed) if use_cross_attention else None
        self.norm_mlp = RMS_Norm(args.n_embed)
        self.attn = flash_att(args)
        self.cross_attn = (
            nn.MultiheadAttention(
                args.n_embed,
                args.n_head,
                dropout=args.dropout,
                bias=args.bias,
                batch_first=True,
            )
            if use_cross_attention
            else None
        )
        self.cross_gate = nn.Parameter(torch.tensor(0.1)) if use_cross_attention else None
        self.mlp = MLP(args)
        self.cond_modulation = (
            nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias) if use_adaln else None
        )

    def reset_condition_parameters(self):
        if self.cond_modulation is not None:
            nn.init.zeros_(self.cond_modulation.weight)
            if self.cond_modulation.bias is not None:
                nn.init.zeros_(self.cond_modulation.bias)

    def forward(self, x, property_tokens, condition_state):
        attn_input = self.norm_attn(x)
        if self.cond_modulation is not None:
            shift_attn, scale_attn, shift_mlp, scale_mlp = self.cond_modulation(condition_state).chunk(4, dim=-1)
            attn_input = attn_input * (1 + scale_attn[:, None, :]) + shift_attn[:, None, :]
        x = x + self.attn(attn_input)
        if self.cross_attn is not None:
            cross_input = self.norm_cross(x)
            cross_output, _ = self.cross_attn(
                cross_input,
                property_tokens,
                property_tokens,
                need_weights=False,
            )
            x = x + self.cross_gate * cross_output
        mlp_input = self.norm_mlp(x)
        if self.cond_modulation is not None:
            mlp_input = mlp_input * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        return x + self.mlp(mlp_input)


class MemoryConditionedBlock(nn.Module):
    """Causal Transformer block with per-layer gated AdaLN conditioning."""

    def __init__(self, args):
        super().__init__()
        self.norm_attn = RMS_Norm(args.n_embed)
        self.norm_mlp = RMS_Norm(args.n_embed)
        self.attn = flash_att(args)
        self.mlp = MLP(args)
        self.cond_proj = nn.Linear(args.n_embed, 6 * args.n_embed, bias=args.bias)

    def forward(self, x, condition_state):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.cond_proj(condition_state).chunk(6, dim=-1)
        attn_in = self.norm_attn(x) * (1 + scale_a[:, None, :]) + shift_a[:, None, :]
        x = x + torch.tanh(gate_a)[:, None, :] * self.attn(attn_in)
        mlp_in = self.norm_mlp(x) * (1 + scale_m[:, None, :]) + shift_m[:, None, :]
        return x + torch.tanh(gate_m)[:, None, :] * self.mlp(mlp_in)


class GPT(nn.Module):
    """带条件编码器的 GPT。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_memory_architecture = args.architecture in {
            "memory_ar",
            "memory_direct_ar",
            "memory_table_ar",
            "memory_factor_ar",
            "memory_latent_ar",
        }
        self.condition_injection = args.condition_injection
        valid_condition_injections = {"prefix", "cross_attention", "adaln", "cross_attention_adaln"}
        if self.condition_injection not in valid_condition_injections:
            raise ValueError(
                f"unknown condition_injection={self.condition_injection!r}; "
                f"expected one of {sorted(valid_condition_injections)}"
            )
        if args.cross_attention_every < 1:
            raise ValueError("cross_attention_every must be >= 1")
        self.logit_condition_len = (
            int(args.condition_len)
            if not self.is_memory_architecture and self.condition_injection == "prefix"
            else 0
        )
        self.condition_encoder = None
        if self.is_memory_architecture:
            self.UnetEncoder = None
        elif self.condition_injection != "prefix":
            self.UnetEncoder = None
            self.condition_encoder = PropertyTokenEncoder(
                condition_dim=args.condition_dim,
                d_model=args.n_embed,
                dropout=args.condition_dropout,
            )
        elif not args.UnetEncoder:
            self.UnetEncoder = None
        elif args.condition_encoder_type == "mlp":
            self.UnetEncoder = ScalarConditionMLPEncoder(
                input_dim=args.condition_dim,
                d_model=args.n_embed,
                condition_len=args.condition_len,
                dropout=args.dropout,
            )
        elif args.condition_encoder_type == "mlp_sample_id":
            self.UnetEncoder = ScalarConditionWithSampleIdEncoder(
                input_dim=args.condition_dim,
                d_model=args.n_embed,
                condition_len=args.condition_len,
                dropout=args.dropout,
                max_sample_id=args.max_sample_id,
                sample_id_embed_dim=args.sample_id_embed_dim,
            )
        elif args.condition_encoder_type == "dof":
            self.UnetEncoder = SeqEncoder(d_model=args.n_embed, compressed_len=args.condition_len)
        else:
            raise ValueError(f"unknown condition_encoder_type: {args.condition_encoder_type}")
        if self.is_memory_architecture:
            max_rods = args.block_size // 10 + 2
            modules = dict(
                wte=nn.Embedding(args.vocab_size, args.n_embed),
                wpe=nn.Embedding(args.block_size + 1, args.n_embed),
                rod_embedding=nn.Embedding(max_rods, args.n_embed),
                slot_embedding=nn.Embedding(11, args.n_embed),
                sample_code_embedding=nn.Embedding(args.max_sample_id, args.n_embed),
                condition_vector=nn.Sequential(
                    nn.Linear(args.condition_dim, args.n_embed, bias=args.bias),
                    nn.GELU(),
                    nn.Linear(args.n_embed, args.n_embed, bias=args.bias),
                    nn.LayerNorm(args.n_embed),
                ),
                sample_code_norm=nn.LayerNorm(args.n_embed),
                drop=nn.Dropout(args.dropout),
                h=nn.ModuleList([MemoryConditionedBlock(args) for _ in range(args.n_layer)]),
                norm=RMS_Norm(args.n_embed),
            )
            if args.architecture == "memory_direct_ar":
                modules.update(
                    dict(
                        sample_rod_embedding=nn.Embedding(args.max_sample_id * max_rods, args.n_embed),
                        direct_memory=nn.Sequential(
                            nn.LayerNorm(args.n_embed),
                            nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias),
                            nn.GELU(),
                            nn.Linear(4 * args.n_embed, args.n_embed, bias=args.bias),
                            RMS_Norm(args.n_embed),
                        ),
                        direct_lm_head=nn.Linear(args.n_embed, args.vocab_size, bias=False),
                    )
                )
                self.direct_logit_scale = nn.Parameter(torch.tensor(1.0))
            elif args.architecture == "memory_table_ar":
                modules.update(
                    dict(
                        sample_rod_slot_logits=nn.Embedding(args.max_sample_id * max_rods * 11, args.vocab_size),
                    )
                )
                self.table_logit_scale = nn.Parameter(torch.tensor(1.0))
            elif args.architecture == "memory_factor_ar":
                modules.update(
                    dict(
                        factor_sample_embedding=nn.Embedding(args.max_sample_id, args.memory_factor_rank),
                        factor_rod_embedding=nn.Embedding(max_rods, args.memory_factor_rank),
                        factor_slot_embedding=nn.Embedding(11, args.memory_factor_rank),
                        factor_norm=nn.LayerNorm(args.memory_factor_rank),
                        factor_lm_head=nn.Linear(args.memory_factor_rank, args.vocab_size, bias=False),
                    )
                )
                self.factor_logit_scale = nn.Parameter(torch.tensor(1.0))
            elif args.architecture == "memory_latent_ar":
                modules.update(
                    dict(
                        sample_latent_memory=nn.Embedding(args.max_sample_id * args.memory_latent_count, args.n_embed),
                        latent_query=nn.Sequential(
                            nn.LayerNorm(args.n_embed),
                            nn.Linear(args.n_embed, args.n_embed, bias=args.bias),
                            nn.GELU(),
                            nn.Linear(args.n_embed, args.n_embed, bias=args.bias),
                        ),
                        latent_key=nn.Linear(args.n_embed, args.n_embed, bias=args.bias),
                        latent_value=nn.Linear(args.n_embed, args.n_embed, bias=args.bias),
                        latent_out=nn.Sequential(
                            nn.LayerNorm(args.n_embed),
                            nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias),
                            nn.GELU(),
                            nn.Linear(4 * args.n_embed, args.n_embed, bias=args.bias),
                            RMS_Norm(args.n_embed),
                        ),
                        latent_lm_head=nn.Linear(args.n_embed, args.vocab_size, bias=False),
                    )
                )
                self.latent_logit_scale = nn.Parameter(torch.tensor(1.0))
            self.transformer = nn.ModuleDict(modules)
        else:
            use_conditioned_blocks = self.condition_injection != "prefix"
            use_cross_attention = self.condition_injection in {"cross_attention", "cross_attention_adaln"}
            use_adaln = self.condition_injection in {"adaln", "cross_attention_adaln"}
            blocks = []
            for layer_index in range(args.n_layer):
                if use_conditioned_blocks:
                    blocks.append(
                        PropertyConditionedBlock(
                            args,
                            use_cross_attention=(
                                use_cross_attention and layer_index % args.cross_attention_every == 0
                            ),
                            use_adaln=use_adaln,
                        )
                    )
                else:
                    blocks.append(Block(args))
            if use_cross_attention and args.share_cross_attention:
                shared_cross_attention = next(
                    block.cross_attn
                    for block in blocks
                    if isinstance(block, PropertyConditionedBlock) and block.cross_attn is not None
                )
                for block in blocks:
                    if isinstance(block, PropertyConditionedBlock) and block.cross_attn is not None:
                        block.cross_attn = shared_cross_attention
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(args.vocab_size, args.n_embed),
                    # 为条件前缀预留 64 个位置编码
                    wpe=nn.Embedding(args.block_size + args.condition_len, args.n_embed),
                    drop=nn.Dropout(args.dropout),
                    h=nn.ModuleList(blocks),
                    norm=RMS_Norm(args.n_embed),
                )
            )
            if args.use_parameter_type_embedding:
                self.transformer["parameter_type_embedding"] = nn.Embedding(10, args.n_embed)
            if args.use_component_index_embedding:
                self.transformer["component_index_embedding"] = nn.Embedding(
                    args.block_size // 10 + 3,
                    args.n_embed,
                )
            if args.use_token_role_embedding:
                self.transformer["token_role_embedding"] = nn.Embedding(6, args.n_embed)
        self.lm_head = nn.Linear(args.n_embed, args.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for block in self.transformer.h:
            if isinstance(block, PropertyConditionedBlock):
                block.reset_condition_parameters()
        n_sum = 0
        for pname, p in self.named_parameters():
            n_sum += p.numel()
            if pname.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layer))
        print(f"model parameter count: {n_sum}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _structure_token_embedding(self, idx):
        embedding = self.transformer.wte(idx)
        if not any(
            name in self.transformer
            for name in (
                "parameter_type_embedding",
                "component_index_embedding",
                "token_role_embedding",
            )
        ):
            return embedding
        parameter_type, component_index, token_role = oh_mmc_structure_ids(idx)
        if "parameter_type_embedding" in self.transformer:
            embedding = embedding + self.transformer.parameter_type_embedding(parameter_type)
        if "component_index_embedding" in self.transformer:
            component_index = component_index.clamp(
                max=self.transformer.component_index_embedding.num_embeddings - 1
            )
            embedding = embedding + self.transformer.component_index_embedding(component_index)
        if "token_role_embedding" in self.transformer:
            embedding = embedding + self.transformer.token_role_embedding(token_role)
        return embedding

    def _structure_position_embedding(self, idx):
        device = idx.device
        seq_len = idx.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        rod_ids = torch.div(torch.clamp(positions - 1, min=0), 10, rounding_mode="floor")
        rod_ids = rod_ids.clamp(max=self.transformer.rod_embedding.num_embeddings - 1)
        slot_ids = torch.where(positions == 0, torch.full_like(positions, 10), (positions - 1) % 10)
        return (
            self.transformer.wpe(positions)
            + self.transformer.rod_embedding(rod_ids)
            + self.transformer.slot_embedding(slot_ids)
        )

    def _memory_condition_state(self, condition, sample_ids, device, batch_size):
        pieces = []
        if condition is not False and condition is not None:
            pieces.append(self.transformer.condition_vector(condition.to(torch.float32).to(device)))
        if sample_ids is not None:
            sample_ids = sample_ids.to(device=device, dtype=torch.long).reshape(-1)
            pieces.append(self.transformer.sample_code_norm(self.transformer.sample_code_embedding(sample_ids)))
        if not pieces:
            return torch.zeros((batch_size, self.args.n_embed), dtype=self.transformer.wte.weight.dtype, device=device)
        state = pieces[0]
        for piece in pieces[1:]:
            state = state + piece
        return state

    def _direct_memory_logits(self, idx, condition_state, sample_ids):
        device = idx.device
        batch_size, seq_len = idx.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        rod_ids = torch.div(torch.clamp(positions - 1, min=0), 10, rounding_mode="floor")
        rod_ids = rod_ids.clamp(max=self.transformer.rod_embedding.num_embeddings - 1)
        direct_state = self._structure_position_embedding(idx) + condition_state[:, None, :]
        if sample_ids is not None:
            sample_ids = sample_ids.to(device=device, dtype=torch.long).reshape(-1)
            sample_rod_ids = sample_ids[:, None] * self.transformer.rod_embedding.num_embeddings + rod_ids[None, :]
            direct_state = direct_state + self.transformer.sample_rod_embedding(sample_rod_ids)
        direct_state = self.transformer.direct_memory(direct_state)
        return self.direct_logit_scale * self.transformer.direct_lm_head(direct_state)

    def _memory_table_logits(self, idx, sample_ids):
        if sample_ids is None:
            return torch.zeros(
                (*idx.shape, self.args.vocab_size),
                dtype=self.transformer.wte.weight.dtype,
                device=idx.device,
            )
        device = idx.device
        _, seq_len = idx.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        max_rods = self.transformer.rod_embedding.num_embeddings
        rod_ids = torch.div(torch.clamp(positions - 1, min=0), 10, rounding_mode="floor").clamp(max=max_rods - 1)
        slot_ids = torch.where(positions == 0, torch.full_like(positions, 10), (positions - 1) % 10)
        sample_ids = sample_ids.to(device=device, dtype=torch.long).reshape(-1)
        table_ids = (sample_ids[:, None] * max_rods + rod_ids[None, :]) * 11 + slot_ids[None, :]
        return self.table_logit_scale * self.transformer.sample_rod_slot_logits(table_ids)

    def _memory_factor_logits(self, idx, sample_ids):
        if sample_ids is None:
            return torch.zeros(
                (*idx.shape, self.args.vocab_size),
                dtype=self.transformer.wte.weight.dtype,
                device=idx.device,
            )
        device = idx.device
        _, seq_len = idx.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        max_rods = self.transformer.rod_embedding.num_embeddings
        rod_ids = torch.div(torch.clamp(positions - 1, min=0), 10, rounding_mode="floor").clamp(max=max_rods - 1)
        slot_ids = torch.where(positions == 0, torch.full_like(positions, 10), (positions - 1) % 10)
        sample_ids = sample_ids.to(device=device, dtype=torch.long).reshape(-1)
        factor_state = (
            self.transformer.factor_sample_embedding(sample_ids)[:, None, :]
            + self.transformer.factor_rod_embedding(rod_ids)[None, :, :]
            + self.transformer.factor_slot_embedding(slot_ids)[None, :, :]
        )
        factor_state = self.transformer.factor_norm(factor_state)
        return self.factor_logit_scale * self.transformer.factor_lm_head(factor_state)

    def _memory_latent_logits(self, idx, condition_state, sample_ids):
        if sample_ids is None:
            return torch.zeros(
                (*idx.shape, self.args.vocab_size),
                dtype=self.transformer.wte.weight.dtype,
                device=idx.device,
            )
        device = idx.device
        batch_size, _ = idx.size()
        latent_count = int(self.args.memory_latent_count)
        latent_offsets = torch.arange(0, latent_count, dtype=torch.long, device=device)
        sample_ids = sample_ids.to(device=device, dtype=torch.long).reshape(-1)
        latent_ids = sample_ids[:, None] * latent_count + latent_offsets[None, :]
        sample_latents = self.transformer.sample_latent_memory(latent_ids)

        query_state = self._structure_position_embedding(idx) + condition_state[:, None, :]
        query = self.transformer.latent_query(query_state)
        key = self.transformer.latent_key(sample_latents)
        value = self.transformer.latent_value(sample_latents)
        attn = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1)),
            dim=-1,
        )
        latent_state = torch.matmul(attn, value)
        latent_state = self.transformer.latent_out(latent_state + query_state)
        return self.latent_logit_scale * self.transformer.latent_lm_head(latent_state)

    def _forward_memory_ar(self, idx, targets=None, condition=None, sample_ids=None):
        device = idx.device
        batch_size, seq_len = idx.size()
        if seq_len > self.args.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size={self.args.block_size}")
        token_embed = self.transformer.wte(idx)
        pos_embed = self._structure_position_embedding(idx)
        x = self.transformer.drop(token_embed + pos_embed)
        condition_state = self._memory_condition_state(condition, sample_ids, device, batch_size)

        for block in self.transformer.h:
            x = block(x, condition_state)
        x = self.transformer.norm(x)
        logits = self.lm_head(x)
        if self.args.architecture == "memory_direct_ar":
            logits = logits + self._direct_memory_logits(idx, condition_state, sample_ids)
        elif self.args.architecture == "memory_table_ar":
            logits = logits + self._memory_table_logits(idx, sample_ids)
        elif self.args.architecture == "memory_factor_ar":
            logits = logits + self._memory_factor_logits(idx, sample_ids)
        elif self.args.architecture == "memory_latent_ar":
            logits = logits + self._memory_latent_logits(idx, condition_state, sample_ids)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.to(torch.int64).view(-1),
                ignore_index=self.args.loss_ignore_index,
                label_smoothing=self.args.label_smoothing,
            )
            mae_loss = torch.tensor(MAE_loss(logits, targets), device=loss.device)
        else:
            loss = None
            mae_loss = None
        return logits, loss, mae_loss

    def forward(self, idx, targets=None, condition=None, sample_ids=None):
        if self.is_memory_architecture:
            return self._forward_memory_ar(idx, targets=targets, condition=condition, sample_ids=sample_ids)

        device = idx.device

        if self.condition_injection != "prefix":
            if condition is False or condition is None:
                raise ValueError(f"condition is required for condition_injection={self.condition_injection}")
            condition = condition.to(torch.float32).to(device)
            property_tokens = self.condition_encoder(condition)
            condition_state = self.condition_encoder.fused_state(property_tokens)
            _, seq_len = idx.size()
            if seq_len > self.args.block_size:
                raise ValueError(f"sequence length {seq_len} exceeds block_size={self.args.block_size}")
            positions = torch.arange(seq_len, dtype=torch.long, device=device)
            x = self.transformer.drop(
                self._structure_token_embedding(idx) + self.transformer.wpe(positions)
            )
            for block in self.transformer.h:
                x = block(x, property_tokens, condition_state)
            logits = self.lm_head(self.transformer.norm(x))
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.to(torch.int64).reshape(-1),
                    ignore_index=self.args.loss_ignore_index,
                    label_smoothing=self.args.label_smoothing,
                )
                mae_loss = torch.tensor(MAE_loss(logits, targets), device=loss.device)
            else:
                loss = None
                mae_loss = None
            return logits, loss, mae_loss

        # 先将条件序列编码为前缀 token
        if condition is not False and condition is not None:
            if self.UnetEncoder is None:
                raise ValueError('condition is provided but UnetEncoder is disabled')
            condition = condition.to(torch.float32).to(device)
            if self.args.condition_encoder_type == "mlp_sample_id":
                condition_encoded = self.UnetEncoder(condition, sample_ids=sample_ids)
            else:
                condition_encoded = self.UnetEncoder(condition)
            _, seq_len = idx.size()
            cond_len = condition_encoded.size(1)

            token_embed = self._structure_token_embedding(idx)
            pos = torch.arange(0, cond_len + seq_len, dtype=torch.long, device=device)
            pos_embed = self.transformer.wpe(pos)
            full_token_embed = torch.cat([condition_encoded, token_embed], dim=1)
            x = self.transformer.drop(full_token_embed + pos_embed)

            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.norm(x)
            logits = self.lm_head(x)

            # 只对结构 token 计算损失
            if targets is not None:
                shift_logits = slice_structure_logits_for_targets(logits, targets, cond_len)
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    targets.to(torch.int64).view(-1),
                    ignore_index=self.args.loss_ignore_index,
                    label_smoothing=self.args.label_smoothing,
                )
                mae_loss = torch.tensor(MAE_loss(shift_logits, targets), device=loss.device)
            else:
                loss = None
                mae_loss = None
            return logits, loss, mae_loss

        _, seq_len = idx.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        token_embed = self._structure_token_embedding(idx)
        pos_embed = self.transformer.wpe(pos)
        x = self.transformer.drop(token_embed + pos_embed)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.to(torch.int64).view(-1),
                ignore_index=self.args.loss_ignore_index,
                label_smoothing=self.args.label_smoothing,
            )
            mae_loss = MAE_loss(logits, targets)
        else:
            loss = None
            mae_loss = None
        return logits, loss, mae_loss

    def prepare_prefix_kv_cache(self, idx, condition):
        """Consume a prefix-conditioned initial structure sequence once.

        This inference-only API is intentionally limited to the unmodified
        ``prefix_gpt`` baseline.  It adds no modules or parameters and therefore
        cannot alter training behavior or checkpoint keys.
        """
        if self.training:
            raise ValueError("KV-cached generation requires model.eval()")
        if self.is_memory_architecture or self.condition_injection != "prefix":
            raise ValueError("KV-cached generation supports prefix_gpt only")
        if self.UnetEncoder is None or condition is None or condition is False:
            raise ValueError("prefix_gpt KV generation requires a condition encoder")
        if any(
            name in self.transformer
            for name in (
                "parameter_type_embedding",
                "component_index_embedding",
                "token_role_embedding",
            )
        ):
            raise ValueError("KV adapter does not support extra structure-role embeddings")
        if idx.ndim != 2 or idx.size(1) < 1:
            raise ValueError("initial idx must have shape [batch, sequence>=1]")
        if idx.size(1) > self.args.block_size:
            raise ValueError("initial structure sequence exceeds block_size")
        device = idx.device
        condition = condition.to(device=device, dtype=torch.float32)
        if self.args.condition_encoder_type == "mlp_sample_id":
            raise ValueError("sample-id condition encoders are not supported by this adapter")
        condition_encoded = self.UnetEncoder(condition)
        cond_len = int(condition_encoded.size(1))
        positions = torch.arange(
            cond_len + idx.size(1), dtype=torch.long, device=device
        )
        inputs = torch.cat(
            (condition_encoded, self._structure_token_embedding(idx)), dim=1
        )
        x = self.transformer.drop(inputs + self.transformer.wpe(positions))
        layers = []
        capacity = cond_len + int(self.args.block_size)
        for block in self.transformer.h:
            x, present = block.forward_with_preallocated_kv(
                x, current_length=0, capacity=capacity
            )
            layers.append(present)
        logits = self.lm_head(self.transformer.norm(x[:, -1:, :]))[:, 0, :]
        return logits, tuple(layers), cond_len, cond_len + idx.size(1), capacity

    def step_prefix_kv_cache(
        self, input_token, layers, condition_len, current_length, capacity
    ):
        """Consume one structure token using cache returned by prepare."""
        if self.training:
            raise ValueError("KV-cached generation requires model.eval()")
        if input_token.ndim == 1:
            idx = input_token[:, None]
        elif input_token.ndim == 2 and input_token.size(1) == 1:
            idx = input_token
        else:
            raise ValueError("input_token must have shape [batch] or [batch,1]")
        if len(layers) != len(self.transformer.h) or not layers:
            raise ValueError("KV cache layer count does not match model")
        cache_length = int(current_length)
        if any(
            int(key.size(2)) != int(capacity) or value.shape != key.shape
            for key, value in layers
        ):
            raise ValueError("KV cache layers have inconsistent capacities")
        structure_position = cache_length - int(condition_len)
        if structure_position < 1 or structure_position >= int(self.args.block_size):
            raise ValueError("transformer_context_exhausted")
        position = torch.tensor([cache_length], dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self._structure_token_embedding(idx) + self.transformer.wpe(position)
        )
        new_layers = []
        for block, past in zip(self.transformer.h, layers):
            x, present = block.forward_with_preallocated_kv(
                x, past[0], past[1], current_length=cache_length,
                capacity=capacity,
            )
            new_layers.append(present)
        logits = self.lm_head(self.transformer.norm(x))[:, 0, :]
        return logits, tuple(new_layers), cache_length + 1

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """构建 AdamW 参数组。"""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"weight decay params: {num_decay}, no-decay params: {num_nodecay}")

        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device_type == 'cuda'
        if use_fused:
            print('AdamW optimiser use fused!')
        extra_args = {'fused': True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    def generate(
        self,
        idx,
        condition,
        max_generate_tokens,
        temperature=1.0,
        top_k=None,
        eos_token=None,
        sample_ids=None,
        pad_token=None,
        grammar_mask=None,
    ):
        """在条件约束下自回归生成 token。"""
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_generate_tokens):
            idx_cond = idx if idx.size(1) <= self.args.block_size else idx[:, -self.args.block_size:]
            logits, _, _ = self(idx_cond, condition=condition, sample_ids=sample_ids)
            logits = logits[:, -1, :] / temperature

            use_grammar_mask = getattr(self.args, "use_grammar_mask", False) if grammar_mask is None else grammar_mask
            if use_grammar_mask:
                if eos_token is None:
                    raise ValueError("eos_token is required when OH-MMC grammar masking is enabled")
                logits = apply_oh_mmc_generation_grammar(
                    logits,
                    idx,
                    eos_token=eos_token,
                    pad_token=515 if pad_token is None else pad_token,
                )

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if eos_token is not None:
                fill_token = eos_token if pad_token is None else pad_token
                idx_next = torch.where(
                    finished[:, None],
                    torch.full_like(idx_next, int(fill_token)),
                    idx_next,
                )
                finished |= idx_next.squeeze(1).eq(int(eos_token))
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token is not None and torch.all(finished):
                break
        return idx
