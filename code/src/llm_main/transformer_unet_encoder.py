"""条件编码器。"""

import torch
import torch.nn as nn


class PhysicsViTEncoder(nn.Module):
    """用 ViT 风格编码物理场条件。"""

    def __init__(self, n_embed, img_size=64, patch_size=8, hidden_dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 输入形状为 `(bs, 4, 64, 64)`
        self.patch_embed = nn.Conv2d(4, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_embed))

    def forward(self, condition_input):
        # `condition_input` 形状为 `(bs, 4, 64, 64)`
        x = self.patch_embed(condition_input)
        x = x.flatten(2).transpose(1, 2)

        bs = x.shape[0]
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)

        cls_output = x[:, 0]
        out = self.head(cls_output)
        return out.unsqueeze(1)


class TopOptDOFEncoder(nn.Module):
    """将边界条件与载荷序列压缩为条件 token。"""

    def __init__(self, d_model=768, num_nodes=4225, compressed_len=64, num_condition_types=2, num_directions=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_condition_types = num_condition_types
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.direction_embedding = nn.Embedding(num_directions, d_model)
        self.condition_type_embedding = nn.Embedding(num_condition_types, d_model)
        self.value_encoder = nn.Linear(1, d_model)
        self.compressor = nn.AdaptiveAvgPool1d(compressed_len)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 输入形状为 `(bs, 2, max_len, 2)`，最后一维依次是 `(dof_index, value)`
        bs, num_condition_types, seq_len, _ = x.shape
        if num_condition_types != self.num_condition_types:
            raise ValueError(
                f'expected {self.num_condition_types} condition types, got {num_condition_types}'
            )

        x_flat = x.reshape(bs, -1, 2)
        raw_dof_indices = x_flat[:, :, 0].long().clamp(min=0, max=self.num_nodes * 2 - 1)
        values = x_flat[:, :, 1].unsqueeze(-1)

        # dof 按 `ux, uy` 交错存储，因此节点编号是 `dof // 2`，方向编号是 `dof % 2`
        node_ids = torch.div(raw_dof_indices, 2, rounding_mode='floor')
        direction_ids = raw_dof_indices % 2

        condition_type_ids = (
            torch.arange(num_condition_types, device=x.device)
            .view(1, num_condition_types, 1)
            .expand(bs, num_condition_types, seq_len)
            .reshape(bs, -1)
        )

        valid_mask = values.ne(0.0)

        e_node = self.node_embedding(node_ids)
        e_direction = self.direction_embedding(direction_ids)
        e_condition_type = self.condition_type_embedding(condition_type_ids)
        e_value = self.value_encoder(values)

        combined = e_node + e_direction + e_condition_type + e_value
        combined = combined * valid_mask.to(combined.dtype)

        out = combined.transpose(1, 2)
        out = self.compressor(out)
        out = out.transpose(1, 2)
        return self.layer_norm(out)


class ScalarConditionMLPEncoder(nn.Module):
    """Encode scalar material targets into prefix condition tokens."""

    def __init__(self, input_dim=4, d_model=768, condition_len=64, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, input_dim * 8)
        self.condition_len = condition_len
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_len * d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"expected condition shape (batch, features), got {tuple(x.shape)}")
        out = self.net(x)
        out = out.view(x.shape[0], self.condition_len, self.d_model)
        return self.layer_norm(out)


class ScalarConditionWithSampleIdEncoder(nn.Module):
    """Encode scalar material targets plus a sample id into prefix tokens."""

    def __init__(
        self,
        input_dim=4,
        d_model=768,
        condition_len=64,
        hidden_dim=None,
        dropout=0.0,
        max_sample_id=1024,
        sample_id_embed_dim=64,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, (input_dim + sample_id_embed_dim) * 8)
        self.condition_len = condition_len
        self.d_model = d_model
        self.sample_embedding = nn.Embedding(max_sample_id, sample_id_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + sample_id_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, condition_len * d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sample_ids=None):
        if x.ndim != 2:
            raise ValueError(f"expected condition shape (batch, features), got {tuple(x.shape)}")
        if sample_ids is None:
            raise ValueError("sample_ids are required for ScalarConditionWithSampleIdEncoder")
        sample_ids = sample_ids.to(device=x.device, dtype=torch.long)
        if sample_ids.ndim != 1:
            sample_ids = sample_ids.reshape(-1)
        sample_embed = self.sample_embedding(sample_ids)
        combined = torch.cat([x, sample_embed], dim=1)
        out = self.net(combined)
        out = out.view(x.shape[0], self.condition_len, self.d_model)
        return self.layer_norm(out)
