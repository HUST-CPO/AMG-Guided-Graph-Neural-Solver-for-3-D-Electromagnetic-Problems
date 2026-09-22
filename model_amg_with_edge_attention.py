import torch
import torch.nn as nn
import os
from torch_geometric.nn import MessagePassing
from training.config import N_ITER
from training.config import NETWORK_USE_CUSTOM_DIMS, NETWORK_BASE_DIM, NETWORK_CUSTOM_DIMS, NETWORK_USE_GEOMETRIC_EDGE_FEATURES, EDGE_ALGEBRAIC_FEATURE_DIM, NETWORK_USE_FREQUENCY_MODULATION, FREQUENCY_INPUT_SCALE, FREQUENCY_NORM_MIN, FREQUENCY_NORM_MAX, FREQUENCY_USE_FOURIER_FEATURES, FREQUENCY_EMBED_DIM, FREQUENCY_FILM_STRENGTH
EDGE_MLP_FP32 = os.environ.get('FEMGRL_EDGE_MLP_FP32', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    rank = os.environ.get('RANK')
    if rank is not None:
        return int(rank) == 0
    return True

def get_batch_norm(num_features):
    return nn.LayerNorm(num_features)

class FrequencyEncoder(nn.Module):

    def __init__(self, embed_dim=FREQUENCY_EMBED_DIM):
        super(FrequencyEncoder, self).__init__()
        input_dim = 3 if FREQUENCY_USE_FOURIER_FEATURES else 1
        hidden_dim = max(16, embed_dim // 2)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim), nn.GELU())

    def forward(self, freq_hz):
        freq_hz = freq_hz.reshape(freq_hz.size(0), -1)[:, :1]
        freq_scaled = freq_hz / float(FREQUENCY_INPUT_SCALE)
        denom = max(float(FREQUENCY_NORM_MAX) - float(FREQUENCY_NORM_MIN), 1e-12)
        freq_norm = 2.0 * (freq_scaled - float(FREQUENCY_NORM_MIN)) / denom - 1.0
        if FREQUENCY_USE_FOURIER_FEATURES:
            features = torch.cat([freq_norm, torch.sin(torch.pi * freq_norm), torch.cos(torch.pi * freq_norm)], dim=-1)
        else:
            features = freq_norm
        return self.net(features)

class FrequencyFiLM(nn.Module):

    def __init__(self, embed_dim, num_channels):
        super(FrequencyFiLM, self).__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * num_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.to_gamma_beta.weight, mean=0.0, std=0.0001)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, freq_emb_node):
        (gamma, beta) = self.to_gamma_beta(freq_emb_node).chunk(2, dim=-1)
        scale = 1.0 + float(FREQUENCY_FILM_STRENGTH) * torch.tanh(gamma)
        shift = float(FREQUENCY_FILM_STRENGTH) * beta
        return x * scale + shift

class EMFullComplexLayer(MessagePassing):

    def __init__(self, in_feats, out_feats):
        super(EMFullComplexLayer, self).__init__(aggr='mean')
        total_in_dim = in_feats + in_feats
        self.lin_fusion = nn.Linear(total_in_dim, out_feats)
        self.edge_feature_dim = EDGE_ALGEBRAIC_FEATURE_DIM + (3 if NETWORK_USE_GEOMETRIC_EDGE_FEATURES else 0)
        self.edge_mlp = nn.Sequential(nn.Linear(self.edge_feature_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr=None, edge_dir=None, edge_length=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_dir=edge_dir, edge_length=edge_length)

    def message(self, x_j, edge_attr=None, edge_dir_i=None, edge_dir_j=None, edge_length_i=None, edge_length_j=None):
        dtype = x_j.dtype
        device = x_j.device
        n_edges = x_j.size(0)
        edge_dtype = torch.float32 if EDGE_MLP_FP32 else dtype
        if edge_attr is None:
            alg_feats = torch.zeros((n_edges, EDGE_ALGEBRAIC_FEATURE_DIM), dtype=edge_dtype, device=device)
        else:
            alg_feats = edge_attr.to(device=device, dtype=edge_dtype)
            if alg_feats.size(1) < EDGE_ALGEBRAIC_FEATURE_DIM:
                pad = torch.zeros((n_edges, EDGE_ALGEBRAIC_FEATURE_DIM - alg_feats.size(1)), dtype=edge_dtype, device=device)
                alg_feats = torch.cat([alg_feats, pad], dim=-1)
            else:
                alg_feats = alg_feats[:, :EDGE_ALGEBRAIC_FEATURE_DIM]
        if NETWORK_USE_GEOMETRIC_EDGE_FEATURES:
            if edge_dir_i is not None and edge_dir_j is not None:
                cos_theta = (edge_dir_i * edge_dir_j).sum(dim=-1, keepdim=True).to(edge_dtype)
            else:
                cos_theta = torch.zeros((n_edges, 1), dtype=edge_dtype, device=device)
            if edge_length_i is not None:
                l_i = edge_length_i.to(edge_dtype)
            else:
                l_i = torch.ones((n_edges, 1), dtype=edge_dtype, device=device)
            if edge_length_j is not None:
                l_j = edge_length_j.to(edge_dtype)
            else:
                l_j = torch.ones((n_edges, 1), dtype=edge_dtype, device=device)
            geom_feats = torch.cat([cos_theta, l_i, l_j], dim=-1)
            edge_feats = torch.cat([alg_feats, geom_feats], dim=-1)
        else:
            edge_feats = alg_feats
        if EDGE_MLP_FP32:
            with torch.autocast(device_type=device.type, enabled=False):
                attn_weight = torch.sigmoid(self.edge_mlp(edge_feats.float()))
            attn_weight = attn_weight.to(dtype=dtype)
        else:
            attn_weight = torch.sigmoid(self.edge_mlp(edge_feats))
        return x_j * attn_weight

    def update(self, aggs, x):
        combined = torch.cat([x, aggs], dim=1)
        fused_msg = self.lin_fusion(combined)
        return fused_msg

def make_edge_attention_layer(in_feats, out_feats):
    return EMFullComplexLayer(in_feats, out_feats)

class GCN(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.conv = make_edge_attention_layer(in_feats, out_feats)
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()

    def forward(self, x, edge_index, edge_attr=None, edge_dir=None, edge_length=None):
        x = self.conv(x, edge_index, edge_attr=edge_attr, edge_dir=edge_dir, edge_length=edge_length)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class FFTFeatureLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(FFTFeatureLayer, self).__init__()
        self.lin_in = nn.Linear(in_feats, out_feats)
        self.freq_dim = out_feats // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(self.freq_dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        x = self.lin_in(x)
        original_dtype = x.dtype
        if x.dtype in {torch.float16, torch.bfloat16}:
            x = x.to(torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fft = torch.fft.rfft(x, dim=-1)
            weight = torch.view_as_complex(self.complex_weight)
            x_fft = x_fft * weight
            x_out = torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)
        if original_dtype in {torch.float16, torch.bfloat16}:
            x_out = x_out.to(original_dtype)
        return x_out

class SpectralGCN(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(SpectralGCN, self).__init__()
        self.spatial_conv = make_edge_attention_layer(in_feats, out_feats)
        self.spectral_conv = FFTFeatureLayer(in_feats, out_feats)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()

    def forward(self, x, edge_index, edge_attr=None, edge_dir=None, edge_length=None):
        x_spatial = self.spatial_conv(x, edge_index, edge_attr=edge_attr, edge_dir=edge_dir, edge_length=edge_length)
        x_spectral = self.spectral_conv(x)
        gate = torch.sigmoid(self.gate_logit)
        x_out = x_spatial + gate * x_spectral
        x_out = self.bn(x_out)
        x_out = self.gelu(x_out)
        return x_out

class AMGPool(nn.Module):

    def __init__(self):
        super(AMGPool, self).__init__()

    def forward(self, x_fine, R):
        original_dtype = x_fine.dtype
        if x_fine.dtype in {torch.float16, torch.bfloat16}:
            x_fine = x_fine.to(torch.float32)
            R = R.to(dtype=torch.float32)
        with torch.autocast(device_type=x_fine.device.type, enabled=False):
            x_coarse = torch.sparse.mm(R, x_fine)
        if original_dtype in {torch.float16, torch.bfloat16}:
            x_coarse = x_coarse.to(original_dtype)
        return x_coarse

class AMGUnpool(nn.Module):

    def __init__(self):
        super(AMGUnpool, self).__init__()

    def forward(self, x_coarse, x_skip, P):
        original_dtype = x_coarse.dtype
        if x_coarse.dtype in {torch.float16, torch.bfloat16}:
            x_coarse = x_coarse.to(torch.float32)
            P = P.to(dtype=torch.float32)
        with torch.autocast(device_type=x_coarse.device.type, enabled=False):
            x_unpool = torch.sparse.mm(P, x_coarse)
        if original_dtype in {torch.float16, torch.bfloat16}:
            x_unpool = x_unpool.to(original_dtype)
        return x_unpool + x_skip

class StackedLinearBlock(nn.Module):

    def __init__(self, in_feats, out_feats, dropout):
        super(StackedLinearBlock, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

class FinalLinear(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(FinalLinear, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        return x

class BuildGCN(nn.Module):

    def __init__(self, input_feats, output_feats, base_dim=None, custom_dims=None):
        super(BuildGCN, self).__init__()
        self.input_feats = input_feats
        self.output_feats = output_feats
        if custom_dims is not None:
            dims = custom_dims
        elif base_dim is not None:
            dims = [base_dim, base_dim * 2, base_dim * 4]
        elif NETWORK_USE_CUSTOM_DIMS:
            dims = NETWORK_CUSTOM_DIMS
        else:
            dims = [NETWORK_BASE_DIM, NETWORK_BASE_DIM * 2, NETWORK_BASE_DIM * 4]
        self.hidden_feats = dims[0]
        self.gcn1 = SpectralGCN(input_feats, dims[0])
        self.pool1 = AMGPool()
        self.gcn2 = GCN(dims[0], dims[1])
        self.pool2 = AMGPool()
        self.gcn3 = GCN(dims[1], dims[2])
        self.bottom_gcn = GCN(dims[2], dims[2])
        self.dec_gcn1 = GCN(dims[2], dims[1])
        self.unpool1 = AMGUnpool()
        self.dec_gcn2 = GCN(dims[1], dims[0])
        self.unpool2 = AMGUnpool()
        self.dec_gcn3 = GCN(dims[0], dims[0])
        self.lin_stacked = StackedLinearBlock(dims[0], dims[0], dropout=0.1)
        self.final_linear = FinalLinear(dims[0], output_feats)
        self.use_frequency_modulation = NETWORK_USE_FREQUENCY_MODULATION
        self.freq_encoder = None
        self.film_gcn1 = None
        self.film_gcn2 = None
        self.film_gcn3 = None
        self.film_bottom = None
        self.film_dec_gcn1 = None
        self.film_dec_gcn2 = None
        self.film_dec_gcn3 = None
        if self.use_frequency_modulation:
            self.freq_encoder = FrequencyEncoder(FREQUENCY_EMBED_DIM)
            self.film_gcn1 = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[0])
            self.film_gcn2 = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[1])
            self.film_gcn3 = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[2])
            self.film_bottom = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[2])
            self.film_dec_gcn1 = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[1])
            self.film_dec_gcn2 = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[0])
            self.film_dec_gcn3 = FrequencyFiLM(FREQUENCY_EMBED_DIM, dims[0])

    @staticmethod
    def _count_tensor(value, device):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.long).reshape(-1)
        if isinstance(value, (list, tuple)):
            counts = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    if item.numel() > 0:
                        counts.append(int(item.reshape(-1)[0].item()))
                else:
                    counts.append(int(item))
            return torch.tensor(counts, dtype=torch.long, device=device)
        return torch.tensor([int(value)], dtype=torch.long, device=device)

    def _frequency_embedding(self, data, dtype, device):
        if not self.use_frequency_modulation:
            return None
        if not hasattr(data, 'freq'):
            raise RuntimeError('Frequency modulation is enabled but data.freq is missing. Rebuild graph caches after adding Freq_*.txt files, or disable FEMGRL_USE_FREQUENCY_MODULATION.')
        freq = data.freq.to(device=device, dtype=dtype)
        if freq.dim() == 1:
            freq = freq.view(-1, 1)
        return self.freq_encoder(freq)

    def _level_batch(self, data, level, num_nodes, num_graphs, device):
        if not self.use_frequency_modulation:
            return None
        if level == 0:
            node_batch = getattr(data, 'batch', None)
            if node_batch is None:
                return torch.zeros(num_nodes, dtype=torch.long, device=device)
            return node_batch.to(device=device, dtype=torch.long)
        counts = self._count_tensor(getattr(data, f'num_nodes_level_{level}', None), device)
        if counts is not None:
            counts = counts[:num_graphs]
            if counts.numel() == num_graphs and int(counts.sum().item()) == int(num_nodes):
                graph_ids = torch.arange(num_graphs, dtype=torch.long, device=device)
                return torch.repeat_interleave(graph_ids, counts)
        if num_graphs == 1:
            return torch.zeros(num_nodes, dtype=torch.long, device=device)
        raise RuntimeError(f'Cannot derive batch vector for AMG level {level}: num_nodes={num_nodes}, num_graphs={num_graphs}, counts={counts}')

    def _apply_frequency_film(self, x, film, freq_emb, node_batch):
        if freq_emb is None:
            return x
        if node_batch.numel() != x.size(0):
            raise RuntimeError(f'FiLM batch size mismatch: node_batch={node_batch.numel()}, x={x.size(0)}')
        if int(node_batch.max().item()) >= freq_emb.size(0):
            raise RuntimeError(f'FiLM graph index exceeds frequency batch size: max_batch={int(node_batch.max().item())}, freq_batch={freq_emb.size(0)}')
        return film(x, freq_emb[node_batch])

    def encode(self, data):
        x = data.x
        freq_emb = self._frequency_embedding(data, x.dtype, x.device)
        num_graphs = freq_emb.size(0) if freq_emb is not None else 1
        batch_0 = self._level_batch(data, 0, x.size(0), num_graphs, x.device)
        edge_index_0 = data.edge_index
        edge_attr_0 = getattr(data, 'edge_attr', None)
        edge_dir_0 = getattr(data, 'edge_dir', None)
        edge_length_0 = getattr(data, 'edge_length', None)
        edge_index_1 = getattr(data, 'edge_index_1', None)
        edge_attr_1 = getattr(data, 'edge_attr_1', None)
        edge_dir_1 = getattr(data, 'edge_dir_1', None)
        edge_length_1 = getattr(data, 'edge_length_1', None)
        edge_index_2 = getattr(data, 'edge_index_2', None)
        edge_attr_2 = getattr(data, 'edge_attr_2', None)
        edge_dir_2 = getattr(data, 'edge_dir_2', None)
        edge_length_2 = getattr(data, 'edge_length_2', None)
        R_0 = data.R_0[0] if isinstance(getattr(data, 'R_0', None), list) else getattr(data, 'R_0', None)
        P_0 = data.P_0[0] if isinstance(getattr(data, 'P_0', None), list) else getattr(data, 'P_0', None)
        R_1 = data.R_1[0] if isinstance(getattr(data, 'R_1', None), list) else getattr(data, 'R_1', None)
        P_1 = data.P_1[0] if isinstance(getattr(data, 'P_1', None), list) else getattr(data, 'P_1', None)
        x1 = self.gcn1(x, edge_index_0, edge_attr=edge_attr_0, edge_dir=edge_dir_0, edge_length=edge_length_0)
        x1 = self._apply_frequency_film(x1, self.film_gcn1, freq_emb, batch_0)
        if R_0 is None or P_0 is None or edge_index_1 is None:
            raise ValueError('缺少 Level-1 AMG 算子或粗层图信息：需要 R_0, P_0, edge_index_1。')
        x2 = self.pool1(x1, R_0)
        batch_1 = self._level_batch(data, 1, x2.size(0), num_graphs, x.device)
        x2 = self.gcn2(x2, edge_index_1, edge_attr=edge_attr_1, edge_dir=edge_dir_1, edge_length=edge_length_1)
        x2 = self._apply_frequency_film(x2, self.film_gcn2, freq_emb, batch_1)
        if R_1 is not None and P_1 is not None and (edge_index_2 is not None):
            x3 = self.pool2(x2, R_1)
            batch_2 = self._level_batch(data, 2, x3.size(0), num_graphs, x.device)
            x3 = self.gcn3(x3, edge_index_2, edge_attr=edge_attr_2, edge_dir=edge_dir_2, edge_length=edge_length_2)
            x3 = self._apply_frequency_film(x3, self.film_gcn3, freq_emb, batch_2)
            x_bottom = self.bottom_gcn(x3, edge_index_2, edge_attr=edge_attr_2, edge_dir=edge_dir_2, edge_length=edge_length_2)
            x_bottom = self._apply_frequency_film(x_bottom, self.film_bottom, freq_emb, batch_2)
            x_up1 = self.dec_gcn1(x_bottom, edge_index_2, edge_attr=edge_attr_2, edge_dir=edge_dir_2, edge_length=edge_length_2)
            x_up1 = self._apply_frequency_film(x_up1, self.film_dec_gcn1, freq_emb, batch_2)
            x_up1 = self.unpool1(x_up1, x2, P_1)
        else:
            x_up1 = self.bottom_gcn(x2, edge_index_1, edge_attr=edge_attr_1, edge_dir=edge_dir_1, edge_length=edge_length_1)
            x_up1 = self._apply_frequency_film(x_up1, self.film_bottom, freq_emb, batch_1)
        x_up2 = self.dec_gcn2(x_up1, edge_index_1, edge_attr=edge_attr_1, edge_dir=edge_dir_1, edge_length=edge_length_1)
        x_up2 = self._apply_frequency_film(x_up2, self.film_dec_gcn2, freq_emb, batch_1)
        x_up2 = self.unpool2(x_up2, x1, P_0)
        x_out = self.dec_gcn3(x_up2, edge_index_0, edge_attr=edge_attr_0, edge_dir=edge_dir_0, edge_length=edge_length_0)
        x_out = self._apply_frequency_film(x_out, self.film_dec_gcn3, freq_emb, batch_0)
        x_out = self.lin_stacked(x_out)
        return x_out

    def forward(self, data):
        x_out = self.encode(data)
        x_out = self.final_linear(x_out)
        return x_out

class BuildGCNSharedRealImag(BuildGCN):

    def __init__(self, input_feats, base_dim=None, custom_dims=None):
        super(BuildGCNSharedRealImag, self).__init__(input_feats, output_feats=1, base_dim=base_dim, custom_dims=custom_dims)
        self.output_feats = 2
        self.final_linear = nn.Identity()
        self.real_head = FinalLinear(self.hidden_feats, 1)
        self.imag_head = FinalLinear(self.hidden_feats, 1)

    def forward(self, data):
        hidden = self.encode(data)
        delta_real = self.real_head(hidden)
        delta_imag = self.imag_head(hidden)
        return torch.cat([delta_real, delta_imag], dim=-1)

class BuildGCNSharedHeadsList(nn.Module):

    def __init__(self, input_feats, n_iter=N_ITER, base_dim=None, custom_dims=None):
        super(BuildGCNSharedHeadsList, self).__init__()
        self.n_iter = n_iter
        self.input_feats = input_feats
        self.output_feats = 2
        self.networks = nn.ModuleList()
        for i in range(n_iter):
            network = BuildGCNSharedRealImag(input_feats, base_dim=base_dim, custom_dims=custom_dims)
            self.networks.append(network)
            setattr(self, f'iter_{i}', network)
        if not is_main_process():
            return
        print(f'Created {n_iter} shared-backbone AMG Graph U-Nets with separate real/imag heads')

    def forward(self, data, iter_idx=None):
        if iter_idx is not None:
            if iter_idx < 0 or iter_idx >= self.n_iter:
                raise ValueError(f'iter_idx must be in [0, {self.n_iter - 1}]')
            return self.networks[iter_idx](data)
        else:
            outputs = [net(data) for net in self.networks]
            return outputs

    def get_network(self, iter_idx):
        return self.networks[iter_idx]

    def __len__(self):
        return self.n_iter
