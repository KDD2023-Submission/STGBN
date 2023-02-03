from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class DegreeEncoder(nn.Module):
    def __init__(self, in_degree, out_degree,
                 num_in_degree,
                 num_out_degree,
                 n_hid,
                 n_layers #for parameter initialization
                 ):
        super(DegreeEncoder, self).__init__()
        self.in_degree = nn.Parameter(in_degree, requires_grad=False)
        self.out_degree = nn.Parameter(out_degree, requires_grad=False)
        self.in_degree_encoder = nn.Embedding(num_in_degree, n_hid, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, n_hid, padding_idx=0)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x):
        in_degree, out_degree = (
            self.in_degree,
            self.out_degree,
        )
        batched_in_degree = in_degree.unsqueeze(0).expand(x.shape[0], -1)
        batched_out_degree = out_degree.unsqueeze(0).expand(x.shape[0], -1)
        return self.in_degree_encoder(batched_in_degree) + self.out_degree_encoder(batched_out_degree)


class SVDEmbedding(nn.Module):
    def __init__(self, svd, svd_dim, n_hid):
        super(SVDEmbedding,self).__init__()
        self.svd = nn.Parameter(svd, requires_grad=False)
        self.svd_dim=svd_dim
        self.embeddings = nn.Linear(svd_dim*2,n_hid)
    def forward(self, x):
        pos = self.svd
        batched_pos = pos.unsqueeze(0).expand(x.shape[0], -1, -1)
        sign = torch.randn(1)[0]>0
        sign = 1 if sign else -1
        pos_u = batched_pos[:,:,:self.svd_dim]*sign
        pos_v = batched_pos[:,:,self.svd_dim:]*(-sign)
        pos = torch.cat([pos_u,pos_v],dim=-1)
        return self.embeddings(pos)


class GraphAttnSpatialBias(nn.Module):
    def __init__(
        self,
        spatial,
        num_heads,
        num_spatial,
        n_layers
    ):
        super(GraphAttnSpatialBias, self).__init__()
        self.spatial = nn.Parameter(spatial, requires_grad=False)
        self.num_heads = num_heads
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x):
        # [N, N]
        spatial_pos = self.spatial
        # [B, N, N]
        spatial_pos = spatial_pos.unsqueeze(0).expand((x.shape[0],-1,-1))

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)

        return spatial_pos_bias# [n_graph, n_head, n_node, n_node]


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self,hidden_dim , ffn_hidden_dim, activation_fn="GELU", dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = F.relu

    def forward(self, x):
        residual=x
        x = self.dropout(self.fc2(self.act_dropout(self.ffn_act_func(self.fc1(x)))))
        x = x + residual
        x = self.ffn_layer_norm(x)
        return x


class MultiheadAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 dropout=0.1,
                 attn_dropout=0.1,
                 temperature = 1):
        super().__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads  # number of heads
        self.temperature =temperature
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim,eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.a_proj.weight)

    def forward(self, x, mask=None, attn_bias=None):
        residual = x
        batch_size = x.size(0)

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        #ScaledDotProductAttention
        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        scores = torch.matmul(query/self.temperature, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if attn_bias is not None:
            scores = scores+attn_bias

        if mask is not None:
            if scores.shape==mask.shape:#different heads have different mask
                scores = scores * mask
                scores = scores.masked_fill(scores == 0, -1e12)
            else:
                scores = scores.masked_fill(mask == 0, -1e12)

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        #ScaledDotProductAttention

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.dropout(self.a_proj(out))
        out = out + residual
        out = self.layer_norm(out)

        return out, attn


class Transformer_Layer(nn.Module):
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 ffn_hidden_dim,
                 dropout,
                 attn_dropout,
                 temperature = 1,
                 activation_fn='GELU'):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.attention = MultiheadAttention(num_heads,
                                            hidden_dim,
                                            dropout,
                                            attn_dropout,
                                            temperature)
        self.ffn_layer = PositionwiseFeedForward(hidden_dim,ffn_hidden_dim,activation_fn=activation_fn)


    def forward(self, x, attn_mask, attn_bias=None):
        x, attn = self.attention(x, mask=attn_mask, attn_bias=attn_bias)
        x = self.ffn_layer(x)

        return x, torch.tensor(0)


class GraphTransformer(nn.Module):
    def __init__(self, g_feat, n_nodes, n_hid, n_layers, n_heads, node_level_modules, attn_level_modules, svd_pos_dim, dropout):
        super().__init__()
        self.n_heads = n_heads

        self.node_level_layers = nn.ModuleList([])
        for module_name in node_level_modules:
            if module_name=='degree':
                layer = DegreeEncoder(g_feat.ndata['in_degrees'], g_feat.ndata['out_degrees'],
                                      num_in_degree=n_nodes,
                                      num_out_degree=n_nodes,
                                      n_hid=n_hid,
                                      n_layers=n_layers)
            elif module_name=='svd':
                layer = SVDEmbedding(g_feat.ndata['svd_emb'], svd_dim=svd_pos_dim,n_hid=n_hid)
            else:
                raise ValueError('node level module error!')
            self.node_level_layers.append(layer)
        #attention-level graph-structural feature encoder
        self.attn_level_layers = nn.ModuleList([])
        for module_name in attn_level_modules:
            if module_name=='spatial':
                layer = GraphAttnSpatialBias(g_feat.ndata['spatial_emb'], num_heads=n_heads,
                                             num_spatial=512,
                                             n_layers=n_layers)

            self.attn_level_layers.append(layer)

        #transformer layers
        self.transformer_layers =nn.ModuleList([
            Transformer_Layer(
                num_heads=n_heads,
                hidden_dim=n_hid,
                ffn_hidden_dim=n_hid,
                dropout=dropout,
                attn_dropout=dropout,
                temperature=1,
                activation_fn=F.relu
            ) for _ in range(n_layers)
        ])

    def forward(self, x):
        # node positional encoding
        # x--> T*B, N, F(n_hid)
        B, N, F = x.shape
        for nl_layer in self.node_level_layers:
            node_bias = nl_layer(x)
            x = x + node_bias

        # attention bias computation,  B x H x (T+1) x (T+1)  or B x H x T x T
        for al_layer in self.attn_level_layers:
            attn_bias = al_layer(x)

        for layer in self.transformer_layers:#23975
            x, _ = layer(x, None, attn_bias)

        return x


class ClassTokenConcatenator(nn.Module):
	"""
	Concatenates a class token to a set of tokens
	"""
	def __init__(self, token_dim: int,):

		super().__init__()

		class_token = torch.zeros(token_dim)
		self.class_token = nn.Parameter(class_token)

	def forward(self, input):

		# This gives a tensor of shape batch_size X 1 X token_dim
		class_token = self.class_token.expand(1, input.shape[1], -1)
    		# Then, it is concatenated alongside the other tokens,
    		# which gives an output of shape batch_size X number of original tokens + 1 X token_dim
    		# Of course, the +1 comes from the newly added class token
		output = torch.cat((input, class_token), dim=0)
		return output


class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self,
                 n_hid,
                 n_heads,
                 n_layers,
                 dropout,
                 class_token,
                 positional_encoding):
        super().__init__()

        self.class_token = class_token
        self.positional_encoding = positional_encoding

        if class_token:
            self.class_token_concatenator = ClassTokenConcatenator(n_hid)

        if positional_encoding:
            self.pos_encoder = PositionalEncoding(n_hid, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(n_hid, n_heads, n_hid)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, x):
        if self.class_token:
            x = self.class_token_concatenator(x)

        if self.positional_encoding:
            x = self.pos_encoder(x)

        h = self.transformer_encoder(x)

        return h


class STGTransformerBlock(nn.Module):
    def __init__(self, g,
                 n_nodes: int,
                 n_hid: int,
                 n_temporal_layers,
                 n_spatial_layers: int,
                 n_heads: int,
                 node_level_modules: list,
                 attn_level_modules: list,
                 svd_pos_dim: int,
                 dropout: float):
        super().__init__()

        self.temporal_up = Transformer(n_hid, n_heads, n_temporal_layers, dropout, class_token = False, positional_encoding = True)
        self.spatial = GraphTransformer(g, n_nodes, n_hid, n_spatial_layers, n_heads, node_level_modules, attn_level_modules, svd_pos_dim, dropout)
        self.temporal_down = Transformer(n_hid, n_heads, n_temporal_layers, dropout, class_token = True, positional_encoding = True)

    def forward(self, shape, h):
        (B, T, N, F) = shape

        # T, B*N, F(n_hid)
        h = self.temporal_up(h)

        # T*B, N, F(n_hid)
        h = torch.reshape(h, (T*B, N, -1))

        # T*B, N, F(n_hid)
        h = self.spatial(h)

        # T, B*N, F(n_hid)
        h = torch.reshape(h, (T, B*N, -1))

        # T+1, B*N, F(n_hid)
        h = self.temporal_down(h)

        # B, N, F(n_hid)
        h = torch.reshape(h[-1], (B, N, -1))

        return h


class STGNN(nn.Module):
    def __init__(self,
                 g,
                 n_nodes: int,
                 n_inp: int,
                 n_hid: int,
                 n_out: int,
                 n_temporal_layers: int,
                 n_spatial_layers: int,
                 n_heads: int,
                 node_level_modules: list,
                 attn_level_modules: list,
                 svd_pos_dim: int,
                 dropout: float = 0.0):
        """

        :param n_nodes           : int  , number of nodes
        :param n_inp             : int  , input dimension
        :param n_hid             : int  , hidden dimension
        :param n_out             : int  , output dimension
        :param n_temporal_layers : int  , number of temporal transformer encoders
        :param n_spatial_layers  : int  , number of spatial transformer encoders
        :param n_heads           : int  , number of attention heads
        :param node_level_modules: list , graph transformer PEs
        :param attn_level_modules: list , graph transformer attention bias
        :param dropout           : float, dropout rate
        """
        super(STGNN, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_heads   = n_heads

        self.input_transformation = nn.Linear(n_inp, n_hid, bias=False)

        self.stg_transformer_block = STGTransformerBlock(g, n_nodes, n_hid, n_temporal_layers, n_spatial_layers, n_heads, node_level_modules, attn_level_modules, svd_pos_dim, dropout)

        self.output_transformation = nn.Sequential(
            nn.Linear(n_hid, 32),
            nn.ReLU(),
            nn.Linear(32, n_out)
        )

    def forward(self, x):
        """

        :param x    : torch.Tensor, size: B, T, N, F
        """

        B, T, N, F = x.shape

        # T, B, N, F(1)
        x = torch.transpose(x, 0, 1)

        # T, B*N, F(1)
        x = torch.reshape(x, (T, -1, F))

        # T, B*N, F(n_hid)
        h = self.input_transformation(x)

        # B, N, F(n_hid)
        h = self.stg_transformer_block((B, T, N, F), h)

        # B, T, N
        output = torch.transpose(self.output_transformation(h), 1, 2)

        return output
