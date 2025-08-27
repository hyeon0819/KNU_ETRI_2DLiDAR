# models/point_mae.py  (PyTorch-only / no PyTorch3D)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Torch-only FPS + KNN
# ----------------------------
@torch.no_grad()
def farthest_point_sampling(xyz: torch.Tensor, K: int):
    """
    KNN을 위한 중심점 샘플링 (최원거리 샘플링)
    
    xyz: (B,N,3)
    return: centers(B,K,3), center_idx(B,K)
    """
    B, N, _ = xyz.shape
    device = xyz.device
    center_idx = torch.zeros(B, K, dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    distances = torch.full((B, N), 1e10, device=device)
    bidx = torch.arange(B, device=device)

    for i in range(K):
        center_idx[:, i] = farthest
        centroid = xyz[bidx, farthest, :].unsqueeze(1)     # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)    # (B,N)
        distances = torch.minimum(distances, dist)
        farthest = torch.max(distances, dim=1).indices

    centers = xyz[bidx.unsqueeze(1), center_idx]           # (B,K,3)
    return centers, center_idx

@torch.no_grad()
def knn_gather_torch(xyz: torch.Tensor, centers: torch.Tensor, K: int):
    """
    KNN으로 각 센터 주변 M=K개의 이웃을 모음
    
    xyz: (B,N,3), centers: (B,G,3)
    return: idx(B,G,K), neighborhood(B,G,K,3)
    """
    # (B,G,N) pairwise dist
    dist = torch.cdist(centers, xyz, p=2)                  # (B,G,N)
    idx = torch.topk(dist, k=K, dim=-1, largest=False).indices  # (B,G,K)
    B, G, N = dist.shape
    _, _, Kk = idx.shape
    xyz_exp = xyz.unsqueeze(1).expand(B, G, N, 3)          # (B,G,N,3)
    idx_exp = idx.unsqueeze(-1).expand(B, G, Kk, 3)
    neighborhood = torch.gather(xyz_exp, 2, idx_exp)       # (B,G,K,3)
    return idx, neighborhood


class Group(nn.Module):
    """
    (B,N,3) 포인트 집합을 G개의 센터와 각 센터 주변 K개의 이웃으로 묶어 패치(B,G,K,3)를 생성하는 모듈 
    """
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        input: (B,N,3) or (N,3) or numpy
        output:
          neighborhood: (B,G,M,3) centered
          center:       (B,G,3)
          idx:          (B,G,M)   indices into original N
        """
        if not isinstance(xyz, torch.Tensor):
            xyz = torch.from_numpy(xyz).float()
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)
        xyz = xyz.contiguous()
        centers, _ = farthest_point_sampling(xyz, self.num_group)      # 센터 G개 선택 (B,G,3)
        idx, neighborhood = knn_gather_torch(xyz, centers, self.group_size)  # KNN으로 이웃 gather (B,G,M),(B,G,M,3)
        neighborhood = neighborhood - centers.unsqueeze(2)   # KNN 이웃을 센터 기준으로 정렬
        return neighborhood, centers, idx

# ----------------------------
# Encoder (Point-MAE style)
# ----------------------------
class Encoder(nn.Module):
    """
    패치 (각 Group)의 이웃 point들을 받아 그룹 임베딩 벡터로 변환
    """
    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        point_groups : B G M 3
        return: B G C
        """
        bs, g, m, _ = point_groups.shape
        pg = point_groups.reshape(bs * g, m, 3)                  # (BG,M,3)
        feat = self.first_conv(pg.transpose(2,1))                # (BG,256,M)
        feat_global = torch.max(feat, dim=2, keepdim=True)[0]    # (BG,256,1)
        feat = torch.cat([feat_global.expand(-1,-1,m), feat], dim=1) # (BG,512,M)
        feat = self.second_conv(feat)                            # (BG,C,M)
        feat_global = torch.max(feat, dim=2, keepdim=False)[0]   # (BG,C)
        return feat_global.reshape(bs, g, self.encoder_channel)

# ----------------------------
# Transformer blocks (ViT-like)
# ----------------------------
class Mlp(nn.Module):
    """
    Transformer의 FNN 부분: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Attention(nn.Module):
    """
    Multi Head Self Attention (MHSA)
    
    - qkv projection 후, (B, num_heads, N, head_dim) 형태로 나눔
    - scaled dot-product attention 수행 후 원래 차원 복원    
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj_drop(self.proj(x))
        return x

class DropPath(nn.Module):
    """
    Stochastic Depth. 학습 시 잔차 분기를 확률적으로 drop 하여 일반화에 도움
    """
    def __init__(self, drop_prob=0.0): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep) * random_tensor

class Block(nn.Module):
    """
    Transformer 블록 = Layer Norm → MHSA → DropPath 잔차 + Layer Norm → MLP → DropPath 잔차
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """
    positional embedding과 함께 입력 token에 더해 각 블록에서 처리
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
    def forward(self, x, pos):
        for blk in self.blocks:
            x = blk(x + pos)
        return x

# ----------------------------
# Detector (returns logits (B,N))
# ----------------------------
class PointMAE2DLiDARDetector(nn.Module):
    """
    2D LiDAR 포인트 검출기 (Point-MAE 스타일 백본 + per-point 헤드)
    
    Input:
      feats or pts:
        - (B,N,4) = (x,y,r,theta_norm)   -> 내부에서 (x,y,r)만 사용
        - (B,N,3) = (x,y,r)
    Output:
      logits: (B,N) -> 각 point가 target (1)일 확률에 대한 logit 
    """
    def __init__(self, group_size=64, num_group=256, encoder_dims=256,
                 trans_dim=256, depth=4, num_heads=6, drop_path_rate=0.1,
                 point_head_hidden=128, scatter_reduce='mean'):
        super().__init__()
        # 패치 생성기(FPS+KNN)
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        # 그룹 임베딩 인코더(Conv1d 기반)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        # Transformer 차원/토큰 설정
        self.trans_dim = trans_dim
        self.cls_token = nn.Parameter(torch.zeros(1,1,trans_dim))
        self.cls_pos   = nn.Parameter(torch.zeros(1,1,trans_dim))

        self.proj = nn.Linear(encoder_dims, trans_dim) if encoder_dims != trans_dim else nn.Identity()
        self.pos_embed = nn.Sequential(nn.Linear(3,128), nn.GELU(), nn.Linear(128, trans_dim))
        self.encoder_tr = TransformerEncoder(embed_dim=trans_dim, depth=depth, num_heads=num_heads,
                                             drop_path_rate=drop_path_rate)
        self.norm = nn.LayerNorm(trans_dim)

        # per-point 헤드: [그룹 특징 C || 로컬 좌표(3)] → 1 로짓
        self.point_head = nn.Sequential(
            nn.Linear(trans_dim + 3, point_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(point_head_hidden, 1)
        )

        # 그룹 → 원본 인덱스로 합치는 방식 선택 (지금은 mean으로 구현)
        #  - 'max' : 그룹 중 가장 큰 로짓 선택 (비미분/불안정할 수 있음)
        #  - 'mean': 평균(미분 가능, 보통 학습 안정)
        assert scatter_reduce in ['max', 'mean']
        self.scatter_reduce = scatter_reduce


        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def forward(self, x):
        """
        x: (B,N,4) or (B,N,3)
        return logits: (B,N)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        
        # 지금은 x, y, r만 input으로 들어가게끔 함
        if C == 4:
            pts3 = torch.stack([x[...,0], x[...,1], x[...,2]], dim=-1)  # (B,N,3)
        elif C == 3:
            pts3 = x
        else:
            raise ValueError(f"Expected last dim 3 or 4, got {C}")

        # patches
        neigh, center, idx = self.group_divider(pts3)   # neigh:(B,G,M,3), center:(B,G,3), idx:(B,G,M)

        # group tokens
        group_tokens = self.encoder(neigh)              # (B,G,Ce)
        group_tokens = self.proj(group_tokens)          # (B,G,C)

        # transformer with cls
        cls_tok = self.cls_token.expand(B, -1, -1)      # (B,1,C)
        cls_pos = self.cls_pos.expand(B, -1, -1)        # (B,1,C)
        pos = self.pos_embed(center)                    # (B,G,C)
        xt = torch.cat([cls_tok, group_tokens], dim=1)  # (B,1+G,C)
        pt = torch.cat([cls_pos, pos], dim=1)           # (B,1+G,C)
        xt = self.encoder_tr(xt, pt)
        xt = self.norm(xt)
        group_feat = xt[:, 1:, :]                       # (B,G,C)

        # per-point logit in each group
        gfeat_exp = group_feat.unsqueeze(2).expand(-1, -1, neigh.size(2), -1)  # (B,G,M,C)
        concat = torch.cat([gfeat_exp, neigh], dim=-1)                         # (B,G,M,C+3)
        logits_gm = self.point_head(concat).squeeze(-1)                        # (B,G,M)

        # scatter back to (B,N)
        if self.scatter_reduce == 'max':
            logits_full = pts3.new_full((B, N), float('-inf'))
            for b in range(B):
                logits_full[b].scatter_reduce_(0, idx[b].reshape(-1), logits_gm[b].reshape(-1),
                                               reduce='amax', include_self=True)
        else:
            logits_full = pts3.new_zeros((B, N))
            counts = pts3.new_zeros((B, N))
            for b in range(B):
                logits_full[b].scatter_add_(0, idx[b].reshape(-1), logits_gm[b].reshape(-1))
                counts[b].scatter_add_(0, idx[b].reshape(-1), torch.ones_like(logits_gm[b].reshape(-1)))
            logits_full = logits_full / counts.clamp_min(1.0)

        # 미커버 포인트는 -inf(혹은 0)일 수 있음 → 학습/평가 시 mask로 제외됨
        return logits_full
