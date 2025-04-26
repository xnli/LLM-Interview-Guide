import torch 
from torch import nn


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000,device=None,scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算逆频率，shape:(dim//2,). theta_i = 1 / base^(2i / dim) 决定了不同维度下对旋转的速度。
        # 只取偶数对，是因为RoPE中是成对操作的，维度j和维度j+dim//2是使用的相同的theta_j。
        # 默认torch.arange生成的是torch.int64类型张量，这里显式指定保证安全。
        # 在除以dim之前使用float()显示指定torch.float32类型，也是确保执行的是浮点除法。
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        # x:[bs, num_attention_heads, seq_len, head_size]
        # (dim//2,)->(1,dim//2,1)
        # 矩阵乘法和三角函数的数学运算必须是浮点类型，显式float()变为torch.float32类型
        # position_ids.shape[0]是当前输入批次的batch大小。
        # .expand(position_ids.shape[0],-1,1)是将第一个维度扩展到batch大小，形状变化是（1,dim//2,1） -> (B,dim//2,1)
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0], -1, 1)
        # (B,seq_len)->(B,1,seq_len); 并转换为浮点类型
        position_ids_expanded = position_ids[:,None,:].float()
        
        # torch.autocast 需要一个 device_type 参数（通常是 "cuda" 或 "cpu"）。
        # 这段代码确保即使原始设备是 MPS，传递给 autocast 的 device_type 也是其支持的类型之一 ("cpu")。
        # 这可能是因为 autocast 对 MPS 的支持有限，或者是在禁用 autocast 时使用 "cpu" 作为类型参数更通用或安全。
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        # 关键点enabled=False, 避免自动将某些PyTorch运算切换为较低精度的数据类型来计算计算
        with torch.autocast(device_type=device_type, enabled=False):
            # (B,dim//2,1)@(B,1,seq_len)->(B,dim//2,seq_len)->(B,seq_len,dim//2)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            # (B,seq_len,dim//2)->(B,seq_len,dim)
            emb = torch.cat((freqs,freqs), dim=-1)
            #计算张量中每个角度的余弦值与正弦值
            cos = emb.cos()
            sin = emb.sin()
        # 将计算得到的 float32 的 cos 和 sin 转换回 原始输入 x 的数据类型（float16 或 bfloat16 或 float32），
        # 从而保证了它们能与 q 和 k 进行类型匹配的运算。
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # x: (bs,n_head,seq_len,head_size)
    # x1: (bs,n_head,seq_len,head_size//2)
    # x2: (bs,n_head,seq_len,head_size//2)
    x1 = x[..., : x.shape[-1] //2]
    x2 = x[..., x.shape[-1] // 2:]
    # (bs,n_head,seq_len,head_size)
    # 最后一维: [-q_{d/2}, -q_{d/2+1},... -q_{d-1}, q_0, q_1,...,q_{d/2-1}]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # q,k : [bs,num_heads,seq_len,head_dim]
    
    # unsqueeze_dim是根据q,k的shape来设定
    # (bs,seq_len,head_dim) -> (bs,1,seq_len,head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # 原始q 乘以cos 加上rotate_half(q)乘以sin,实现将q中每一对(q_j,q_{j+dim/2})视为二维向量，将其旋转相应的角度m*theta_j
    q_embd = q * cos + (rotate_half(q) * sin)
    k_embd = k * cos + (rotate_half(k) * sin)
    return q_embd, k_embd
    
            