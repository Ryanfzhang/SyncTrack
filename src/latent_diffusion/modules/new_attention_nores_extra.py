from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import xformers
from latent_diffusion.modules.diffusionmodules.util import checkpoint
from diffusers.models.attention_processor import Attention
# attention extra, no replacement of attn1
# cancel residual add inside attn_extra

def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = True
    # use_flash_attention: bool = False
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        is_inplace: bool = True,
    ):
        # def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = heads
        self.d_head = dim_head

        # Attention scaling factor
        self.scale = dim_head**-0.5

        # The normal self-attention layer
        if context_dim is None:
            context_dim = query_dim

        # Query, key and value mappings
        d_attn = dim_head * heads
        self.to_q = nn.Linear(query_dim, d_attn, bias=False)
        self.to_k = nn.Linear(context_dim, d_attn, bias=False)
        self.to_v = nn.Linear(context_dim, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, query_dim), nn.Dropout(dropout))

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attention import FlashAttention

            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def forward(self, x, context=None, mask=None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = context is not None
        if not has_cond:
            context = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if (
            CrossAttention.use_flash_attention
            and self.flash is not None
            and not has_cond
            and self.d_head <= 128
        ):
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f"Head size ${self.d_head} too large for Flash Attention")

        # Pad the heads
        if pad:
            qkv = torch.cat(
                (qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1
            )

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        # TODO here I add the dtype changing
        out, _ = self.flash(qkv.type(torch.float16))
        # Truncate the extra head size
        out = out[:, :, :, : self.d_head].float()
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)  # [bs, 64, 20, 32]
        k = k.view(*k.shape[:2], self.n_heads, -1)  # [bs, 1, 20, 32]
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # attn: [bs, 20, 64, 1]
        # v: [bs, 1, 20, 32]
        out = torch.einsum("bhij,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)


# class BasicTransformerBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         n_heads,
#         d_head,
#         dropout=0.0,
#         context_dim=None,
#         gated_ff=True,
#         checkpoint=True,
#     ):
#         super().__init__()
#         self.attn1 = CrossAttention(
#             query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
#         )  # is a self-attention
        
#         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
#         self.attn2 = CrossAttention(
#             query_dim=dim,
#             context_dim=context_dim,
#             heads=n_heads,
#             dim_head=d_head,
#             dropout=dropout,
#         )  # is self-attn if context is none
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.checkpoint = checkpoint

#     def forward(self, x, context=None):
#         if context is None:
#             return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)
#         else:
#             return checkpoint(
#                 self._forward, (x, context), self.parameters(), self.checkpoint
#             )

#     def _forward(self, x, context=None):
#         x = self.attn1(self.norm1(x)) + x
#         x = self.attn2(self.norm2(x), context=context) + x
#         x = self.ff(self.norm3(x)) + x
#         return x




# class SpatialTransformer(nn.Module):
#     """
#     Transformer block for image-like data.
#     First, project the input (aka embedding)
#     and reshape to b, t, d.
#     Then apply standard transformer action.
#     Finally, reshape to image
#     """

#     def __init__(
#         self,
#         in_channels,
#         n_heads,
#         d_head,
#         depth=1,
#         dropout=0.0,
#         context_dim=None,
#         no_context=False,
#     ):
#         super().__init__()

#         if no_context:
#             context_dim = None

#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = Normalize(in_channels)

#         self.proj_in = nn.Conv2d(
#             in_channels, inner_dim, kernel_size=1, stride=1, padding=0
#         )

#         self.transformer_blocks = nn.ModuleList(
#             [
#                 BasicTransformerBlock(
#                     inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
#                 )
#                 for d in range(depth)
#             ]
#         )

#         self.proj_out = zero_module(
#             nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
#         )

#     def forward(self, x, context=None):
#         # note: if no context is given, cross-attention defaults to self-attention
#         b, c, h, w = x.shape
#         x_in = x
#         x = self.norm(x)
#         x = self.proj_in(x)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         for block in self.transformer_blocks:
#             x = block(x, context=context)
#         x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
#         x = self.proj_out(x)
#         return x + x_in


class XFormersJointAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=4
    ):
        b, t, c = hidden_states.shape
        s = 4
        b = int(b / s)

        residual = hidden_states # [B*4, H*W=9216, C=320]
        hidden_states = rearrange(hidden_states, '(b s) t c -> (s b) t c', s=s)  # [4*B, T, C]
        

        query = attn.to_q(residual) # Linear(320->320), hidden_states [B*4, H*W=9216, C=320] -> [B*4, H*W=9216, C=320]

        key = attn.to_k(hidden_states) # [4*B, H*W=9216, C=320]
        value = attn.to_v(hidden_states) # [4*B, H*W=9216, C=320]

        assert num_tasks == 4  # only support two tasks now

        key_0, key_1, key_2, key_3 = torch.chunk(key, dim=0, chunks=4)  # keys shape (b t) d c, key0=[1,9216,320], key1=[1,9216,320], key2=[1,9216,320], key3=[1,9216,320]
        value_0, value_1, value_2, value_3 = torch.chunk(value, dim=0, chunks=4) # value0=[1,9216,320], value1=[1,9216,320], value2=[1,9216,320], value3=[1,9216,320]
        
        key = torch.cat([key_0, key_1, key_2, key_3], dim=1)  # (b t) 4d c  [1, 4*9216, 320]
        value = torch.cat([value_0, value_1, value_2, value_3], dim=1)  # (b t) 4d c
        key = torch.cat([key]*4, dim=0)   # [B*4, 9216*4, 320]
        value = torch.cat([value]*4, dim=0)  # [B*4, 9216*4, 320]

        query = attn.head_to_batch_dim(query).contiguous() # [4, 9216, 320] -> [4*5, 9216, 64] 10=num_heads*2 64=320/num_heads=320/5 
        key = attn.head_to_batch_dim(key).contiguous() # '\n        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is the number of heads initialized while constructing the `Attention` class.\n\n        Args:\n            tensor (`torch.Tensor`): The tensor to reshape.\n            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is\n                reshaped to `[batch_size * heads, seq_len, dim // heads]`.\n\n  
        value = attn.head_to_batch_dim(value).contiguous() # [4, 9216*4, 320] -> [4*5, 9216*4, 64]

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask) # [B*4*5, 9216, 64]
        hidden_states = attn.batch_to_head_dim(hidden_states) # [B*4, 9216, 320]

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) # [4, 9216, 320]
        # dropout
        hidden_states = attn.to_out[1](hidden_states) # [4, 9216, 320]

        attn.residual_connection = False
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class CustomJointAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):  
        processor = XFormersJointAttnProcessor()
        self.set_processor(processor)
        print("using xformers attention processor")



class CustomTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention

        self.attn_cross_track = CustomJointAttention( 
            query_dim=dim, # 320 C
            heads=n_heads, # 5
            dim_head=d_head, # 64
            dropout=dropout, # 0.0
            bias=False, # False
            cross_attention_dim=None, # None
            upcast_attention=False, # False
            out_bias=True # True
        ) #  x = b (h w) c
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm_ct = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        self.attn_cross_track.set_use_memory_efficient_attention_xformers(True)

    def forward(self, x, context=None):
        if context is None:
            return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)
        else:
            return checkpoint(
                self._forward, (x, context), self.parameters(), self.checkpoint
            )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn_cross_track(self.norm_ct(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        no_context=False,
    ):
        super().__init__()

        if no_context:
            context_dim = None

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CustomTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )
        

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape # b = 4*B
        x_in = x

        x = self.norm(x) # x [4B, C, H, W]
        x = self.proj_in(x)  # [4B, C, H, W]

        # 转换为序列
        x = rearrange(x, "b c h w -> b (h w) c") # [4B, H*W, C]
        
        # 处理每个 Transformer Block
        for block in self.transformer_blocks:
            x = block(x, context=context)
        
        # 恢复原始形状 [4B, C, H, W]
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
    

if __name__ == "__main__":
    # a_old = torch.randn((2,3,512,256))
    # b, c, h, w = a_old.size()
    # a_old_rearranged = rearrange(a_old, "b c h w -> b (h w) c")
    # x1_old = a_old_rearranged[0,:,:]
    # x2_old = a_old_rearranged[1,:,:]

    # xck1_old, xck2_old  = torch.chunk(a_old_rearranged,dim=0,chunks=2)

    # print(x1_old.size(), xck1_old.size())
    # print(x1_old==xck1_old)


    # a = torch.randn((10, 4, 3, 512, 256))
    # x1 = a[:,0,:,:,:]
    # x2 = a[:,1,:,:,:]
    # x3 = a[:,2,:,:,:]
    # x4 = a[:,3,:,:,:]

    # b, s, c, h, w = a.size()
    # x = rearrange(a, "b s c h w -> (b s) c h w", s=s, b=b)
    # x = rearrange(x, "(b s) c h w -> (s b) c h w", s=s, b=b)
    # xck1, xck2, xck3, xck4 = torch.chunk(x, dim=0, chunks=4)
    # print(x1==xck1)

    x = torch.randn((10, 4, 320, 64, 64)).to("cuda")
    b, s, c, h, w = x.size()
    x = rearrange(x, "b s c h w -> (b s) c h w", s=s, b=b)
    x = rearrange(x, "B c h w -> B (h w) c") # [4B, H*W, C]
    attn1 = CustomJointAttention( 
            query_dim=320, # 320 C
            heads=5, # 5
            dim_head=64, # 64
            dropout=0.0, # 0.0
            bias=False, # False
            cross_attention_dim=None, # None
            upcast_attention=False, # False
            out_bias=True # True
        ).to("cuda") #  x = b (h w) c
    attn1.set_use_memory_efficient_attention_xformers(True)
    x_out = attn1(x)[0]