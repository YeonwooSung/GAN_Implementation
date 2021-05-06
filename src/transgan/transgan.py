import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Paper uses Vaswani (2017) Attention with minimal changes.
Multi-head self-attention with a feed-forward MLP with GELU non-linearity. Layer normalisation is used before each segment and employs residual skip connections.
"""

class Attention(nn.Module):
    def __init__(self, D, heads=8):
        super().__init__()
        self.D = D
        self.heads = heads

        assert (D % heads == 0), "Embedding size should be divisble by number of heads"
        self.head_dim = self.D // heads

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.H = nn.Linear(self.D, self.D)

    def forward(self, Q, K, V, mask):
        batch_size = Q.shape[0]
        q_len, k_len, v_len = Q.shape[1], K.shape[1], V.shape[1]

        Q = Q.reshape(batch_size, q_len, self.heads, self.head_dim)
        K = K.reshape(batch_size, k_len, self.heads, self.head_dim)
        V = V.reshape(batch_size, v_len, self.heads, self.head_dim)

        # performing batch-wise matrix multiplication
        raw_scores = torch.einsum("bqhd,bkhd->bhqk", [Q, K])

        # shut off triangular matrix with very small value
        scores = raw_scores.masked_fill(mask == 0, -np.inf) if mask else raw_scores

        attn = torch.softmax(scores / np.sqrt(self.D), dim=3)
        attn_output = torch.einsum("bhql,blhd->bqhd", [attn, V])
        attn_output = attn_output.reshape(batch_size, q_len, self.D)

        output = self.H(attn_output)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, D, heads, p, fwd_exp):
        super().__init__()
        self.mha = Attention(D, heads)
        self.drop_prob = p
        self.n1 = nn.LayerNorm(D)
        self.n2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, fwd_exp*D),
            nn.ReLU(),
            nn.Linear(fwd_exp*D, D),
        )
        self.dropout = nn.Dropout(p)

    def forward(self, Q, K, V, mask):
        attn = self.mha(Q, K, V, mask)

        """
        Layer normalisation with residual connections
        """
        x = self.n1(attn + Q)
        x = self.dropout(x)
        forward = self.mlp(x)
        x = self.n2(forward + x)
        out = self.dropout(x)

        return out


class MLP(nn.Module):
    def __init__(self, noise_w, noise_h, channels):
        super().__init__()
        self.l1 = nn.Linear(
                    noise_w*noise_h*channels, 
                    (8*8)*noise_w*noise_h*channels, 
                    bias=False
                )

    def forward(self, x):
        out = self.l1(x)
        return out


class PixelShuffle(nn.Module):
    def __init__(self):
        super().__init__()
        pass



#--------------------------------------------------------------------------------------------
# Generator
#--------------------------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(32, 32, 1)
        
        # stage 1
        self.s1_enc = nn.ModuleList([
                        EncoderBlock(1024*8*8)
                        for _ in range(5)
                    ])

        # stage 2
        self.s2_pix_shuffle = PixelShuffle()
        self.s2_enc = nn.ModuleList([
                        EncoderBlock(256*16*16)
                        for _ in range (4)
                    ])

        # stage 3
        self.s3_pix_shuffle = PixelShuffle()
        self.s3_enc = nn.ModuleList([
                        EncoderBlock(64*32*32)
                        for _ in range(2)
                    ])

        # stage 4
        self.linear = nn.Linear(32*32*64, 32*32*3)

    def forward(self, noise):
        x = self.mlp(noise)
        for layer in self.s1_enc:
            x = layer(x)
        
        x = self.s2_pix_shuffle(x)
        for layer in self.s2_enc:
            x = layer(x)

        x - self.s3_pix_shuffle(x)
        for layer in self.s3_enc:
            x = layer(x)

        img = self.linear(x)

        return img


#--------------------------------------------------------------------------------------------
# Discriminator
#--------------------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(32*32*3, (8*8+1)*384)
        self.s2_enc = nn.ModuleList([
                        EncoderBlock((8*8+1)*284)
                        for _ in range(7)
                    ])

        self.classification_head = nn.Linear(1*384, 1)

    def forward(self, img):
        x = self.l1(img)
        for layer in self.s2_enc:
            x = layer(x)

        logits = self.classification_head(x)
        pred = F.softmax(logits)
        
        return pred



#--------------------------------------------------------------------------------------------
# Main model
#--------------------------------------------------------------------------------------------

class TransGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = Generator()
        self.disc = Discriminator()

    def forward(self, noise):
        img = self.gen(noise)
        pred = self.disc(img)

        return img, pred
