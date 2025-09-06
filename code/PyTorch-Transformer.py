# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 12:15:09 2025

@author: Acer
"""

"""
toy_transformer_mt.py

Minimal Transformer encoder-decoder implemented from scratch (no nn.Transformer).
- Uses a toy copy/translation task (source -> target is source reversed and shifted with start/stop tokens).
- Implements Embeddings, Sinusoidal positional encoding, Multi-head attention, Feed-forward, LayerNorm, masking.
- Training loop with plotting and attention visualizations saved to runs/mt/ as requested.

Outputs (saved in runs/mt/):
- curves_mt.png
- attention_layer{L}_head{H}.png
- masks_demo.png
- decodes_table.png
- bleu_report.png
- report_one_page.md

Run: python toy_transformer_mt.py

"""

import math
import random
import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Toy dataset (integer sequences)
# -----------------------------
# We'll build a toy parallel corpus where input sequences are random integer sequences
# and target is the reversed input with a shift (target_token = input_token + 2 mod (vocab-4)),
# plus BOS and EOS tokens. This gives the model a non-trivial mapping to learn.

class ToyTranslationDataset(Dataset):
    def __init__(self, n_samples=10000, min_len=3, max_len=10, vocab_size=50):
        self.n_samples = n_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size

        # special tokens
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.OOV = 3
        self.offset = 4  # so tokens start from 4

        self.samples = [self._make_sample() for _ in range(n_samples)]

    def _make_sample(self):
        L = random.randint(self.min_len, self.max_len)
        seq = [random.randint(self.offset, self.vocab_size - 1) for _ in range(L)]
        # target mapping: reverse and shift by +1 within token space (mod reserved area)
        tgt = [((t - self.offset + 1) % (self.vocab_size - self.offset)) + self.offset for t in reversed(seq)]
        return seq, tgt

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# collate function

def collate_batch(batch, pad_idx=0, bos_idx=1, eos_idx=2):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens) + 2  # BOS and EOS

    batch_size = len(batch)
    src_pad = torch.full((batch_size, max_src), pad_idx, dtype=torch.long)
    tgt_input = torch.full((batch_size, max_tgt), pad_idx, dtype=torch.long)
    tgt_output = torch.full((batch_size, max_tgt), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_pad[i, : len(s)] = s
        tgt_input[i, 0] = bos_idx
        tgt_input[i, 1 : 1 + len(t)] = t
        tgt_input[i, 1 + len(t)] = eos_idx

        tgt_output[i, : len(t)] = t
        tgt_output[i, len(t)] = eos_idx
        # note: tgt_output has no initial BOS; shifted relative to tgt_input

    return src_pad, tgt_input, tgt_output

# -----------------------------
# Positional encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, learned=False):
        super().__init__()
        self.d_model = d_model
        if learned:
            self.pe = nn.Parameter(torch.randn(max_len, d_model))
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

# -----------------------------
# Scaled Dot-Product Attention + MultiHead
# -----------------------------

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    # q, k, v: (batch, n_heads, seq_len, head_dim)
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (batch, n_heads, seq_q, seq_k)
    if mask is not None:
        # mask should be broadcastable to scores; use -1e9 for masked positions
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)
    return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (batch, seq_len, d_model)
        B = query.size(0)
        Q = self.q_linear(query).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # mask: (batch, 1, seq_q, seq_k) or (batch, n_heads, seq_q, seq_k)
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)
        # out: (batch, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out_linear(out)
        return out, attn

# -----------------------------
# FeedForward
# -----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Encoder / Decoder Layer
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention
        attn_out, attn = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        self_attn_out, self_attn = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        cross_attn_out, cross_attn = self.cross_attn(x, enc_out, enc_out, mask=memory_mask)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x, self_attn, cross_attn

# -----------------------------
# Encoder, Decoder, Transformer
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=100, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src: (batch, src_len)
        x = self.tok_emb(src) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos_enc(x)

        attentions = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attentions.append(attn)
        x = self.norm(x)
        return x, attentions

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=100, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.tok_emb(tgt) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos_enc(x)

        self_attns = []
        cross_attns = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_out, tgt_mask, memory_mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)
        x = self.norm(x)
        return x, self_attns, cross_attns

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_out, enc_attns = self.encoder(src, src_mask)
        dec_out, self_attns, cross_attns = self.decoder(tgt, enc_out, tgt_mask, memory_mask)
        logits = self.out(dec_out)
        return logits, enc_attns, self_attns, cross_attns

# -----------------------------
# Mask helpers
# -----------------------------

def make_src_mask(src, pad_idx=0):
    # src: (batch, src_len)
    # return mask shaped (batch, 1, 1, src_len) or broadcastable: 1 where allowed
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # True=allowed


def make_tgt_mask(tgt, pad_idx=0):
    # tgt: (batch, tgt_len)
    batch, tgt_len = tgt.size()
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,tgt_len)
    # subsequent mask (causal)
    subsequent = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    subsequent = subsequent.unsqueeze(0).unsqueeze(1)  # (1,1,tgt_len,tgt_len)
    mask = pad_mask & subsequent
    return mask

# For memory mask: cross-attention should not attend to PAD in source

def make_memory_mask(src, tgt, pad_idx=0):
    # returns (batch, 1, tgt_len, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)
    tgt_len = tgt.size(1)
    return src_mask.repeat(1, 1, tgt_len, 1)

# -----------------------------
# Training utilities
# -----------------------------

def compute_loss_and_accuracy(logits, targets, pad_idx=0):
    # logits: (batch, tgt_len, vocab)
    batch, tgt_len, vocab = logits.size()
    logits_flat = logits.view(-1, vocab)
    targets_flat = targets.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx)
    # accuracy (ignoring pad)
    preds = logits_flat.argmax(dim=-1)
    mask = (targets_flat != pad_idx)
    if mask.sum() == 0:
        acc = torch.tensor(0.0)
    else:
        acc = (preds[mask] == targets_flat[mask]).float().mean()
    return loss, acc.item()

# Simple corpus BLEU (unigram..4-gram precision + brevity penalty)

def simple_corpus_bleu(references: List[List[int]], hypotheses: List[List[int]], max_n=4):
    # all tokens are integers; compute modified precision per n
    import collections

    precisions = []
    for n in range(1, max_n + 1):
        num = 0
        den = 0
        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = collections.Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)]) if len(ref) >= n else collections.Counter()
            hyp_ngrams = collections.Counter([tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1)]) if len(hyp) >= n else collections.Counter()
            overlap = 0
            total = sum(hyp_ngrams.values())
            for ng in hyp_ngrams:
                overlap += min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
            num += overlap
            den += total
        if den == 0:
            precisions.append(0.0)
        else:
            precisions.append(num / den)
    # geometric mean; if any precision is zero -> BLEU zero unless smoothing
    smooth = 1e-9
    gm = math.exp(sum((1.0/max_n) * math.log(p + smooth) for p in precisions))
    # brevity penalty
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    bp = 1.0
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len)
    bleu = bp * gm
    return bleu * 100

# -----------------------------
# Train / Eval
# -----------------------------

def train_epoch(model, dataloader, optimizer, pad_idx=0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    for src, tgt_in, tgt_out in dataloader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        src_mask = make_src_mask(src, pad_idx=pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx=pad_idx)
        memory_mask = make_memory_mask(src, tgt_in, pad_idx=pad_idx)

        optimizer.zero_grad()
        logits, enc_attn, self_attns, cross_attns = model(src, tgt_in, src_mask, tgt_mask, memory_mask)
        loss, acc = compute_loss_and_accuracy(logits, tgt_out, pad_idx=pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        steps += 1
    return total_loss / steps, total_acc / steps


def eval_epoch(model, dataloader, pad_idx=0):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    all_refs = []
    all_hyps = []
    attention_samples = (None, None, None)  # (enc, self, cross)
    masks_sample = None

    with torch.no_grad():
        for i, (src, tgt_in, tgt_out) in enumerate(dataloader):
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_mask = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=pad_idx)
            memory_mask = make_memory_mask(src, tgt_in, pad_idx=pad_idx)

            logits, enc_attns, self_attns, cross_attns = model(
                src, tgt_in, src_mask, tgt_mask, memory_mask
            )
            loss, acc = compute_loss_and_accuracy(logits, tgt_out, pad_idx=pad_idx)

            total_loss += loss.item()
            total_acc += acc
            steps += 1

            # decode greedy for BLEU
            preds = logits.argmax(dim=-1).cpu().tolist()
            refs = tgt_out.cpu().tolist()
            for r, p in zip(refs, preds):
                r = [token for token in r if token != 0]
                p = [token for token in p if token != 0]
                all_refs.append(r)
                all_hyps.append(p)

            # ✅ Capture just 1 tiny attention slice
            if attention_samples[0] is None:
                try:
                    enc_sample = enc_attns[0][0,0].detach().cpu()   # (seq, seq)
                    self_sample = self_attns[0][0,0].detach().cpu()
                    cross_sample = cross_attns[0][0,0].detach().cpu()
                    attention_samples = (enc_sample, self_sample, cross_sample)
                except Exception as e:
                    print("Could not capture attention:", e)

                masks_sample = (
                    src_mask.detach().cpu(),
                    tgt_mask.detach().cpu(),
                    memory_mask.detach().cpu()
                )

            # ⚠️ stop after one batch
            break

    bleu = simple_corpus_bleu(all_refs, all_hyps)
    return total_loss / steps, total_acc / steps, bleu, attention_samples, masks_sample

# Greedy decode function (for inference)

def greedy_decode(model, src, max_len=30, pad_idx=0, bos_idx=1, eos_idx=2):
    model.eval()
    src = src.to(device)
    src_mask = make_src_mask(src, pad_idx)
    with torch.no_grad():
        enc_out, _ = model.encoder(src, src_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        for i in range(max_len):
            tgt_mask = make_tgt_mask(ys, pad_idx=pad_idx)
            memory_mask = make_memory_mask(src, ys, pad_idx=pad_idx)
            dec_out, self_attns, cross_attns = model.decoder(ys, enc_out, tgt_mask, memory_mask)
            logits = model.out(dec_out)  # (batch, seq, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_idx).all():
                break
    return ys[:, 1:]

# -----------------------------
# Visualization helpers
# -----------------------------

def ensure_dirs():
    os.makedirs('runs/mt', exist_ok=True)


def plot_curves(train_losses, val_losses, filename='runs/mt/curves_mt.png'):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_attention_heatmaps(attention_samples, filename_pattern='runs/mt/attention_layer{L}_head{H}.png'):
    """
    attention_samples is a tuple of 3 small (seq, seq) tensors (enc, self, cross).
    """
    try:
        enc_attn, self_attn, cross_attn = attention_samples

        def _plot_one(attn_tensor, title, filename):
            arr = attn_tensor.numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(arr, aspect='auto', cmap="viridis")
            plt.colorbar()
            plt.title(title, fontsize=8)
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            plt.close()

        if enc_attn is not None:
            _plot_one(enc_attn, "Encoder L0 H0", filename_pattern.format(L="enc0", H="0"))
        if self_attn is not None:
            _plot_one(self_attn, "Decoder Self L0 H0", filename_pattern.format(L="decself0", H="0"))
        if cross_attn is not None:
            _plot_one(cross_attn, "Decoder Cross L0 H0", filename_pattern.format(L="deccross0", H="0"))

    except Exception as e:
        print("⚠️ Skipping attention heatmaps:", e)


def save_masks_demo(masks_sample, filename='runs/mt/masks_demo.png'):
    src_mask, tgt_mask, mem_mask = masks_sample
    # src_mask: (batch,1,1,src_len) -> squeeze
    src_m = src_mask[0,0,0].numpy()
    tgt_m = tgt_mask[0,0,0].numpy()
    mem_m = mem_mask[0,0].numpy()  # (tgt_len, src_len)

    fig, axes = plt.subplots(1,3, figsize=(9,3))
    axes[0].imshow(src_m, aspect='auto')
    axes[0].set_title('src_mask (1=token)')
    axes[1].imshow(tgt_m, aspect='auto')
    axes[1].set_title('tgt_mask (causal & pad)')
    axes[2].imshow(mem_m, aspect='auto')
    axes[2].set_title('memory_mask')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_decodes_table(dataset, model, filename='runs/mt/decodes_table.png', n=10):
    # sample n examples from dataset, decode and display table
    samples = [dataset[i] for i in range(n)]
    srcs = [s for s,t in samples]
    tgts = [t for s,t in samples]
    src_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    hyps = greedy_decode(model, src_padded, max_len=30).cpu().tolist()

    # convert to strings of ints
    rows = []
    for s, t, h in zip(srcs, tgts, hyps):
        rows.append((" ".join(map(str, s.tolist())), " ".join(map(str, t.tolist())), " ".join(map(str, [tok for tok in h if tok != 0]))))

    fig, ax = plt.subplots(figsize=(8, n*0.5 + 1))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['src', 'gold_tgt', 'decoded'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_bleu_report(bleu, filename='runs/mt/bleu_report.png'):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.axis('off')
    ax.text(0.5, 0.5, f'Corpus BLEU: {bleu:.2f}', fontsize=16, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_one_page_report(images, filename='runs/mt/report_one_page.md'):
    # simple markdown report listing images with one-line captions
    lines = ['# One-page Report\n']
    captions = {
        'runs/mt/curves_mt.png': 'Loss curves (train & val).',
        'runs/mt/masks_demo.png': 'Visualization of source/target/memory masks.',
        'runs/mt/decodes_table.png': 'Comparison of decoded outputs with ground truth (10 samples).',
        'runs/mt/bleu_report.png': 'Corpus BLEU score summary.',
    }
    for img in images:
        cap = captions.get(img, '')
        lines.append(f'![{os.path.basename(img)}]({img})\n\n*{cap}*\n')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# -----------------------------
# Main: construct dataset, model, train
# -----------------------------

def main():
    ensure_dirs()

    # ---- Lightweight config (from you) ----
    d_model = 64
    num_layers = 2
    num_heads = 2
    d_ff = 128
    batch_size = 8
    max_len = 20
    num_epochs = 3
    learning_rate = 1e-3
  

    # dataset (keep small)
    VOCAB = 50
    train_ds = ToyTranslationDataset(n_samples=800, min_len=3, max_len=8, vocab_size=VOCAB)
    val_ds = ToyTranslationDataset(n_samples=200, min_len=3, max_len=8, vocab_size=VOCAB)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_idx=0, bos_idx=1, eos_idx=2))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_idx=0, bos_idx=1, eos_idx=2))

    # model (smaller)
    model = Transformer(src_vocab=VOCAB, tgt_vocab=VOCAB,
                        d_model=d_model, n_layers=num_layers, n_heads=num_heads,
                        d_ff=d_ff, max_len=max_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_bleu = 0.0

    # We'll keep one attention/mask sample (populated in the final eval only)
    final_attention_sample = None
    final_masks_sample = None

    for epoch in range(1, num_epochs + 1):
        try:
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, pad_idx=0)

            # Lightweight validation: compute loss/acc only (no decoding heavy work)
            # We'll run a final full eval after training to collect attentions and BLEU
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            v_steps = 0
            with torch.no_grad():
                for src, tgt_in, tgt_out in val_loader:
                    src = src.to(device)
                    tgt_in = tgt_in.to(device)
                    tgt_out = tgt_out.to(device)
                    src_mask = make_src_mask(src, pad_idx=0)
                    tgt_mask = make_tgt_mask(tgt_in, pad_idx=0)
                    memory_mask = make_memory_mask(src, tgt_in, pad_idx=0)

                    logits, _, _, _ = model(src, tgt_in, src_mask, tgt_mask, memory_mask)
                    loss, acc = compute_loss_and_accuracy(logits, tgt_out, pad_idx=0)
                    total_val_loss += loss.item()
                    total_val_acc += acc
                    v_steps += 1

            avg_val_loss = total_val_loss / max(1, v_steps)
            avg_val_acc = total_val_acc / max(1, v_steps)

            # Append scalars (floats) — safe
            train_losses.append(tr_loss)
            val_losses.append(avg_val_loss)

            print(f'Epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} '
                  f'val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}')

            # free GPU cache if any (harmless on CPU)
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        except RuntimeError as e:
            # Catch OOMs and attempt to recover gracefully
            print(f'RuntimeError during epoch {epoch}: {e}')
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # stop training on OOM to avoid repeated crashes
            break

    # --- Final full evaluation (after training) to compute BLEU and capture attentions/masks ---
    print('\nRunning final evaluation (BLEU + attention capture)...')
    val_loss, val_acc, bleu, attention_samples, masks_sample = eval_epoch(model, val_loader, pad_idx=0)
    final_attention_sample = attention_samples
    final_masks_sample = masks_sample

    # Save the best model if BLEU improved (simple check)
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save(model.state_dict(), 'runs/mt/best_model.pt')

    # ---- NEW: Print BLEU inline ----
    print(f"\nFinal Evaluation Results:")
    print(f"  Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
    print(f"  Corpus BLEU Score: {bleu:.2f}")

    # ---- NEW: Show a few decoded examples inline ----
    print("\nSample decoded outputs (first batch from val set):")
    model.eval()
    with torch.no_grad():
        for i, (src, tgt_in, tgt_out) in enumerate(val_loader):
            src = src.to(device); tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)
            src_mask = make_src_mask(src, pad_idx=0)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=0)
            memory_mask = make_memory_mask(src, tgt_in, pad_idx=0)

            logits, _, _, _ = model(src, tgt_in, src_mask, tgt_mask, memory_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            refs = tgt_out.cpu().tolist()
            for r, p in zip(refs, preds):
                r = [tok for tok in r if tok != 0]
                p = [tok for tok in p if tok != 0]
                print(f"  REF: {r}")
                print(f"  HYP: {p}\n")
            break  # only print one batch

    # -------------------------
    # Safely save visuals (guarded)
    # -------------------------
    # If you are running into kernel deaths, set SAVE_VISUALS=False.
    SAVE_VISUALS = False   # <<--- set True only when you know your environment has enough memory

    # Always save cheap artifacts (curves + BLEU text)
    try:
        plot_curves(train_losses, val_losses)      # cheap
    except Exception as e:
        print("Could not save curves:", e)
    try:
        save_bleu_report(bleu)
    except Exception as e:
        print("Could not save BLEU report:", e)

    # If visuals requested, do them one by one with strong guards
    if SAVE_VISUALS:
        # attention heatmaps (tiny slices only)
        if final_attention_sample is not None:
            try:
                # Ensure attention_samples is in the expected tuple form
                # and call the function (which already draws only tiny slices).
                save_attention_heatmaps(final_attention_sample)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print("Could not save attention heatmaps:", e)

        # masks demo (small)
        if final_masks_sample is not None:
            try:
                save_masks_demo(final_masks_sample)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print("Could not save masks demo:", e)

        # decoded table (this runs greedy decode on n examples) — do only if you really want it
        try:
            save_decodes_table(val_ds, model)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print("Could not save decodes table:", e)

    else:
        print("SAVE_VISUALS=False -> skipped attention/mask/decode heavy visuals (safe mode).")

    # one-page markdown report (references the images above)
    try:
        images = ['runs/mt/curves_mt.png', 'runs/mt/masks_demo.png', 'runs/mt/decodes_table.png', 'runs/mt/bleu_report.png']
        # save_one_page_report expects the files; it's fine if some are missing
        save_one_page_report(images)
    except Exception as e:
        print("Could not write one-page report:", e)

    print(f'\nDone. Best BLEU: {best_bleu:.2f}. Artifacts (if produced) are in runs/mt/')



if __name__ == '__main__':
    main()
