"""Build the from-scratch ML notebooks (nbformat 4) — one source of truth.

    python ml/notebooks/build_notebooks.py

Keeps the .ipynb file valid JSON and consistent with the production
modules in the spec2desc package.
"""

from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def nb(cells):
    return {
        "cells": [
            ({"cell_type": "markdown", "metadata": {},
              "source": md.splitlines(keepends=True)} if kind == "md" else
             {"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": code.splitlines(keepends=True)})
            for kind, body in cells for md, code in [(body, body)]
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NB2 = nb([
    ("md", "# 02 · Seq2seq + attention from scratch\n\n"
           "A tiny encoder-decoder with multi-head cross-attention trained to turn "
           "linearised specs into descriptions — the architecture family behind the "
           "production T5 (`app/generator.py`), with hand-written beam search."),
    ("code", "import os, sys, json\n"
             "sys.path.insert(0, os.path.abspath('..'))\n"
             "import torch\n"
             "from spec2desc.linearise import linearise_spec\n"
             "from spec2desc.attention import EncoderBlock, DecoderBlock, causal_mask\n"
             "from spec2desc.beam_search import beam_search\n\n"
             "CONFIG = dict(d_model=64, n_heads=4, d_ff=128, epochs=8, batch=16, lr=3e-3, seed=42)\n"
             "torch.manual_seed(CONFIG['seed'])\n"
             "device = 'cuda' if torch.cuda.is_available() else 'cpu'"),
    ("code", "# corpus\n"
             "rows = [json.loads(l) for l in open('../../ml/data/generated/train.jsonl')][:800]\n"
             "vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2}\n"
             "for r in rows:\n"
             "    for tok in (linearise_spec(r['spec']) + ' ' + r['description']).lower().split():\n"
             "        vocab.setdefault(tok, len(vocab))\n"
             "inv = {v: k for k, v in vocab.items()}\n"
             "def enc(s): return [vocab[t] for t in s.lower().split() if t in vocab]\n"
             "print('vocab:', len(vocab))"),
    ("code", "class TinySeq2Seq(torch.nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.emb = torch.nn.Embedding(len(vocab), CONFIG['d_model'])\n"
             "        self.enc = EncoderBlock(CONFIG['d_model'], CONFIG['n_heads'], CONFIG['d_ff'])\n"
             "        self.dec = DecoderBlock(CONFIG['d_model'], CONFIG['n_heads'], CONFIG['d_ff'])\n"
             "        self.out = torch.nn.Linear(CONFIG['d_model'], len(vocab))\n\n"
             "    def forward(self, src, tgt):\n"
             "        mem = self.enc(self.emb(src))\n"
             "        x, attn = self.dec(self.emb(tgt), mem,\n"
             "                           causal_mask(tgt.shape[1], tgt.device), return_weights=True)\n"
             "        return self.out(x), attn\n\n"
             "model = TinySeq2Seq().to(device)\n"
             "print('parameters:', sum(p.numel() for p in model.parameters()))"),
    ("code", "# teacher-forced training on (spec -> description)\n"
             "opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])\n"
             "losses = []\n"
             "for epoch in range(CONFIG['epochs']):\n"
             "    for i in range(0, len(rows) - CONFIG['batch'], CONFIG['batch']):\n"
             "        batch = rows[i:i + CONFIG['batch']]\n"
             "        src = torch.tensor([enc(linearise_spec(r['spec']))[:48] for r in batch]).to(device)\n"
             "        tgt = torch.tensor([[vocab['<bos>']] + enc(r['description'])[:40] + [vocab['<eos>']] for r in batch]).to(device)\n"
             "        logits, _ = model(src, tgt[:, :-1])\n"
             "        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), tgt[:, 1:].reshape(-1))\n"
             "        opt.zero_grad(); loss.backward(); opt.step()\n"
             "        losses.append(float(loss))\n"
             "print('final loss:', losses[-1])"),
    ("md", "## Attention maps: which spec fields does the decoder look at?"),
    ("code", "import matplotlib.pyplot as plt\n"
             "src = torch.tensor([enc(linearise_spec(rows[0]['spec']))[:48]]).to(device)\n"
             "tgt = torch.tensor([[vocab['<bos>']] + enc(rows[0]['description'])[:12]]).to(device)\n"
             "with torch.no_grad():\n"
             "    _, attn = model(src, tgt)\n"
             "plt.figure(figsize=(8, 4))\n"
             "plt.imshow(attn[0, 0].cpu(), aspect='auto')\n"
             "plt.xlabel('spec tokens'); plt.ylabel('output steps')\n"
             "plt.title('cross-attention (head 0)'); plt.show()\n"
             "print('source:', ' '.join(inv[t] for t in src[0].tolist()))"),
    ("md", "## Hand-written beam search decoding"),
    ("code", "def step_fn(prefix):\n"
             "    tgt = torch.tensor([list(prefix)]).to(device)\n"
             "    with torch.no_grad():\n"
             "        mem = model.enc(model.emb(src))\n"
             "        x, _ = model.dec(model.emb(tgt), mem, causal_mask(tgt.shape[1], device))\n"
             "        logp = torch.log_softmax(model.out(x)[0, -1], dim=-1)\n"
             "    return {i: float(logp[i]) for i in range(len(vocab))}\n\n"
             "beams = beam_search(step_fn, bos=vocab['<bos>'], eos=vocab['<eos>'],\n"
             "                    beam_width=4, max_len=24, num_return=3)\n"
             "for b in beams:\n"
             "    print(round(b.score, 3), ' '.join(inv[t] for t in b.tokens[1:-1]))"),
])

def main() -> None:
    path = os.path.join(OUT, "02_seq2seq_attention_from_scratch.ipynb")
    with open(path, "w") as fh:
        json.dump(NB2, fh, indent=1)
    json.load(open(path))  # validate
    print("wrote", path, f"({len(NB2['cells'])} cells)")


if __name__ == "__main__":
    main()
