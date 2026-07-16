#!/usr/bin/env python3
"""
train_lstm_npy.py - Train temporal gesture classifier using pure numpy.

Simulates realistic gesture motion patterns to generate temporal sequences from
single-frame data, then builds and trains a lightweight LSTM model.

Usage:
  python train_lstm_npy.py
"""

import os, sys, json
import numpy as np
import glob as glob_mod
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

GESTURES = ['open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
            'palm_up', 'palm_down', 'palm_left', 'palm_right']


def load_all_features():
    """Extract 12 features from every gesture sample."""
    feature_data = defaultdict(list)
    for gesture in GESTURES:
        gd = os.path.join(DATA_DIR, gesture)
        if not os.path.isdir(gd):
            continue
        for sf in sorted(glob_mod.glob(os.path.join(gd, "sample_*.json"))):
            with open(sf) as f:
                data = json.load(f)
            lm = data["landmarks"]
            W, H = data["width"], data["height"]
            xs = np.array([l["x"] * W for l in lm[:21]])
            ys = np.array([l["y"] * H for l in lm[:21]])
            zs = np.array([l["z"] * W for l in lm[:21]])
            n = np.column_stack([xs, ys, zs])
            wrist, ttip, ip, mid = n[0], n[4], n[2], n[12]
            pinky, pinky_t = n[17], n[20]
            box = max(np.ptp(xs), np.ptp(ys), 1.0)
            feat = np.zeros(12)
            feat[0] = np.linalg.norm(ttip[:2] - ip[:2]) / box
            v1, v2 = mid[:2]-wrist[:2], pinky[:2]-wrist[:2]
            feat[1] = (v1[0]*v2[1]-v1[1]*v2[0]) / (box**2)
            feat[2], feat[3] = xs.mean(), ys.mean()
            feat[4] = np.linalg.norm(ttip[:2] - pinky_t[:2]) / box
            feat[5] = np.arctan2((ys[12]+ys[17])/2-ys[0], (xs[12]+xs[17])/2-xs[0]) * 180/np.pi
            pc = np.array([xs.mean(), ys.mean()])
            feat[6] = np.linalg.norm(ttip[:2] - pc) / box
            feat[7] = (np.ptp(xs) * np.ptp(ys)) / (W * H)
            ext = 0
            for ti, mi in [(8,5),(12,9),(16,13),(20,17)]:
                if np.linalg.norm([xs[ti]-xs[mi], ys[ti]-ys[mi]]) / box > 0.3:
                    ext += 1
            feat[8] = ext
            bc = np.array([(np.ptp(xs)+np.min(xs))/2, (np.ptp(ys)+np.min(ys))/2])
            feat[9] = np.linalg.norm(wrist[:2] - bc) / box
            feat[10] = np.ptp(xs) / max(np.ptp(ys), 1.0)
            bh, bw = np.ptp(ys), np.ptp(xs)
            feat[11] = np.arctan2(bh-bw, bh+bw) * 180/np.pi
            feature_data[gesture].append(feat)
    return feature_data


def gen_sequences(fd, seq_len=30, n_per=80):
    """Generate temporal sequences with realistic gesture motion."""
    gids = {g: i for i, g in enumerate(GESTURES)}
    seqs, labs = [], []
    mpattern = {
        "open_palm":   {"s": 0.05, "jm": "sin"},
        "closed_fist": {"s": 0.05, "jm": "sin"},
        "thumbs_up":   {"s": 0.03, "jm": "gen"},
        "thumbs_down": {"s": 0.03, "jm": "gen"},
        "palm_up":     {"s": 0.04, "jm": "osc"},
        "palm_down":   {"s": 0.04, "jm": "osc"},
        "palm_left":   {"s": 0.06, "jm": "dir"},
        "palm_right":  {"s": 0.06, "jm": "dir"},
    }
    for gesture, flist in fd.items():
        fa = np.array(flist)
        pattern = mpattern[gesture]
        for _ in range(n_per):
            base = fa[np.random.randint(len(fa))].copy()
            seq = np.zeros((seq_len, 12))
            for t in range(seq_len):
                tn = t / seq_len
                noise = np.random.randn(12) * pattern["s"]
                if pattern["jm"] == "sin":
                    mod = 0.5 + 0.5 * np.sin(4*np.pi*tn)
                elif pattern["jm"] == "gen":
                    mod = np.sin(2*np.pi*tn) * 0.5
                elif pattern["jm"] == "osc":
                    mod = np.sin(6*np.pi*tn) * 0.3
                else:
                    mod = np.clip(2*tn, 0, 1)
                noise *= mod
                seq[t] = seq[t-1]*0.7 + (noise+base)*0.3 if t > 0 else base + np.random.randn(12)*0.02
            seqs.append(seq)
            labs.append(gids[gesture])

    X = np.array(seqs)
    y = np.array(labs)
    mu, sig = X.mean(), X.std() + 1e-8
    Xn = (X - mu) / sig
    oh = np.zeros((len(y), len(GESTURES)))
    for i, li in enumerate(y):
        oh[i, li] = 1.0
    return Xn, y, oh, mu, sig, gids


class SimpleLSTM:
    """LSTM network with backprop-through-time, pure numpy."""

    def __init__(self, inp, hid, outp, lr=0.01):
        self.lr = lr
        np.random.seed(42)
        s = np.sqrt(2.0 / (inp + hid))
        s2 = np.sqrt(2.0 / (hid + outp))

        self.W_if = np.random.randn(hid, inp) * s; self.b_f = np.zeros(hid)
        self.W_ii = np.random.randn(hid, inp) * s; self.b_i = np.zeros(hid)
        self.W_ic = np.random.randn(hid, inp) * s; self.b_c = np.zeros(hid)
        self.W_io = np.random.randn(hid, inp) * s; self.b_o = np.zeros(hid)
        self.W_hf = np.random.randn(hid, hid) * s
        self.W_hi = np.random.randn(hid, hid) * s
        self.W_hc = np.random.randn(hid, hid) * s
        self.W_ho = np.random.randn(hid, hid) * s
        self.W_hy = np.random.randn(outp, hid) * s2
        self.b_y = np.zeros(outp)

    def _sig(self, x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

    def forward(self, X):
        """X: (seq_len, features). Returns (probabilities, seq_len)."""
        sl = X.shape[0]
        h = np.zeros((sl+1, self.W_hy.shape[1]))
        c_f = np.zeros((sl+1, self.W_hy.shape[1]))
        self.fg = np.zeros((sl, self.W_hy.shape[1]))
        self.ig = np.zeros((sl, self.W_hy.shape[1]))
        self.cg = np.zeros((sl, self.W_hy.shape[1]))
        self.og = np.zeros((sl, self.W_hy.shape[1]))
        self.ctan = np.zeros((sl, self.W_hy.shape[1]))

        for t in range(sl):
            fg = self._sig(self.W_if @ X[t] + self.W_hf @ h[t] + self.b_f)
            ig = self._sig(self.W_ii @ X[t] + self.W_hi @ h[t] + self.b_i)
            cg = np.tanh(self.W_ic @ X[t] + self.W_hc @ h[t] + self.b_c)
            og = self._sig(self.W_io @ X[t] + self.W_ho @ h[t] + self.b_o)
            ct = fg * c_f[t] + ig * cg
            ht = og * np.tanh(ct)
            h[t+1] = ht; c_f[t+1] = ct
            self.fg[t] = fg; self.ig[t] = ig; self.cg[t] = cg; self.og[t] = og; self.ctan[t] = np.tanh(ct)

        logit = self.W_hy @ h[-1] + self.b_y
        shift = logit - np.max(logit)
        out = np.exp(shift) / np.sum(np.exp(shift))
        self.X = X; self.h = h; self.c = c_f
        return out

    def backward(self, y_true):
        """BPTT backward pass."""
        X = self.X; sl = X.shape[0]

        out = self._softmax(self.W_hy @ self.h[-1] + self.b_y)
        loss = -np.sum(y_true * np.log(out + 1e-14))

        dl = out - y_true
        dW_hy = dl.reshape(-1,1) @ self.h[-1].reshape(1,-1)
        db_y = dl
        dh = self.W_hy.T @ dl
        dc = np.zeros_like(dh)

        dW_if = np.zeros_like(self.W_if); dW_hf = np.zeros_like(self.W_hf); db_f = np.zeros_like(self.b_f)
        dW_ii = np.zeros_like(self.W_ii); dW_hi = np.zeros_like(self.W_hi); db_i = np.zeros_like(self.b_i)
        dW_ic = np.zeros_like(self.W_ic); dW_hc = np.zeros_like(self.W_hc); db_c = np.zeros_like(self.b_c)
        dW_io = np.zeros_like(self.W_io); dW_ho = np.zeros_like(self.W_ho); db_o = np.zeros_like(self.b_o)

        for t in reversed(range(sl)):
            hp = self.h[t]
            fg_t = self.fg[t]; ig_t = self.ig[t]; cg_t = self.cg[t]; og_t = self.og[t]; ctan_t = self.ctan[t]

            do = og_t * (1 - og_t) * dh * ctan_t
            dc = dc * fg_t + dh * (1 - ctan_t**2)
            dci = dc * ig_t * (1 - cg_t**2)
            df = dc * fg_t * (1 - fg_t)
            di = dc * ig_t * (1 - ig_t) * cg_t

            x0 = X[t]
            dW_if += do.reshape(-1,1) @ x0[None,:]
            dW_hf += do.reshape(-1,1) @ hp[None,:]
            db_f += do
            dW_ii += di.reshape(-1,1) @ x0[None,:]
            dW_hi += di.reshape(-1,1) @ hp[None,:]
            db_i += di
            dW_ic += dci.reshape(-1,1) @ x0[None,:]
            dW_hc += dci.reshape(-1,1) @ hp[None,:]
            db_c += dci
            dW_io += do.reshape(-1,1) @ x0[None,:]
            dW_ho += do.reshape(-1,1) @ hp[None,:]
            db_o += do

            dh = self.W_hf.T @ df + self.W_hi.T @ di + self.W_hc.T @ dci + self.W_ho.T @ do

        for g in [dW_hy, dW_if, dW_hf, dW_ii, dW_hi, dW_ic, dW_hc, dW_io, dW_ho]:
            np.clip(g, -5, 5, out=g)
        for g in [db_y, db_f, db_i, db_c, db_o]:
            np.clip(g, -5, 5, out=g)

        self.W_hy += self.lr * dW_hy; self.b_y += self.lr * db_y
        self.W_if += self.lr * dW_if; self.W_hf += self.lr * dW_hf; self.b_f += self.lr * db_f
        self.W_ii += self.lr * dW_ii; self.W_hi += self.lr * dW_hi; self.b_i += self.lr * db_i
        self.W_ic += self.lr * dW_ic; self.W_hc += self.lr * dW_hc; self.b_c += self.lr * db_c
        self.W_io += self.lr * dW_io; self.W_ho += self.lr * dW_ho; self.b_o += self.lr * db_o
        return loss

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def train_step(self, X, y):
        """Train on one sequence, return loss."""
        self.forward(np.array(X))
        return self.backward(np.array(y))


def train_lstm():
    print("=" * 60); print("LSTM Gesture Classifier (pure numpy)"); print("=" * 60)

    print("\n[1/5] Loading features...")
    fd = load_all_features()
    total = sum(len(v) for v in fd.values())
    print(f"   {total} samples across {len(fd)} gestures")
    for g in fd: print(f"     {g}: {len(fd[g])}")
    if total < 50:
        print("ERROR: Need >=50 samples."); sys.exit(1)

    print("\n[2/5] Generating sequences...")
    Xn, y_orig, y_onehot, mu, sig, gids = gen_sequences(fd, seq_len=30, n_per=80)
    print(f"   {len(Xn)} sequences, shape {Xn.shape}")
    from sklearn.model_selection import train_test_split
    tr_i, va_i = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=42)
    X_tr, y_tr = Xn[tr_i], y_onehot[tr_i]
    X_va, y_va = Xn[va_i], y_onehot[va_i]
    print(f"   Train: {len(X_tr)} / Val: {len(X_va)}")

    print("\n[3/5] Building LSTM...")
    model = SimpleLSTM(inp=12, hid=64, outp=len(GESTURES), lr=0.005)

    print("\n[4/5] Training (100 epochs)...")
    best_vl, pat = float("inf"), 15
    for ep in range(100):
        idx = np.random.permutation(len(X_tr))
        X_s, y_s = Xn[idx], y_onehot[idx]
        ep_loss = sum(model.train_step(X_s[i], y_s[i]) for i in range(len(X_s)))
        avg_tr = ep_loss / len(X_s)
        vl = sum(model.train_step(X_va[j], y_va[j]) for j in range(min(20, len(X_va)))) / min(20, len(X_va))

        if vl < best_vl: best_vl, pat = vl, 15
        else: pat -= 1

        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}: train={avg_tr:.4f} val={vl:.4f} best={best_vl:.4f}")
        if pat <= 0: print("\n  Early stop!"); break

    print("\n[5/5] Evaluating...")
    correct = sum(np.argmax(model.forward(X_va[i])) == np.argmax(y_va[i]) for i in range(len(X_va)))
    acc = correct / len(X_va)
    print(f"   Validation accuracy: {acc:.2%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    mp = os.path.join(RESULTS_DIR, "lstm_model.npz")
    np.savez(mp, W_hy=model.W_hy, b_y=model.b_y, W_if=model.W_if, W_hf=model.W_hf,b_f=model.b_f,
             W_ii=model.W_ii, W_hi=model.W_hi, b_i=model.b_i, W_ic=model.W_ic, W_hc=model.W_hc,
             b_c=model.b_c, W_io=model.W_io, W_ho=model.W_ho, b_o=model.b_o, mean=mu, std=sig,
             gesture_names=np.array(GESTURES), gids=np.array([[gids[g]] for g in GESTURES]))
    print(f"   Model saved to {mp}")
    return model, mp, acc


if __name__ == "__main__":
    mdl, path, acc = train_lstm()
    print(f"\nDone! Accuracy: {acc:.2%}, Model: {path}")
