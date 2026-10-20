"""Atribusi pemicu deforestasi dari patch NDVI (data simulasi) dengan Random Forest."""
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRV = ["tambang", "sawit", "ladang_kecil", "pembalakan", "kebakaran"]
S, rng = 32, np.random.default_rng(0)
NDVI = dict(tambang=.08, sawit=.25, ladang_kecil=.35, pembalakan=.30, kebakaran=.15)

def mask(d):
    m = np.zeros((S, S), bool)
    if d == "sawit":
        h, w = rng.integers(10, 20, 2); r, c = rng.integers(2, S - 20, 2); m[r:r+h, c:c+w] = 1
    elif d == "tambang":
        n = ndi.gaussian_filter(rng.normal(size=(S, S)), 3); m = n > np.quantile(n, .65)
    elif d == "ladang_kecil":
        for _ in range(rng.integers(1, 4)):
            h, w = rng.integers(3, 7, 2); r, c = rng.integers(0, S - 7, 2); m[r:r+h, c:c+w] = 1
    elif d == "pembalakan":
        r = rng.integers(4, S - 4); m[r:r+rng.integers(1, 3), :] = 1
        m = m.T if rng.random() < .5 else m
    else:
        n = ndi.gaussian_filter(rng.normal(size=(S, S)), 4); m = (n > np.quantile(n, .55)) & (rng.random((S, S)) > .3)
    return m

def fitur(d):
    bentuk = rng.choice(DRV) if rng.random() < .15 else d       # 15% ambigu
    m = mask(bentuk)
    before = rng.normal(.8, .05, (S, S)); after = before.copy()
    after[m] = rng.normal(NDVI[d] + rng.normal(0, .08), .18, m.sum())
    det = (before - after) > .25
    lab, n = ndi.label(det); sz = ndi.sum(det, lab, range(1, n + 1)) if n else [0]
    if det.any():
        rr, cc = np.where(det); bh, bw = np.ptp(rr) + 1, np.ptp(cc) + 1
        fill, el, nv = det.sum() / (bh * bw), max(bh, bw) / min(bh, bw), after[det].mean()
    else: fill = el = nv = 0
    return dict(luas=det.mean(), n_bagian=n, terbesar=max(sz) / S**2, fill=fill, memanjang=el, ndvi_sesudah=nv)

if __name__ == "__main__":
    y = rng.choice(DRV, 2500, p=[.15, .30, .25, .15, .15])
    X = pd.DataFrame([fitur(d) for d in y])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xtr, ytr); p = rf.predict(Xte)
    print(classification_report(yte, p, digits=3))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cm = confusion_matrix(yte, p, labels=DRV); ax[0].imshow(cm, cmap="Blues")
    ax[0].set(xticks=range(5), yticks=range(5), title="Confusion matrix", xlabel="Prediksi", ylabel="Sebenarnya")
    ax[0].set_xticklabels(DRV, rotation=35, ha="right"); ax[0].set_yticklabels(DRV)
    for i in range(5):
        for j in range(5): ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    pd.Series(rf.feature_importances_, X.columns).sort_values().plot.barh(ax=ax[1], color="#0072B2", title="Fitur penting")
    plt.tight_layout(); plt.savefig("atribusi_hasil.png", dpi=120); print("Grafik: atribusi_hasil.png")
