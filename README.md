# AI Atribusi Pemicu Deforestasi (citra satelit simulasi)

Terinspirasi repo *Attributing direct drivers to deforestation alerts with satellite imagery and AI*.
Tiap alert = patch NDVI 32×32 sebelum/sesudah. Fitur bentuk (luas, kekompakan, kememanjangan, NDVI) → **Random Forest** memprediksi pemicu:
tambang, sawit, ladang kecil, pembalakan, kebakaran.

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python atribusi_deforestasi.py
```
Keluaran: laporan klasifikasi + `atribusi_hasil.png`.

**Catatan:** data sintetis. Untuk data nyata gunakan ekspor Google Earth Engine (Sentinel-2 / Hansen GFC) dari modul `gee/` repo Anda; model lanjutan: CNN pada patch mentah.
