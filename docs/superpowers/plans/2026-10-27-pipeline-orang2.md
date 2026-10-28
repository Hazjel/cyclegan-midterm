# Pipeline orang 2

Tujuan: memenuhi job desk halaman 2 panduan dan kontrak integrasi yang sudah ada.

Keputusan implementasi: gunakan PyTorch/Pillow terpasang tanpa dependency baru;
dataset Kaggle sebagai pembanding identitas, hanya data_cleaned sebagai kandidat.
Simpan sumber ZIP utuh, keluaran RGB PNG terpisah, metadata diperbaiki dan manifest.
Deduplikasi piksel RGB deterministik, test diprioritaskan; konflik domain ditolak.
Train: resize 286, crop acak 256, horizontal flip 0.5, tensor float32 [-1,1].
Test: resize 256 deterministik, loader terpisah per domain untuk mencegah hitung ulang.
Tidak mengubah model atau mengerjakan training milik orang 3.

- [x] Tambahkan tes preprocessing: sumber berbeda, metadata ambigu, blur, RGB,
  deduplikasi lintas split, arsip berbahaya, keluaran tidak menimpa data.
- [x] Implementasi scripts/prepare_data.py dan manifest audit sumber Kaggle.
- [x] Tambahkan tes dataset: normalisasi warna, grayscale, augmentasi train,
  test deterministik, panjang domain berbeda, worker Windows dan input rusak.
- [x] Implementasi cyclegan/data.py dan ekspor publik.
- [x] Jalankan pipeline asli, seluruh tes dan batch nyata pada empat model baseline.
- [x] Minta review Claude Opus 5.5 High; catat hasil atau kegagalan secara jujur.
- [x] Perbarui README, kontrak, laporan, log dan bukti orang 2.

Risiko: deduplikasi ini mendeteksi byte/piksel identik, belum crop/encoding berbeda.
Evaluasi pada subset cleaned harus dinyatakan terfilter, bukan seluruh benchmark asli.

Hasil: 2.608 RGB256 (1.040/1.311/118/139), 21 tes lulus, seluruh epoch loader dan
batch nyata empat model default diperiksa. Claude Opus 5.5 High memverifikasi
syarat review telah ditangani, tanpa bug penting yang menghalangi handoff.
