# Rencana fondasi CycleGAN

Tujuan: menyelesaikan Peran 1 dan 4 sesuai instruksi pengguna, langsung di workspace ini.
Spesifikasi: permintaan pengguna pada sesi 27 Oktober 2026 dan panduan lokal lima halaman.
Eksekusi langsung sudah diminta pengguna; tidak ada tahap persetujuan desain tambahan.

## Batas dan keputusan

- A=kuda, B=zebra; dua generator dan dua discriminator dibuat sebagai instance independen.
- Python/PyTorch layer dasar, tanpa pretrained weights atau implementasi lengkap dari luar.
- `cyclegan/config.py`: dataclass immutable, dapat diserialisasi untuk checkpoint.
- `cyclegan/models.py`: generator residual, PatchGAN, validasi input, inisialisasi acak.
- `cyclegan/__init__.py`: ekspor publik untuk anggota lain.
- `tests/test_models.py`: tensor dummy, unittest tanpa dependensi test tambahan.
- `scripts/audit_archive.py`: audit baca-saja, bukan pipeline data.
- `README.md`, `docs/kontrak_integrasi.md`, `docs/log_interaksi.md`, `docs/laporan.md`, `docs/bukti/`: handoff dan bukti nyata.
- Tidak ada caller atau model lama; workspace awal hanya arsip dan beberapa gambar di tmp.
- Arsip asli dan notebook tidak diubah/dijalankan; tidak ada training, loss pelatihan, atau DataLoader.

## Langkah dan bukti penerimaan

- [x] Tulis tes kontrak sebelum implementasi dan rekam kegagalan karena modul belum ada.
- [x] Implementasikan baseline 64 filter, 9 residual block; PatchGAN 5 convolution dengan stride 2,2,2,1,1. InstanceNorm non-affine, ReLU/LeakyReLU, Tanh generator, tanpa sigmoid discriminator; normal(0,0.02).
- [x] Jalankan unittest CPU: import; input 256; shape/range/finite; backward semua model; bobot terpisah; input tidak valid; konfigurasi dan state_dict dapat dipulihkan.
- [x] Audit ZIP: hitung split, decode cleaned, mode RGB/L, duplikasi salinan terhadap data asli, metadata dan laporan blur, sumber notebook tanpa eksekusi. Simpan JSON dan self-check terhadap kondisi yang ditemukan.
- [x] Tulis kontrak batch dan checkpoint serta dokumentasi; tautkan hasil pemeriksaan nyata, tandai bukti training yang belum tersedia.
- [x] Tinjau ulang file, perintah README, cakupan permintaan, dan hasil tes sebelum handoff.

## Fokus pemeriksaan

Input non-RGB/uint8/NaN/di luar rentang harus ditolak; dimensi bukan kelipatan 4 ditolak agar shape tidak diam-diam berubah; konfigurasi rusak ditolak; checkpoint menyimpan versi arsitektur; parameter antarmodel tidak berbagi storage.

Hardware terdeteksi: RTX 3060 Laptop 6 GiB. Ini bukti inventaris, bukan bukti kapasitas training atau konfigurasi optimal. Pengujian wajib tetap CPU. Tidak membuat Git history karena workspace belum berupa repositori dan riwayat anggota belum diberikan.

## Hasil eksekusi

Sepuluh tes lulus (7 model, 3 audit). Import publik dan audit CLI lulus. Assertion atas audit dataset, SHA-256 arsip tidak berubah, dan tautan dokumen lokal lulus. Review independen AI mengulang 10 tes dan audit dalam memori, tanpa temuan yang perlu ditindaklanjuti. Temuan data diteruskan dalam kontrak: 240 path metadata salah lokasi, 7 grayscale, 25 kelompok duplikasi byte dengan satu lintas train/test.

Belum diperiksa: pemasangan environment baru, eksekusi model di GPU, kebutuhan memori dan kualitas training. Implementasi pipeline/training tetap untuk orang 2/3.
