# Status orang 2: Data Pipeline & Preprocessing

Catatan penggunaan terbaru, 27 Oktober 2026: pakai langsung `data/horse2zebra` melalui `create_dataloaders` atau `scripts/check_data.py`. Pengguna menyatakan `preproc.zip` telah dihapus dan ketidakberadaannya terverifikasi. ZIP bukan prasyarat runtime. Pernyataan tentang dua arsip sumber di bawah adalah riwayat sesi persiapan; hash, audit dan bukti saat itu tetap dipertahankan. Pemeriksaan sesudah penghapusan disimpan terpisah pada [bukti baru](bukti/pemeriksaan_dataset_final_tanpa_zip.json).

Panduan proyek dibaca dari PDF lokal, halaman 2. Orang 2 bertugas menyiapkan dan menyuplai dataset, augmentasi, normalisasi, DataLoader, serta memastikan shape sesuai arsitektur. Implementasi tugas tersebut sudah tersedia; bukti verifikasi dan batas evaluasinya di bawah.

## Verifikasi screenshot dan penyelesaian

Screenshot menggambarkan kondisi awal arsip dengan benar. Kode awal belum memiliki modul pipeline. Audit ulang disimpan pada [audit awal](bukti/audit_orang2_awal.json).

| Bagian tugas | Kondisi awal terverifikasi | Hasil sesi orang 2 |
|---|---|---|
| Dataset dan folder train/test | Ada dalam ZIP | 2.661 gambar dibandingkan byte demi byte dengan unduhan Kaggle |
| Pemeriksaan/pemisahan blur | Ada, threshold 100 | Seluruh 2.661 skor dihitung ulang; cocok dengan CSV, 28 dikecualikan |
| Ukuran seragam | Semua cleaned 256 x 256 | Semua 2.608 berkas akhir 256 x 256 |
| Input RGB | Tujuh grayscale trainB | Disimpan sebagai PNG RGB; kanal grayscale direplikasi, tidak diwarnai secara artifisial |
| Augmentasi train | Belum ada | Bicubic resize 286, crop acak 256, flip horizontal 0,5 |
| Tensor/normalisasi | Belum ada | RGB CHW float32, CPU, finite, `2*(pixel/255)-1` |
| Dataset/DataLoader | Belum ada | Unpaired train; testA/testB terpisah, batch terakhir tetap disertakan |
| Kecocokan arsitektur | Belum diuji dari data asli | Batch asli diterima dua G dan dua D default; seluruh output finite |

Tambahan yang diperlukan untuk handoff aman: 240 baris metadata diperbaiki berdasarkan folder sumber; 25 salinan piksel identik dikecualikan dengan prioritas test. Tidak ada piksel identik yang tersisa di seluruh manifest final, termasuk lintas train/test.

## Dataset siap pakai

Folder: `D:\genai mid\data\horse2zebra`.

| Split | Domain | Jumlah akhir |
|---|---|---:|
| trainA | kuda | 1.040 |
| trainB | zebra | 1.311 |
| testA | kuda | 118 |
| testB | zebra | 139 |
| Total | | 2.608 |

Manifest SHA-256: `fe9d4e3a940fb9167fead81b21123ed4b12e09487e3a336a695f7d0681e38f2f`.

Sumber [Kaggle Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), versi 1, lastUpdated `2020-10-17T17:12:45.7Z`. Metadata API tersimpan pada [kaggle_metadata.json](bukti/kaggle_metadata.json). Deskripsi sumber mengakui dataset dari direktori CycleGAN UC Berkeley dan meminta sitasi Zhu, Park, Isola, Efros (ICCV 2017). Kolom lisensi API berbunyi `Other (specified in description)`; tidak mengklaim lisensi bebas yang lebih spesifik.

ZIP Kaggle SHA-256: `0655d8feb8eb83911ac08c1d230a0464794418c107412a28adc47e1eb69e90a0`. CLI persiapan menolak sumber dengan hash berbeda. Fungsi `prepare_data` adalah fungsi inti untuk pasangan ZIP, juga dipakai fixture sintetis tes; pin sumber asli diterapkan pada CLI.

Kesalahan 240 path/domain juga ada di CSV Kaggle asli. Kedua ZIP sumber tidak dimodifikasi. Metadata yang dikoreksi memuat seluruh 2.661 gambar sumber; manifest memuat hanya 2.608 sampel yang dipakai. Daftar pengecualian menghubungkan keduanya.

## Handoff ke orang 3

- Pakai `from cyclegan import create_dataloaders` dengan contoh pada [README](../README.md) dan [kontrak v1.1](kontrak_integrasi.md).
- Batch train: dict A/B, masing-masing `[N,3,256,256]`. Panjang epoch 1.311; domain besar dikunjungi sekali, domain kecil disampling acak dengan pengembalian.
- Loader testA dan testB menghasilkan tensor domain tunggal; jalankan terpisah, tanpa zip atau pengulangan sampel. Satu epoch yang diuji mencakup seluruh 118 kuda dan 139 zebra.
- Panggil `torch.manual_seed` sebelum iterasi untuk num_workers=0. Parameter seed loader mengatur shuffle dan seed worker. Pada Windows, gunakan main guard bila worker > 0.
- Simpan manifest/hash, konfigurasi transform, seed dan state RNG pada checkpoint. Ambil validasi dari train sebelum tuning/pemilihan checkpoint; test hanya evaluasi akhir.

## Bukti pemeriksaan

- [Tes lengkap](bukti/tes_orang2.txt): 21 tes model/audit/data, termasuk worker Windows, gambar corrupt, metadata ambigu, sumber berbeda byte, traversal dan domain konflik.
- [Pemeriksaan data asli](bukti/pemeriksaan_data.json): seluruh 2.608 file dicek mode, ukuran, hash file/piksel; satu epoch train/test diperiksa; empat model default menerima batch nyata di CPU.
- [Audit persiapan](bukti/preparation_report.json), [manifest final](bukti/manifest.json), [metadata terkoreksi](bukti/metadata_corrected.csv).
- [Blur dihitung ulang](bukti/verifikasi_blur.json): 2.661 skor cocok, tidak ada mismatch.
- [Diskusi Claude](diskusi_claude.md): Claude Opus 5.5, effort high, diskusi awal dan review implementasi. Review AI membantu pemeriksaan teknis, bukan bukti kontribusi anggota manusia atau persetujuan dosen.

Training, checkpoint dan kualitas hasil belum diuji; itu pekerjaan orang 3. Test adalah subset cleaned, sehingga tidak boleh dilaporkan sebagai benchmark Kaggle penuh. Hash piksel tidak mendeteksi semua gambar serupa yang di-encode/crop ulang. Ekspor chat lengkap dan identitas anggota tetap perlu dilampirkan oleh kelompok.
