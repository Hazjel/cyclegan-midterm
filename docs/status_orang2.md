# Status Teknis Peran 2: Pipeline Data dan Preprocessing

## Lingkup dan atribusi kontribusi

Peran 2 pada pembagian awal mencakup penyiapan dataset, preprocessing, augmentasi, normalisasi, dan DataLoader. **Pengerjaan teknis lingkup ini diambil alih oleh pelaksana Peran 1 dan 4**, sesuai klarifikasi kontribusi aktual. Nama dan NIM pelaksana: [NAMA/NIM PELAKSANA PERAN 1 DAN 4]. Penanggung jawab awal Peran 2: [NAMA/NIM BELUM DIISI].

Nama berkas ini dipertahankan untuk kesinambungan tautan dan log. Sebutan “orang 2” pada arsip menunjukkan lingkup tugas; implementasi tidak diatribusikan kepada anggota tertentu hanya berdasarkan label tersebut. Kontribusi manusia lain yang belum dilaporkan tidak diasumsikan. Rincian penugasan awal terdapat dalam [README](../README.md).

## Status implementasi

Pipeline tersedia pada `cyclegan/data.py`. Root aktif adalah `data/horse2zebra`; dataset dibagikan terpisah dan tidak disertakan dalam Git. Paket final harus memuat empat folder gambar serta `manifest.json`, `metadata_corrected.csv`, dan `preparation_report.json`. Lokasi distribusi paket final: [BELUM DIBERIKAN].

| Komponen | Implementasi atau hasil yang tersedia |
|---|---|
| Dataset final | 2.608 PNG RGB 256 × 256 |
| Koreksi metadata | 240 baris path/domain diperbaiki berdasarkan folder sumber dan image_id |
| Penyaringan blur | 28 gambar dengan skor <100 dikecualikan |
| Duplikasi piksel identik | 25 salinan dikecualikan dengan prioritas mempertahankan test |
| Konversi RGB | Tujuh grayscale dipertahankan sebagai RGB dengan kanal identik |
| Augmentasi train | Resize bicubic 286, crop acak 256, flip horizontal peluang 0,5 saat loading |
| Tensor | RGB, CHW, float32 CPU, finite, normalisasi `2*(pixel/255)-1` |
| Sampling train | Unpaired; panjang epoch 1.311, domain kecil disampling dengan pengembalian |
| Evaluasi | Loader testA/testB terpisah dan deterministik; batch terakhir disertakan |
| Integrasi model | Batch nyata diterima dua generator dan dua discriminator default di CPU |

| Split final | Domain | Jumlah |
|---|---|---:|
| trainA | kuda | 1.040 |
| trainB | zebra | 1.311 |
| testA | kuda | 118 |
| testB | zebra | 139 |
| Total | | 2.608 |

Manifest SHA-256: `fe9d4e3a940fb9167fead81b21123ed4b12e09487e3a336a695f7d0681e38f2f`. Manifest memuat sampel runtime, sedangkan metadata terkoreksi memuat seluruh 2.661 gambar sumber. Daftar pengecualian dalam preparation report menghubungkan keduanya.

## Riwayat persiapan dan provenance

Audit awal menemukan 2.633 kandidat cleaned, tujuh grayscale, metadata salah lokasi/domain, serta duplikasi termasuk satu pasangan lintas train/test. Kondisi ini terdokumentasi pada [audit awal](bukti/audit_orang2_awal.json) dan kemudian ditangani dalam persiapan dataset final.

Sumber yang dicatat adalah [Kaggle Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), versi 1, lastUpdated `2020-10-17T17:12:45.7Z`. [Metadata sumber](bukti/kaggle_metadata.json) mengakui direktori CycleGAN UC Berkeley dan meminta sitasi Zhu, Park, Isola, Efros (ICCV 2017). Kolom lisensi API berbunyi `Other (specified in description)`; tidak ditafsirkan sebagai lisensi bebas yang lebih spesifik.

Pada tahap persiapan, 2.661 gambar asli dibandingkan byte demi byte dengan arsip lokal. Kedua ZIP tidak dimodifikasi selama proses tersebut. ZIP Kaggle SHA-256 `0655d8feb8eb83911ac08c1d230a0464794418c107412a28adc47e1eb69e90a0` dipin pada CLI `prepare_data.py`; fungsi intinya juga digunakan untuk fixture sintetis tes. Koreksi metadata dan pengecualian tidak mengubah sumber pada saat persiapan.

Penghapusan `preproc.zip` dilaporkan pelaksana proyek dan ketidakberadaannya dikonfirmasi pada sesi 27 Oktober 2026. Catatan ini terpisah dari riwayat persiapan sebelumnya. Loader dan `check_data.py` tetap menggunakan folder final tanpa ZIP. `prepare_data.py` dan `audit_archive.py` adalah utilitas opsional berbasis arsip; reproduksi penuh terhadap preprocessing sumber memerlukan ZIP asli yang kini tidak tersedia.

## Integrasi ke Peran 3

Gunakan `create_dataloaders("data/horse2zebra")` sesuai [README](../README.md) dan [kontrak v1.1](kontrak_integrasi.md). Batch train berupa dictionary A/B, masing-masing float32 `[N,3,256,256]` dalam [-1,1]. Transformasi domain dilakukan independen dan tidak menghasilkan berkas augmentasi offline.

Loader testA/testB menghasilkan tensor domain tunggal. Keduanya diiterasi terpisah tanpa `zip` atau pengulangan agar seluruh 118 kuda dan 139 zebra tercakup. Untuk `num_workers=0`, seed torch ditetapkan sebelum iterasi; worker Windows memerlukan main guard.

Peran 3 perlu menyimpan manifest/hash, konfigurasi transform, seed dan state RNG bersama checkpoint, serta mengambil validasi dari train sebelum tuning atau pemilihan checkpoint. Pipeline siap diintegrasikan; implementasi training, checkpoint dan evaluasi kualitas belum tersedia.

## Bukti verifikasi

Bukti berikut berasal dari pemeriksaan 27 Oktober 2026, bukan eksekusi ulang pada sesi penyuntingan dokumentasi:

- [Tes pipeline dan model](bukti/tes_orang2.txt): 21 tes, termasuk worker Windows, gambar rusak, metadata ambigu, sumber berbeda byte, traversal dan konflik domain.
- [Pemeriksaan data/model tanpa ZIP](bukti/pemeriksaan_dataset_final_tanpa_zip.json) dan [log tes](bukti/tes_dataset_final_tanpa_zip.txt): seluruh file/hash/piksel serta epoch train/test diperiksa; empat model default menerima batch nyata.
- [Preservasi dataset](bukti/preservasi_dataset_final_tanpa_zip.json): hash dataset, kode dan bukti historis tidak berubah pada sesi penyesuaian path.
- [Audit persiapan](bukti/preparation_report.json), [manifest final](bukti/manifest.json), dan [metadata terkoreksi](bukti/metadata_corrected.csv).
- [Pemeriksaan ulang blur](bukti/verifikasi_blur.json): 2.661 skor cocok dengan CSV, tanpa mismatch.
- [Diskusi teknis LLM](diskusi_claude.md): respons Claude Opus 5.5 High dipertahankan sebagai bukti proses, bukan persetujuan dosen atau bukti kontribusi anggota manusia.

## Keterbatasan

Test adalah subset cleaned, sehingga hasil kelak tidak mewakili benchmark Kaggle penuh. Hash piksel tidak mendeteksi semua gambar serupa atau crop berbeda. Pemeriksaan fondasi/pipeline belum membuktikan kualitas penerjemahan atau kapasitas memori training. Identitas, catatan kontribusi manusia terperinci, refleksi kelompok, dan ekspor chat lengkap masih perlu dilengkapi.
