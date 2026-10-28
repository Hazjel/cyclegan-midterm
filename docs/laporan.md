# Laporan Proyek Kelompok: CycleGAN Kuda ↔ Zebra

Tema tugas: **Vibe Coding Generative AI from Scratch**.

**Status:** draf laporan akademik. Rancangan, implementasi fondasi model dan pipeline data tersedia. Pelatihan, tuning, dan evaluasi kualitas belum dilakukan. Hasil pengujian teknis yang disajikan berasal dari bukti 27 Oktober 2026; penyuntingan dokumentasi dilakukan 28 Oktober 2026.

## 1. Identitas dan kontribusi kelompok

| Identitas | Isian |
|---|---|
| Mata kuliah | [NAMA MATA KULIAH] |
| Program studi / institusi | [PROGRAM STUDI DAN INSTITUSI] |
| Kelas / kelompok | [KELAS] / [NAMA ATAU NOMOR KELOMPOK] |
| Dosen pengampu | [NAMA DOSEN] |
| Anggota | [NAMA DAN NIM SETIAP ANGGOTA] |

Pembagian awal terdiri atas Peran 1 (Prompt Engineer & LLM Navigator), Peran 2 (Data Pipeline & Preprocessing Engineer), Peran 3 (Training Loop & Hyperparameter Tuning Engineer), dan Peran 4 (Lead Documentarian & Error Handler).

Kontribusi aktual yang diklarifikasi: pelaksana Peran 1 dan 4 juga mengambil alih pengerjaan teknis Peran 2 dengan bantuan LLM. Nama/NIM pelaksana: [BELUM DIISI]. Penanggung jawab awal Peran 2 dan penanggung jawab Peran 3: [BELUM DIISI]. Implementasi training Peran 3 belum tersedia. Sebutan “orang 2” dalam nama berkas dan arsip menunjukkan lingkup tugas, bukan identitas pembuat implementasi. Kontribusi anggota lain, pembagian waktu, persetujuan kelompok, dan refleksi pribadi yang belum diberikan tidak diasumsikan.

## 2. Pendahuluan dan tujuan

Proyek ini merupakan tugas kelompok untuk membangun model generatif dari layer dasar melalui metode vibe coding. Permasalahan yang dipilih adalah penerjemahan gambar kuda menjadi zebra dan sebaliknya tanpa pasangan gambar satu-ke-satu.

Tujuan teknisnya adalah merancang empat model CycleGAN, menyediakan pipeline data yang konsisten, menguji integrasi, dan kemudian melatih serta mengevaluasi penerjemahan gambar. Ruang lingkup yang telah diimplementasikan mencakup fondasi arsitektur, konfigurasi bersama, pipeline, kontrak integrasi dan dokumentasi. Keberhasilan penerjemahan belum dapat disimpulkan sebelum pelatihan dan evaluasi.

## 3. Metode pengembangan dengan LLM

Panduan tugas mengharuskan kolaborasi dengan LLM, pencatatan prompt/respons dan debugging, serta melarang penyalinan implementasi lengkap dari tutorial/GitHub. Dokumentasi resmi diperbolehkan sebagai referensi fungsi dasar. Identitas dan hash panduan tercatat dalam [inventaris](bukti/lingkungan.json).

Proses yang tercatat meliputi perumusan kebutuhan dan batas peran melalui prompt, pemeriksaan berkas, perancangan antarmuka, penyusunan kode dengan bantuan LLM, pengujian, serta penanganan masalah. API dasar diperiksa melalui Context7. Model menggunakan layer PyTorch dan autograd tanpa pretrained weights. Uraian proses ini bukan audit orisinalitas menyeluruh terhadap semua sumber eksternal.

[Log interaksi](log_interaksi.md), [prompt tugas awal](bukti/prompt_pengguna.md), [prompt tahap pipeline](bukti/prompt_orang2.md), dan [diskusi teknis Claude](diskusi_claude.md) mempertahankan bukti proses. Review AI tidak diperlakukan sebagai persetujuan dosen atau kontribusi anggota manusia. Ekspor chat lengkap/share link dan penjelasan penguasaan materi masing-masing anggota: [PERLU DILENGKAPI KELOMPOK].

## 4. Dataset dan provenance

Dataset aktif ditempatkan pada `data/horse2zebra` dan dibagikan terpisah dari Git. Paket final memuat trainA/trainB/testA/testB, manifest, metadata terkoreksi dan preparation report. Lokasi distribusi paket final: [BELUM DIBERIKAN].

| Split | Domain | Jumlah akhir |
|---|---|---:|
| trainA | kuda | 1.040 |
| trainB | zebra | 1.311 |
| testA | kuda | 118 |
| testB | zebra | 139 |
| Total | | 2.608 |

Seluruh gambar akhir berupa PNG RGB 256 × 256. Manifest SHA-256 adalah `fe9d4e3a940fb9167fead81b21123ed4b12e09487e3a336a695f7d0681e38f2f`; manifest memuat sampel runtime, sedangkan metadata terkoreksi mencatat seluruh 2.661 gambar sumber.

Sumber yang diarsipkan adalah [Kaggle Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), versi 1. [Metadata sumber](bukti/kaggle_metadata.json) mengakui sumber UC Berkeley dan meminta sitasi CycleGAN (Zhu et al., ICCV 2017). Lisensi API tercatat sebagai `Other (specified in description)`, tanpa klaim lisensi lebih spesifik.

Riwayat persiapan menggunakan `preproc.zip`, SHA-256 `1dd5f0529247324f986cbceeaf260bb6712ef24013d9527dba57b85d0def3805`, serta ZIP Kaggle, SHA-256 `0655d8feb8eb83911ac08c1d230a0464794418c107412a28adc47e1eb69e90a0`. Pada tahap tersebut, seluruh 2.661 gambar asli dibandingkan byte demi byte. Kandidat cleaned berjumlah 2.633; 28 gambar blur sudah dipisahkan. Persiapan final memperbaiki 240 baris metadata, mereplikasi kanal tujuh grayscale menjadi RGB, dan mengecualikan 25 salinan piksel identik dengan prioritas mempertahankan test.

Notebook awal hanya memeriksa blur dan menyalin gambar menurut threshold 100. Konversi RGB untuk tampilan plot belum mengubah berkas saat itu. Pemeriksaan ulang 2.661 skor blur tercatat tanpa mismatch. [Audit awal](bukti/audit_dataset.json), [preparation report](bukti/preparation_report.json), dan [pemeriksaan blur](bukti/verifikasi_blur.json) dipertahankan sebagai bukti historis.

Penghapusan `preproc.zip` dilaporkan pelaksana proyek dan dikonfirmasi pada sesi 27 Oktober 2026. Loader kini membaca folder final tanpa ZIP. Penghapusan ini tidak mengubah riwayat bahwa dataset dibuat dari arsip, tetapi membatasi reproduksi ulang audit preprocessing sumber.

## 5. Rancangan sistem

### 5.1 Arsitektur model

Domain A=kuda dan B=zebra. G_AB menerjemahkan A→B, G_BA menerjemahkan B→A, sedangkan D_A/D_B menilai real/fake di domain masing-masing. Satu kelas generator dan satu kelas discriminator digunakan untuk membuat dua instance independen masing-masing.

| Komponen | Rancangan baseline dan alasan |
|---|---|
| Generator | Conv 7×7, dua downsampling Conv 3×3 stride 2, sembilan residual block, dua ConvTranspose 3×3 stride 2, Conv 7×7 dan Tanh |
| Kanal generator | 3→64→128→256→128→64→3; jumlah fitur bertambah saat resolusi berkurang |
| Residual block | Dua Conv 3×3 dengan InstanceNorm, ReLU setelah conv pertama, dan penjumlahan x+F(x) untuk menjaga jalur identitas fitur |
| Padding / upsampling | Reflection padding pada conv 7×7 dan residual; ConvTranspose memakai padding 1 dan output_padding 1 agar shape kembali tepat untuk H/W kelipatan empat |
| Normalisasi | InstanceNorm non-affine, tanpa running stats, eps=1e-5; statistik per gambar mendukung batch satu |
| Discriminator | Lima Conv 4×4 padding 1, stride 2/2/2/1/1, kanal 3→64→128→256→512→1; LeakyReLU 0,2 dan InstanceNorm di lapisan tengah |
| Output | Tanh generator menghasilkan [-1,1]; discriminator menghasilkan skor mentah 30×30 tanpa sigmoid untuk input 256×256 |
| Inisialisasi | Convolution memakai normal(0,0.02), bias nol; tidak menggunakan dropout atau bobot pretrained |

Receptive field konvolusional nominal discriminator adalah 70×70. InstanceNorm menggunakan statistik spasial sehingga nominal tersebut bukan klaim independensi spasial penuh. Jumlah parameter per generator=11.378.179 dan per discriminator=2.764.737.

Filter, jumlah block, normalisasi, inisialisasi, dan upsampling adalah baseline yang belum dituning. Inventaris historis mencatat RTX 3060 Laptop 6 GiB; kebutuhan memori training empat model belum diukur. Tidak ada estimasi durasi training atau klaim kualitas berdasarkan inventaris tersebut.

### 5.2 Rancangan data dan integrasi

Batch train berupa dictionary A/B, masing-masing RGB float32 `[N,3,256,256]`, finite, dalam [-1,1]. Dataset unpaired tidak mengharuskan A[i] dan B[i] menggambarkan objek yang sama. TestA/testB dimuat terpisah secara deterministik.

Konfigurasi arsitektur diberi versi dan disimpan bersama empat state_dict agar rekonstruksi model tidak hanya mengandalkan kesamaan shape parameter. Aturan output discriminator, loss yang kompatibel, dan payload checkpoint dijelaskan dalam [kontrak integrasi](kontrak_integrasi.md). Desain checkpoint ini belum merupakan implementasi resume training.

## 6. Implementasi yang tersedia

`cyclegan/models.py` mengimplementasikan residual block, Generator dan PatchDiscriminator. `cyclegan/config.py` menyediakan dataclass immutable dan validasi konfigurasi. Setiap konstruktor mengalokasikan parameter baru; forward memvalidasi shape, dtype, rentang dan finite tanpa transformasi tersembunyi.

`cyclegan/data.py` membaca folder final berdasarkan manifest. Augmentasi train berlangsung saat loading: resize bicubic 286, crop acak 256, flip horizontal 0,5, kemudian normalisasi `2*(pixel/255)-1`. Transform domain independen. Panjang epoch=max(trainA,trainB)=1.311; domain besar dikunjungi sekali dan domain kecil disampling acak dengan pengembalian. Test tidak memakai transformasi acak. Tidak ada gambar augmentasi offline baru.

`scripts/check_data.py` memeriksa dataset final, mengiterasi loader dan menjalankan forward batch nyata pada empat model. `prepare_data.py` dan `audit_archive.py` dipertahankan sebagai utilitas ZIP opsional. Panduan instalasi, penempatan dataset dan penggunaan terdapat dalam [README](../README.md). Status teknis pipeline dirangkum dalam [dokumen Peran 2](status_orang2.md).

Loss CycleGAN, optimizer, training loop, checkpoint pelatihan dan tuning **belum diimplementasikan**. Tanggung jawab tersebut tetap berada pada Peran 3.

## 7. Pengujian teknis dan hasil aktual

Bukti pengujian 27 Oktober 2026 menggunakan Python 3.12.3, PyTorch 2.5.1+cu121, dan Pillow 12.3.0. Tes model memakai CPU, empat thread dan seed 7. Hasil tersimpan: **21 tes lulus** (tujuh model, tiga audit, sebelas data). Sesi penyuntingan 28 Oktober 2026 memeriksa dokumentasi tanpa mengulang suite model atau training.

| Pemeriksaan | Hasil yang tercatat |
|---|---|
| Import, shape dan rentang generator | Dua G menerima 256×256, output [1,3,256,256], finite dan dalam [-1,1] |
| Discriminator | Dua D menghasilkan [1,1,30,30], finite |
| Backward dummy | Gradient input dan parameter tersedia dan finite; tidak seluruhnya nol |
| Independensi instance | Parameter/storage terpisah; mutasi satu instance tidak mengubah pasangan |
| Konfigurasi / state_dict | Round-trip dalam memori mempertahankan output |
| Validasi dan pipeline | Input invalid, gambar rusak, metadata ambigu, traversal dan konflik domain terdeteksi; worker Windows diperiksa |
| Dataset nyata | Seluruh 2.608 gambar RGB 256×256 cocok dengan hash manifest, tanpa hash piksel identik yang tersisa |
| Epoch loader | Train=1.311 pasangan/328 batch; testA=118/30 batch; testB=139/35 batch pada batch size 4 |
| Batch nyata → model | Kedua generator dan discriminator default menerima batch nyata di CPU, output finite |

Panjang epoch train tidak menjamin semua sampel domain kecil terpilih pada epoch yang sama karena sampling dilakukan dengan pengembalian. Pemeriksaan hash piksel adalah pemeriksaan baca-saja, bukan persiapan ulang dataset.

Bukti: [log tes](bukti/tes_dataset_final_tanpa_zip.txt), [pemeriksaan data/model](bukti/pemeriksaan_dataset_final_tanpa_zip.json), [preservasi dataset](bukti/preservasi_dataset_final_tanpa_zip.json), dan [pemeriksaan fondasi awal](bukti/pemeriksaan.txt). Riwayat kegagalan awal dan perbaikannya tetap tersedia dalam log interaksi. Hasil teknis ini belum membuktikan kualitas penerjemahan, kestabilan training adversarial, atau kecukupan memori training.

## 8. Rencana dan hasil eksperimen pelatihan

**Belum dilakukan — [PERLU DILENGKAPI PERAN 3].**

Bagian ini perlu memuat formulasi dan bobot loss adversarial/cycle/identity, optimizer, learning rate/scheduler, jumlah epoch, batch size, seed, urutan pembaruan G/D, konfigurasi data/model aktual, perangkat serta pemakaian memori. Setiap eksperimen harus mempunyai konfigurasi, alasan perubahan dan log yang dapat ditelusuri.

Hasil yang belum tersedia: grafik loss, metrik beserta protokol evaluasi, perbandingan tuning, checkpoint/resume, contoh penerjemahan dua arah dan rekonstruksi cycle, serta analisis kegagalan pada gambar nyata. Tensor dummy tidak dipakai sebagai contoh hasil training. Jika metrik kelak menggunakan jaringan pretrained, bedakan jaringan metrik dari model CycleGAN yang dilatih dari nol dan cocokkan penggunaannya dengan aturan tugas.

Peran 3 perlu mengambil validasi dari train sebelum tuning/pemilihan checkpoint. Test digunakan untuk evaluasi akhir, bukan keputusan hyperparameter. Belum ada keputusan split validasi yang dilaporkan.

## 9. Pembahasan dan keterbatasan

Fondasi model dan pipeline dapat diintegrasikan melalui kontrak v1.1 dan memiliki bukti pengujian CPU. Masalah grayscale, metadata salah dan duplikasi piksel identik yang diketahui telah ditangani pada tahap persiapan.

Evaluasi kelak berlaku untuk subset test yang telah disaring, bukan benchmark Kaggle penuh. Hash piksel belum mendeteksi semua kemiripan atau crop berbeda. Kapasitas training belum diukur, konfigurasi belum dituning, dan kualitas hasil belum dapat dinilai. Arsip preprocessing yang dihapus membatasi reproduksi penuh tahap sumber meskipun folder final tetap dapat diperiksa.

Pembahasan eksperimen: [PERLU DILENGKAPI PERAN 3]. Refleksi pribadi, penguasaan materi, catatan kontribusi manusia terperinci, dan keputusan bersama: [PERLU DILENGKAPI KELOMPOK]. Keberadaan repository Git tidak dengan sendirinya membuktikan atribusi kontribusi individual; tidak ada commit baru dibuat sebagai bagian penyuntingan ini.

## 10. Kesimpulan sementara dan kelengkapan laporan

Rancangan dan implementasi fondasi CycleGAN serta pipeline data telah tersedia dan didukung bukti pengujian teknis. Keberhasilan penerjemahan gambar belum dapat disimpulkan karena pelatihan dan evaluasi kualitas belum dilakukan.

Lampiran tersedia: prompt, log interaksi, respons review AI, kode, tes, audit, metadata sumber, manifest/preparation report dan hasil pemeriksaan. Informasi yang belum tersedia: identitas akademik/anggota, lokasi distribusi dataset final, ekspor percakapan lengkap/share link, bukti kontribusi manusia terperinci, refleksi kelompok dan seluruh hasil eksperimen Peran 3.

## 11. Referensi dan lampiran

1. *[CAK4KBB3 2026-1] Panduan Proyek 1 — Vibe Coding*, dokumen tugas lokal; identitas/hash pada [inventaris](bukti/lingkungan.json).
2. [Horse2zebra Dataset, Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), metadata versi 1 tersimpan pada [arsip sumber](bukti/kaggle_metadata.json). Metadata mengacu pada Zhu, Park, Isola, Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
3. Referensi fungsi dasar: [PyTorch](https://docs.pytorch.org/docs/stable/index.html) dan [Pillow](https://pillow.readthedocs.io/en/stable/). Penelusuran API melalui Context7 dicatat dalam log.
4. [Kontrak integrasi](kontrak_integrasi.md), [status pipeline](status_orang2.md), [log interaksi](log_interaksi.md), dan [diskusi LLM](diskusi_claude.md).

Format sitasi dan kelengkapan daftar pustaka sesuai ketentuan mata kuliah: [PERLU DILENGKAPI KELOMPOK].
