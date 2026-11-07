# Laporan Proyek Kelompok: CycleGAN Kuda ↔ Zebra

**Pembaruan integrasi 3 November 2026:** commit remote anggota lain `5fe225b` menambahkan `scripts/train_cyclegan.py`, `scripts/evaluate_cyclegan.py` dan [panduan training](training_lokal.md). Skrip dipertahankan bersama penataan preprocessing. Pernyataan belum ditemukannya implementasi pada inventaris 31 Oktober di bawah adalah kondisi lokal saat itu, bukan status kode terkini. Default CLI yang tersedia adalah LSGAN-MSE, cycle L1 berbobot 10, identity L1 berbobot 5, Adam lr=0,0002 dengan betas=(0,5;0,999), 100 epoch, batch 1, seed 7, dan holdout train 5%. Ini konfigurasi kode, bukan konfigurasi run yang sudah dibuktikan. Gunakan `--dataset-root data/horse2zebra` secara eksplisit. Hasil training, checkpoint/resume dan kualitas terjemahan belum diverifikasi pada sesi integrasi; bagian eksperimen tetap perlu dilengkapi.

Tema tugas: **Vibe Coding Generative AI from Scratch**.

**Status:** draf kompilasi kelompok, diperbarui 31 Oktober 2026. Fondasi model dan pipeline tersedia. Inventaris kode, folder dan daftar isi arsip lokal belum menemukan implementasi atau hasil pelatihan yang dapat diverifikasi; kegiatan di luar workspace perlu dikonfirmasi Peran 3. Bukti tes/data berasal dari 27 Oktober 2026; penyuntingan 28 Oktober dan kompilasi sekarang tidak mengubah tanggal hasil tersebut.

## 1. Identitas dan kontribusi kelompok

| Identitas | Isian |
|---|---|
| Mata kuliah | Generative AI |
| Program studi / institusi | S1 Informatika / Telkom University |
| Kelas / kelompok | GENERATIVE AI BS1IF-47-GAB03 / Kelompok 2 |
| Anggota | Jian Hazel Sitorus (103012300187); Mikhael Restu Mahardhika (103012300277); Reevanza Abel Desta Arifin (103012330104) |

Penanggung jawab Peran 1 (Prompt Engineer & LLM Navigator) adalah Jian Hazel Sitorus (103012300187); Peran 2 (Data Pipeline & Preprocessing Engineer) adalah Mikhael Restu Mahardhika (103012300277); dan Peran 3 (Training Loop & Hyperparameter Tuning Engineer) adalah Reevanza Abel Desta Arifin (103012330104). Dokumentasi awal Peran 4 (Lead Documentarian & Error Handler) ditangani Jian Hazel Sitorus, kemudian dikerjakan bersama anggota tim.

Jian Hazel Sitorus membantu prapemrosesan awal untuk sementara, lalu kelanjutannya kembali menjadi tanggung jawab Mikhael Restu Mahardhika. Reevanza Abel Desta Arifin menangani pelatihan model dan penyetelan hiperparameter. Identitas dan penjelasan bantuan sementara ini diselaraskan dengan laporan Word pada 7 November 2026. Nama peran dalam arsip menunjukkan lingkup tugas; rincian kontribusi teknis, bukti eksperimen, dan penyuntingan setiap anggota tetap perlu ditelusuri melalui bukti aktual. [Koordinasi dokumentasi](koordinasi_dokumentasi.md) mencatat kebutuhan bukti serta pembagian dokumentasi bersama. [PERLU DILENGKAPI: bagian, tanggal, sumber bukti, dan peninjau untuk setiap penyunting Peran 4].

## 2. Pendahuluan dan tujuan

Proyek ini merupakan tugas kelompok untuk membangun model generatif dari layer dasar melalui metode vibe coding. Permasalahan yang dipilih adalah penerjemahan gambar kuda menjadi zebra dan sebaliknya tanpa pasangan gambar satu-ke-satu.

Tujuan teknisnya adalah merancang empat model CycleGAN, menyediakan pipeline data yang konsisten, menguji integrasi, dan kemudian melatih serta mengevaluasi penerjemahan gambar. Ruang lingkup yang telah diimplementasikan mencakup fondasi arsitektur, konfigurasi bersama, pipeline, kontrak integrasi dan dokumentasi. Keberhasilan penerjemahan belum dapat disimpulkan sebelum pelatihan dan evaluasi.

## 3. Metode pengembangan dengan LLM

Panduan tugas mengharuskan kolaborasi dengan LLM, pencatatan prompt/respons dan debugging, serta melarang penyalinan implementasi lengkap dari tutorial/GitHub. Dokumentasi resmi diperbolehkan sebagai referensi fungsi dasar. Identitas dan hash panduan tercatat dalam [inventaris](bukti/lingkungan.json).

Pengembangan menggunakan **GPT/ChatGPT** menurut klarifikasi tim. Percakapan sumber memperlihatkan perumusan kebutuhan, ringkasan implementasi/pengujian dan penyuntingan; kode serta hasil tes lokal menjadi dukungan teknis terpisah. Nama atau versi model GPT tidak ditentukan dari label produk ChatGPT. Referensi API dasar dan pengungkapan metode tetap dipertahankan; uraian ini bukan audit orisinalitas menyeluruh.

[Log interaksi](log_interaksi.md), [prompt tugas awal](bukti/prompt_pengguna.md), [prompt tahap pipeline](bukti/prompt_orang2.md), dan [diskusi GPT/ChatGPT](diskusi_llm.md) menjadi rujukan proses dengan batas bukti yang dinyatakan. [Percakapan “Bangun fondasi codebase CycleGAN”](https://chatgpt.com/s/cx_6ae56f2a66c4819196c604c1f9cd4f05) yang dicatat pada 31 Oktober 2026 berhasil dibaca ulang melalui browser pada 3 November 2026: halaman ChatGPT menampilkan tiga rangkaian tentang fondasi, penggunaan dataset final dan penyuntingan akademik. Detail tools serta sesi implementasi pipeline/training tidak seluruhnya tercakup. [PERLU DILENGKAPI: ekspor GPT/ChatGPT sesi lain dan penjelasan anggota].

Koreksi atribusi 3 November 2026: permintaan Claude dalam prompt dan berkas lokal berlabel Claude/Opus bukan bukti eksekusi yang tervalidasi. Klaim panggilan CLI, review independen serta putusan reviewer dari arsip tersebut ditarik dari narasi hasil dan daftar bukti valid. Teks/metadata aslinya tetap tersimpan dan tidak diatribusikan ulang kepada GPT; status setiap arsip dijelaskan dalam register pada diskusi LLM. Hasil tes, angka dataset, konfigurasi dan timestamp asli tidak diubah. Layanan penelusuran tetap memberi `Cache miss`, tetapi browser berhasil membaca sumber; riwayat kegagalan akses sebelumnya tidak dihapus.

## 4. Dataset dan provenance

Dataset aktif ditempatkan pada `data/horse2zebra` dan dibagikan terpisah dari Git. Paket final memuat trainA/trainB/testA/testB, manifest, metadata terkoreksi dan preparation report. Lokasi distribusi paket final: [PERLU DILENGKAPI: lokasi atau tautan distribusi yang ditetapkan tim]. Inventaris juga menemukan `data.zip` lokal berisi 2.608 PNG final dan manifest yang identik dengan folder. Arsip itu tidak dianggap sebagai artefak training, pengganti ZIP preprocessing lama, atau paket distribusi yang sudah disetujui; integritas seluruh isinya tidak diuji ulang.

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

`preprocessing/data.py` membaca folder final berdasarkan manifest. Augmentasi train berlangsung saat loading: resize bicubic 286, crop acak 256, flip horizontal 0,5, kemudian normalisasi `2*(pixel/255)-1`. Transform domain independen. Panjang epoch=max(trainA,trainB)=1.311; domain besar dikunjungi sekali dan domain kecil disampling acak dengan pengembalian. Test tidak memakai transformasi acak. Tidak ada gambar augmentasi offline baru.

`preprocessing/check_data.py` memeriksa dataset final, mengiterasi loader dan menjalankan forward batch nyata pada empat model. `preprocessing/prepare_data.py` dan `preprocessing/audit_archive.py` dipertahankan sebagai utilitas ZIP opsional. Penataan 3 November 2026 memisahkan kode tersebut ke `preprocessing/` dari model/config di `cyclegan/`, tanpa mengubah implementasi, dataset hasil di `data/horse2zebra`, atau API publik loader. Tanggal hasil lama tetap berlaku. Panduan penggunaan dan pemetaan path terdapat dalam [README](../README.md); status pipeline dirangkum dalam [dokumen Peran 2](status_orang2.md).

Pada kode yang diperiksa belum ditemukan implementasi loss CycleGAN, optimizer, training loop atau checkpoint pelatihan. Pencarian berkas, folder dan daftar isi ZIP lokal tidak menemukan log epoch/run, grafik metrik, checkpoint, notebook training atau contoh keluaran. Dua laporan pemeriksaan data menyatakan `training_performed=false`; ini berlaku untuk pemeriksaan tersebut, bukan bukti bahwa tim tidak pernah melatih model di tempat lain. [PERLU DILENGKAPI: konfirmasi Peran 3 dan artefak pekerjaan di luar workspace jika ada].

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

Catatan pengujian struktur 3 November 2026: setelah implementasi data/utilitas dipindahkan ke `preprocessing/`, seluruh 21 tes dan pemeriksaan seluruh dataset/batch nyata kembali berhasil. Identitas ekspor publik serta kedua urutan import diperiksa, dan hasil numerik sama dengan bukti lama. Keluaran tersimpan di `tmp/preprocessing_2026-11-03/`; dataset, arsitektur dan bukti historis tetap sama menurut hash. Ini verifikasi refactoring, bukan eksperimen pelatihan.

## 8. Konfigurasi pelatihan dan rancangan eksperimen

**Status bukti:** belum ditemukan implementasi, konfigurasi run atau hasil pelatihan pada berkas yang diperiksa 31 Oktober 2026. [PERLU DILENGKAPI: konfirmasi Peran 3 mengenai pekerjaan di luar workspace]. Bagian ini membedakan konfigurasi kode yang tersedia dari konfigurasi pelatihan yang benar-benar digunakan.

| Komponen | Bukti yang tersedia / kekurangan |
|---|---|
| Model | Default kode: 64 filter generator/discriminator, sembilan residual block, InstanceNorm non-affine, init_std=0,02; [PERLU DILENGKAPI: konfigurasi arsitektur aktual setiap run] |
| Data | Folder final dan manifest pada bagian 4; augmentasi/normalisasi pada bagian 6; [PERLU DILENGKAPI: manifest/hash dan konfigurasi loader aktual setiap run] |
| Loss | Discriminator menghasilkan skor mentah; [PERLU DILENGKAPI: formulasi adversarial, cycle/identity, bobot serta reduction yang dipakai] |
| Optimasi | [PERLU DILENGKAPI: optimizer, learning rate, parameter optimizer, scheduler, strategi pembaruan G/D] |
| Durasi dan batch | [PERLU DILENGKAPI: epoch/step, batch size, seed, konfigurasi RNG, waktu dan penggunaan memori yang diukur] |
| Validasi/evaluasi | [PERLU DILENGKAPI: split validasi dari train, aturan pemilihan checkpoint, metrik/protokol dan penggunaan test akhir] |
| Reproduksi run | [PERLU DILENGKAPI: ID run yang benar-benar dibuat, versi kode/paket, perangkat, log, checkpoint dan hasil uji resume] |

Rancangan berikut adalah **usulan yang belum disepakati atau dijalankan**, bukan riwayat eksperimen:

1. Tetapkan validasi dari train sebelum tuning, catat kebijakan split dan manifest/hash, lalu pastikan test tidak digunakan untuk memilih hyperparameter.
2. Verifikasi satu langkah pembaruan G/D dan jalur cycle dengan konfigurasi yang dilengkapi Peran 3; rekam finite loss/gradient, penggunaan memori dan kondisi berhenti jika gagal.
3. Jalankan baseline yang terdokumentasi. Setiap tuning membandingkan perubahan yang dinyatakan eksplisit dengan protokol validasi yang tetap, disertai alasan dan log asli.
4. Uji checkpoint/resume serta contoh penerjemahan dua arah dan rekonstruksi cycle pada data evaluasi yang dapat ditelusuri.

**Hasil training:** [PERLU DILENGKAPI: log run, grafik loss, metrik/protokol, konfigurasi pembanding, checkpoint, gambar keluaran dan analisis kegagalan dari Peran 3]. Tidak ada angka hasil eksperimen yang diisi dari tensor dummy, hasil unit test atau rancangan di atas. Jika metrik kelak memakai jaringan pretrained, bedakan jaringan metrik dari model CycleGAN dan cocokkan penggunaannya dengan aturan tugas.

## 9. Studi debugging

[Register debugging](debugging.md) memisahkan gejala, hipotesis, bukti sebab, usulan, tindakan aktual, pengujian ulang dan status setiap kejadian. Draf prompt debugging yang disusun pada sesi kompilasi diberi label agar tidak disalahartikan sebagai percakapan historis.

| Kelompok kejadian | Bukti dan pelajaran yang dapat ditulis |
|---|---|
| DBG-01/02/03 — impor pada tahap test-first | Traceback lengkap tersimpan; tes ditulis sebelum modul tersedia menurut kronologi. Output lanjutan menunjukkan import/suite berhasil. Ini penyelesaian tahap implementasi, bukan eror training |
| DBG-04 — variasi augmentasi | Log melaporkan benturan fixture dengan deduplikasi; tes kini tercatat lulus. Fixture gagal/traceback belum tersedia, sehingga sebab historis belum dibuktikan ulang secara independen |
| DBG-05 — helper traversal | Log melaporkan unpacking path sebelum validasi produksi. Helper kini memakai split pertama dan tes penolakan traversal tercatat lulus; snapshot gagal belum tersedia |
| DBG-06/07 — lingkungan dan PDF | Kegagalan tool terminal hanya tercatat ringkas; akar penyebab belum terbukti. PDF tetap memberi warning struktur, tetapi lima halaman teks dapat dibaca. Keduanya tidak membuktikan bug model |
| DBG-08/09 — proses dokumentasi dan akses chat | Konteks patch dipulihkan menurut log. Pengambilan ulang share link pada kompilasi gagal Cache miss; catatan akses terdahulu tidak ditimpa dan status tautan tidak disimpulkan mati |

Traceback tersedia pada [kegagalan awal](bukti/kegagalan_awal.txt) dan [tes data awal](bukti/tes_data_awal.txt). Hasil uji ulang historis tercatat dalam [pemeriksaan fondasi](bukti/pemeriksaan.txt) dan [suite 21 tes](bukti/tes_dataset_final_tanpa_zip.txt). [PERLU DILENGKAPI: traceback lengkap yang hanya diringkas, fixture/versi saat gagal, prompt asli, nama penangan dan analisis kelompok].

Tidak ada bug perilaku model/training baru yang terbukti pada sesi kompilasi. Peran 4 meneruskan kebutuhan diagnosis ke penanggung jawab modul; tidak mengubah kode agar sesuai narasi. Warning PDF, keterbatasan akses chat dan kelengkapan provenance debugging tetap menjadi tindak lanjut terbuka.

## 10. Pembahasan dan keterbatasan

Fondasi model dan pipeline dapat diintegrasikan melalui kontrak v1.1 dan memiliki bukti pengujian CPU. Masalah grayscale, metadata salah dan duplikasi piksel identik yang diketahui telah ditangani pada tahap persiapan.

Evaluasi kelak berlaku untuk subset test yang telah disaring, bukan benchmark Kaggle penuh. Hash piksel belum mendeteksi semua kemiripan atau crop berbeda. Kapasitas training belum diukur, konfigurasi belum dituning, dan kualitas hasil belum dapat dinilai. Arsip preprocessing yang dihapus membatasi reproduksi penuh tahap sumber meskipun folder final tetap dapat diperiksa.

Pembahasan eksperimen: [PERLU DILENGKAPI: konfigurasi, bukti dan hasil aktual dari Peran 3]. Refleksi pribadi, penguasaan materi, catatan kontribusi manusia terperinci, dan keputusan bersama: [PERLU DILENGKAPI: bahan asli dari anggota kelompok]. Keberadaan repository Git tidak dengan sendirinya membuktikan atribusi kontribusi individual; tidak ada commit baru dibuat sebagai bagian penyuntingan ini.

## 11. Kesimpulan sementara dan kelengkapan laporan

Rancangan dan implementasi fondasi CycleGAN serta pipeline data telah tersedia dan didukung bukti pengujian teknis. Keberhasilan penerjemahan gambar belum dapat disimpulkan karena bukti pelatihan dan evaluasi kualitas belum tersedia pada workspace yang diperiksa. Aktivitas di luar cakupan tersebut perlu dikonfirmasi tim.

Lampiran valid tersedia: prompt, log interaksi dengan koreksi atribusi, sumber dan ringkasan GPT/ChatGPT, kode, tes, audit, metadata sumber, manifest/preparation report dan hasil pemeriksaan. Arsip atribusi yang belum terverifikasi disimpan terpisah secara status dan bukan dasar kesimpulan. Informasi yang belum tersedia: lokasi distribusi dataset final, kelengkapan tautan atau ekspor sesi lainnya, bukti kontribusi manusia terperinci, refleksi kelompok dan seluruh hasil eksperimen Peran 3.

## 12. Kelengkapan menurut rubrik

Rubrik dibaca dari halaman 5 panduan yang tersedia. Tabel berikut menilai **kelengkapan bukti**, bukan memberikan skor kelompok.

| Kriteria | Bobot | Bukti tersedia | Kekurangan / status |
|---|---:|---|---|
| Kualitas dan orisinalitas vibe coding | 30% | Prompt, alasan desain, implementasi modular dan percakapan GPT/ChatGPT bersumber | Sebagian; [PERLU DILENGKAPI: cakupan percakapan/variasi prompt dan penjelasan anggota]. Tidak ada audit orisinalitas menyeluruh |
| Log interaksi dan dokumentasi LLM | 25% | Kronologi, ID katalog sesi, prompt, satu share link terdaftar, register debugging | Sebagian; [PERLU DILENGKAPI: ekspor sesi yang kurang, identitas peserta, analisis kelompok, traceback yang belum lengkap] |
| Fungsionalitas model dan pelatihan | 25% | Bukti 21 tes, dataset/manifest dan batch nyata CPU | Fondasi/pipeline didukung; [PERLU DILENGKAPI: implementasi/run training, konfigurasi, grafik/metrik valid, checkpoint/resume dan contoh keluaran] |
| Kerja sama dan manajemen integrasi | 20% | Kontrak, pemisahan penugasan/kontribusi/penyuntingan dan usulan paket dokumentasi | Sebagian; [PERLU DILENGKAPI: pemetaan nama/NIM, log kerja/suntingan, peninjauan anggota dan bukti penguasaan individu] |

- [x] Dataset dan konfigurasi kode merujuk bukti yang tersedia.
- [x] Hasil pengujian dipisahkan dari hasil eksperimen training.
- [x] Riwayat, prompt, traceback, respons asli dan bukti historis dipertahankan.
- [ ] Identitas, pembagian kerja dan kontribusi manusia lengkap.
- [ ] Arsip percakapan dan bukti debugging seluruh tahap lengkap.
- [ ] Hasil training dan evaluasi yang dapat ditelusuri tersedia.
- [ ] Refleksi, peninjauan akhir serta kelengkapan referensi diserahkan anggota.

Pembagian pengumpulan dan daftar bahan per anggota/lingkup terdapat dalam [koordinasi dokumentasi](koordinasi_dokumentasi.md). Laporan tetap **draf**, karena bukti wajib belum lengkap.

## 13. Referensi dan lampiran

1. *[CAK4KBB3 2026-1] Panduan Proyek 1 — Vibe Coding*, dokumen tugas lokal; identitas/hash pada [inventaris](bukti/lingkungan.json).
2. [Horse2zebra Dataset, Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), metadata versi 1 tersimpan pada [arsip sumber](bukti/kaggle_metadata.json). Metadata mengacu pada Zhu, Park, Isola, Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
3. Referensi fungsi dasar: [PyTorch](https://docs.pytorch.org/docs/stable/index.html) dan [Pillow](https://pillow.readthedocs.io/en/stable/). Penelusuran API melalui Context7 dicatat dalam log.
4. [Kontrak integrasi](kontrak_integrasi.md), [status pipeline](status_orang2.md), [log interaksi](log_interaksi.md), [diskusi GPT/ChatGPT](diskusi_llm.md), [register debugging](debugging.md), dan [koordinasi dokumentasi](koordinasi_dokumentasi.md).

Format sitasi dan kelengkapan daftar pustaka sesuai ketentuan mata kuliah: [PERLU DILENGKAPI: bahan asli dari anggota kelompok].
