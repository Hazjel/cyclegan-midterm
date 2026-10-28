# Laporan proyek: Vibe Coding Generative AI from Scratch

**Status dokumen:** draf berbasis bukti fondasi model dan pipeline orang 2, sesi 27 Oktober 2026. Proyek keseluruhan belum selesai; training dan evaluasi kualitas model belum dilakukan.

## 1. Identitas

Nama kelompok: [BELUM DIBERIKAN]. Kelas: [BELUM DIBERIKAN]. Dosen: [BELUM DIBERIKAN].
Nama/NIM dan pemetaan anggota: [BELUM DIBERIKAN].

Pembagian yang diminta pengguna: Peran 1 dan 4 ditangani pengguna, orang 2 menangani data, orang 3 menangani training. Pembagian tersebut tidak menjadi bukti kontribusi selesai dari anggota lain.

## 2. Tujuan dan ruang lingkup

Proyek bertujuan menerjemahkan gambar kuda dan zebra dua arah menggunakan CycleGAN tanpa dataset berpasangan. Implementasi mencakup arsitektur empat model, konfigurasi bersama, kontrak tensor, pipeline data dan dokumentasi. Pemeriksaan menggunakan tensor dummy serta batch data asli. PyTorch digunakan untuk layer dasar serta autograd, tanpa pretrained weights atau arsitektur siap pakai.

Panduan lokal halaman 1 melarang penyalinan implementasi lengkap dari tutorial/GitHub dan membolehkan dokumentasi resmi fungsi dasar. Halaman 2 memuat pembagian peran, halaman 3 mencantumkan CycleGAN, halaman 5 meminta log dan bukti fungsionalitas/pelatihan. Dokumen sumber dan hash-nya tercatat di [lingkungan](bukti/lingkungan.json).

## 3. Metode Vibe Coding dan orisinalitas

Pengguna memberikan prompt tugas terperinci yang menetapkan batas kontribusi, tahapan audit, arsitektur, pemeriksaan dan kejujuran bukti. [Prompt tersedia](bukti/prompt_pengguna.md) dan kronologi dirangkum dalam [log sesi](log_interaksi.md).

AI membaca berkas, memeriksa API dasar melalui Context7, menulis komposisi layer sendiri, menjalankan pemeriksaan lalu menyusun handoff. Tidak mengambil implementasi CycleGAN lengkap dari tutorial atau GitHub. Ini uraian proses sesi, bukan audit plagiarisme menyeluruh terhadap seluruh sumber eksternal. Penguasaan materi/refleksi anggota: [BELUM DIBERIKAN KELOMPOK]. Ekspor chat lengkap/share link: [BELUM DILAMPIRKAN].

## 4. Dataset dan hasil audit

**Pembaruan penggunaan, 27 Oktober 2026:** pengguna menyatakan telah menghapus `preproc.zip`; pemeriksaan workspace mengonfirmasi arsip tidak tersedia. Root runtime aktif `data/horse2zebra` langsung dibaca oleh pipeline dan pemeriksaan data. Deskripsi ZIP, hash dan audit berikut tetap merupakan provenance sesi sebelumnya. Dataset, manifest, transformasi dan split final dipertahankan. Bukti verifikasi sesudah penghapusan disimpan terpisah pada [pemeriksaan dataset final tanpa ZIP](bukti/pemeriksaan_dataset_final_tanpa_zip.json). Tanpa arsip preprocessing asli, pembandingan ulang dan persiapan dari pasangan ZIP tidak dapat direproduksi sepenuhnya; hal ini tidak menghalangi loading atau training orang 3 nantinya. Training belum dijalankan pada sesi penyesuaian ini.

Sumber arsip lokal: `preproc.zip`, SHA-256 `1dd5f0529247324f986cbceeaf260bb6712ef24013d9527dba57b85d0def3805`. Pada sesi orang 2, [dataset Kaggle Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset) versi 1 diunduh dan dibandingkan byte demi byte terhadap seluruh 2.661 gambar asli. ZIP Kaggle SHA-256 `0655d8feb8eb83911ac08c1d230a0464794418c107412a28adc47e1eb69e90a0`; CLI mem-pin hash ini. Metadata sumber ada pada [kaggle_metadata.json](bukti/kaggle_metadata.json). Lisensi API: `Other (specified in description)`; deskripsi mengakui sumber UC Berkeley dan meminta sitasi CycleGAN (Zhu et al., ICCV 2017).

Arsip berisi data asli 2.661 gambar, cleaned 2.633, review 28, notebook dan dua CSV. Cleaned terdiri dari trainA=1.049, trainB=1.327, testA=118, testB=139. Semua cleaned dapat didekode, 256×256, dengan 2.626 RGB dan tujuh grayscale trainB. Seluruh cleaned adalah salinan byte identik dari gambar aslinya; dua folder itu tidak digabung sebagai sampel berbeda.

Metadata awal 2.661 baris memiliki 240 path yang salah lokasi: 120 testB→testA dan 120 trainA→trainB. Kesalahan yang sama ditemukan pada CSV Kaggle. Pipeline memperbaiki path, domain dan split pada salinan metadata, dengan pencocokan satu-ke-satu dan cek silang image_id/prefix nama. Sebanyak 25 salinan duplikat dikecualikan menggunakan hash piksel RGB, termasuk trainA/n02381460_3811.jpg yang identik dengan testA/n02381460_1160.jpg; test diprioritaskan.

Notebook hanya mengaudit blur dan menyalin gambar berdasarkan threshold 100. Tampilan RGB pada plot tidak mengonversi berkas dataset. CSV blur cocok dengan partisi ZIP. Audit tidak menilai kelayakan semantik seluruh gambar dan tidak mendeteksi seluruh duplikasi visual dengan encoding berbeda.

Bukti awal: [audit JSON](bukti/audit_dataset.json). Pipeline orang 2 kini menyimpan 2.608 PNG RGB 256 x 256 pada data/horse2zebra: trainA=1.040, trainB=1.311, testA=118, testB=139. Ketujuh grayscale dipertahankan dengan tiga kanal identik. Seluruh 28 gambar skor blur <100 tetap dikecualikan; penghitungan ulang seluruh skor menggunakan OpenCV terpasang menghasilkan nol mismatch. Metadata terkoreksi, manifest dan audit pengecualian tersimpan pada [status orang 2](status_orang2.md).

Train memakai bicubic resize 286, random crop 256, flip horizontal 0,5 dan normalisasi RGB float32 CHW [-1,1]. Test deterministik dan dimuat terpisah per domain. Domain train lebih besar dikunjungi sekali per epoch, domain kecil disampling acak dengan pengembalian. Test yang digunakan merupakan subset cleaned, bukan keseluruhan benchmark Kaggle; validasi untuk tuning harus diambil orang 3 dari train.

## 5. Perancangan arsitektur oleh Peran 1

A=kuda, B=zebra, G_AB=kuda→zebra, G_BA=zebra→kuda, D_A=domain kuda, D_B=domain zebra. Satu kelas Generator dan satu kelas PatchDiscriminator dipakai untuk membuat dua instance masing-masing dengan parameter terpisah.

Generator: conv 7×7 dengan reflection padding; dua downsampling 3×3 stride 2; sembilan residual block (dua conv 3×3, InstanceNorm dan ReLU di cabang residual); dua transposed convolution 3×3 stride 2; conv 7×7 dan Tanh. Kanal 3→64→128→256→128→64→3. Residual addition mempertahankan jalur fitur identitas; Tanh menyelaraskan rentang output dan normalisasi input [-1,1].

Discriminator: lima conv 4×4 padding 1, stride 2/2/2/1/1, kanal 3→64→128→256→512→1; LeakyReLU 0,2 dan InstanceNorm di lapisan tengah. Output 256×256 menjadi peta skor mentah 30×30 tanpa sigmoid, dengan receptive field konvolusional nominal 70×70. InstanceNorm menambahkan ketergantungan statistik spasial, sehingga angka 70 tidak berarti independensi spasial penuh.

InstanceNorm memakai affine=False, track_running_stats=False, eps=1e-5. Tidak bergantung pada statistik batch atau running statistics. Semua convolution diinisialisasi normal(0,0.02), bias nol. Tidak ada dropout atau bobot pretrained. Jumlah parameter per G=11.378.179 dan per D=2.764.737. Konfigurasi immutable dapat diekspor lewat `asdict` dan diberi versi arsitektur untuk restore.

Kapasitas filter/block, normalisasi, skema inisialisasi, dan upsampling adalah baseline rancangan AI; belum dinyatakan optimal. Hardware terdeteksi RTX 3060 Laptop 6 GiB, tetapi belum ada bukti penggunaan memori training. Tidak memberikan estimasi durasi atau klaim kualitas hasil.

## 6. Kontrak dan manajemen integrasi

Batch yang diminta adalah dictionary `A` dan `B`: tensor RGB float32 `[N,3,256,256]`, finite, [-1,1]. Pipeline menyuplai CPU tensor unpaired; training memindahkan model dan batch ke device yang sama. Model menolak dtype, rentang, kanal, dan ukuran yang tidak memenuhi kontrak.

Orang 3 mengimpor `ArchitectureConfig`, `Generator`, `PatchDiscriminator` dari `cyclegan`, membuat empat instance, serta menyimpan konfigurasi aktual dan empat state_dict. Pemilihan adversarial loss harus sesuai skor mentah; BCEWithLogitsLoss tidak diberi sigmoid tambahan. Seluruh rincian berada di [kontrak integrasi](kontrak_integrasi.md).

Handoff tertulis adalah bukti koordinasi yang tersedia. Riwayat Git, diskusi persetujuan anggota dan penguasaan individual: [BELUM DIBERIKAN].

## 7. Pemeriksaan fondasi dan penanganan masalah

Pemeriksaan memakai Python 3.12.3, PyTorch 2.5.1+cu121, Pillow 12.3.0; semua tes model di CPU, seed 7 dan empat thread. Hasil terbaru: **21 tes lulus**, tujuh model, tiga audit dan sebelas data, termasuk worker Windows. [Log terbaru](bukti/tes_orang2.txt) dan [pemeriksaan data asli](bukti/pemeriksaan_data.json) mencakup seluruh berkas, satu epoch tiap loader, dan forward batch nyata pada empat model default. Tidak menjalankan training loop. Pemeriksaan fondasi awal tetap tersimpan pada [log sebelumnya](bukti/pemeriksaan.txt).

| Yang diperiksa | Bukti hasil |
|---|---|
| Import API publik | berhasil, konfigurasi tercetak |
| Dua G pada 256×256 | `[1,3,256,256]`, finite, output di [-1,1] |
| Dua D pada 256×256 | `[1,1,30,30]`, finite |
| Backward keempat model | gradient input dan semua parameter yang diperiksa tersedia dan finite; gradient tidak seluruhnya nol |
| Independensi instance | identitas parameter/storage tidak bertumpang tindih, mutasi model pertama tidak mengubah pasangan |
| Ukuran tambahan dan eval | batch dua gambar 32×40 diterima, shape sesuai |
| Validasi | input dan konfigurasi tidak valid ditolak |
| Restore model | round-trip konfigurasi/state_dict dalam memori mempertahankan output |
| Audit | ZIP fixture tidak berubah, notebook tidak dijalankan, metadata salah/gambar rusak/duplikasi lintas split terlaporkan |

Kegagalan awal test-first berupa modul yang belum dibuat dan peringatan parser PDF dicatat di log; tidak ada eror training yang diklaim. Penolakan input invalid pada tes bukan kegagalan penggunaan nyata. Pemeriksaan ini membuktikan fondasi shape/numerik/API, **bukan kualitas penerjemahan, kestabilan adversarial training, atau kecukupan memori training**.

## 8. Implementasi pelatihan

[PERLU DILENGKAPI ORANG 3]

Isi yang diperlukan: loss adversarial/cycle/identity beserta rumus/bobot, optimizer, learning rate/scheduler, epoch/batch size, seed, strategi pembaruan G/D, replay buffer jika dipakai, konfigurasi arsitektur/data aktual dan perangkat yang digunakan. Jangan mengisi angka rancangan sebagai hasil eksperimen yang telah dilakukan.

## 9. Eksperimen, tuning, metrik dan grafik

[PERLU DILENGKAPI ORANG 3]

Lampirkan konfigurasi setiap run, alasan perubahan, log loss nyata dan grafik, protokol evaluasi, sumber bobot model metrik jika ada, hasil numerik yang terukur serta batas interpretasinya. Jika memakai metrik berbasis jaringan pretrained, bedakan model metrik tersebut dari model CycleGAN yang harus dilatih dari nol dan cocokkan dengan aturan dosen. Belum ada metrik atau grafik hasil dalam sesi ini.

## 10. Hasil penerjemahan dan checkpoint

[PERLU DILENGKAPI ORANG 3]

Tambahkan contoh kuda→zebra dan zebra→kuda dari test yang telah diperbaiki, rekonstruksi cycle, analisis kegagalan pada sampel nyata, path/identitas checkpoint, uji restore/resume serta konfigurasi yang disimpan. Tidak menggunakan output tensor acak sebagai contoh hasil training.

## 11. Pembahasan, keterbatasan dan refleksi

Fondasi dan pipeline terhubung melalui kontrak v1.1 dan tes CPU. Grayscale, kesalahan metadata dan duplikasi identik yang diketahui telah ditangani. Deduplikasi belum menjamin bebas gambar mirip dengan encoding/crop berbeda. Konfigurasi awal belum dituning dan kemampuan training pada GPU terdeteksi belum diukur.

Pembahasan hasil training: [PERLU DILENGKAPI ORANG 3].
Pelajaran, pembagian kontribusi aktual, keputusan bersama dan analisis pribadi: [BELUM DIBERIKAN KELOMPOK].

## 12. Kesimpulan sementara dan lampiran

Fondasi arsitektur, pipeline orang 2 dan dokumentasi telah dibuat serta diperiksa. Review read-only Claude Opus 5.5 High tercatat pada [diskusi Claude](diskusi_claude.md). Kesimpulan tentang keberhasilan CycleGAN menerjemahkan gambar: [PERLU DILENGKAPI ORANG 3].

Lampiran yang tersedia: prompt tugas, log sesi, kontrak, kode model/pipeline, tes, audit, metadata sumber Kaggle, manifest/metadata final dan keluaran pemeriksaan. Lampiran belum tersedia: identitas kelompok, ekspor lengkap chat/share link, bukti kontribusi manusia, log training, grafik, metrik, checkpoint dan hasil gambar.

Referensi API dasar resmi dan panduan lokal tercantum di README. Daftar pustaka ilmiah dan format sitasi yang diwajibkan kelas: [BELUM DILENGKAPI KELOMPOK].
