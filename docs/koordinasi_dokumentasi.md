# Koordinasi Dokumentasi Kelompok — Peran 4

**Pembaruan integrasi 3 November 2026:** skrip training/evaluasi kini tersedia dari commit remote anggota lain `5fe225b`; lihat [panduan training](training_lokal.md). Inventaris 31 Oktober di bawah tetap mencatat keadaan lokal sebelum fetch tersebut. DOK-03 kini perlu mengumpulkan bukti eksekusi dan hasil dari skrip yang tersedia. Identitas penanggung jawab Peran 1–3 dilengkapi dari laporan Word pada 7 November 2026; rincian penyuntingan setiap anggota tetap perlu dicatat.

Status: **draf kompilasi berbasis bukti, 31 Oktober 2026; bukan laporan final**.
Nama kelompok: [PERLU DILENGKAPI: nama atau nomor kelompok].
Penyusun dokumentasi: beberapa anggota tim; [PERLU DILENGKAPI: nama, NIM, paket kerja dan hasil suntingan setiap anggota].

Catatan struktur 3 November 2026: path kode aktif di bawah mengikuti pemindahan ke `preprocessing/`; tanggal inventaris, hasil pengujian dan bukti 27/31 Oktober tetap historis. `data/horse2zebra` adalah dataset hasil, bukan folder kode. API publik loader tetap diekspor dari `cyclegan`; pemetaan path lama tersedia pada README.

## 1. Penugasan, pelaksanaan dan penyuntingan

Penanggung jawab Peran 1 (prompt, arsitektur, dan integrasi) adalah Jian Hazel Sitorus (103012300187); Peran 2 (pipeline data) adalah Mikhael Restu Mahardhika (103012300277); dan Peran 3 (pelatihan dan tuning) adalah Reevanza Abel Desta Arifin (103012330104). Jian Hazel Sitorus membantu prapemrosesan awal untuk sementara, kemudian Mikhael Restu Mahardhika melanjutkan tanggung jawab Peran 2. Dokumentasi awal Peran 4 ditangani Jian Hazel Sitorus dan kelanjutannya dikerjakan bersama anggota tim. Penanggung jawab modul dan penyunting dokumentasi dicatat sesuai pekerjaan aktual.

| Jenis atribusi | Yang diketahui | Kekurangan bukti |
|---|---|---|
| Penanggung jawab awal | Peran 1: Jian Hazel Sitorus (103012300187); Peran 2: Mikhael Restu Mahardhika (103012300277); Peran 3: Reevanza Abel Desta Arifin (103012330104) | Rincian pembagian dokumentasi Peran 4 per anggota |
| Pelaksana teknis aktual | Jian Hazel Sitorus menangani fondasi dan membantu prapemrosesan awal untuk sementara; kelanjutan prapemrosesan menjadi tanggung jawab Mikhael Restu Mahardhika; pelatihan menjadi tanggung jawab Reevanza Abel Desta Arifin | [PERLU DILENGKAPI: rincian pekerjaan, tanggal, dan log pendukung setiap tahap] |
| Penyunting dokumentasi | Peran 4 dikerjakan bersama; kompilasi sesi ini dibantu LLM | [PERLU DILENGKAPI: identitas penyunting, berkas/bagian, tanggal dan hasil pemeriksaan setiap anggota] |
| Pelaksanaan training | Reevanza Abel Desta Arifin bertanggung jawab pada Peran 3; inventaris historis di bawah mencatat keadaan berkas saat diperiksa | [PERLU DILENGKAPI: bukti run, konfigurasi, metrik, checkpoint, dan keluaran eksperimen] |

Tidak ada nama anggota, persetujuan pembagian baru, commit kontribusi, atau refleksi pribadi yang ditambahkan tanpa sumber. Laporan kerja dapat berupa catatan anggota dan bukti perubahan yang nyata; tidak perlu membuat commit bertanggal surut.

## 2. Pembagian kerja dokumentasi yang diusulkan

Tabel ini merupakan **usulan paket kerja**, bukan penetapan anggota yang telah disepakati. Satu anggota boleh menangani beberapa paket; setiap hasil perlu ditinjau anggota lain yang identitasnya dicatat.

| Paket | Pengumpulan dan keluaran | Pengumpul / peninjau | Status |
|---|---|---|---|
| DOK-01 | Percakapan, prompt asli, alasan rancangan, keputusan arsitektur; lengkapi LOG-01/LOG-04 dan ekspor chat | [PERLU DILENGKAPI: nama pengumpul dan peninjau] | Prompt dan log fondasi ada; cakupan ekspor belum lengkap |
| DOK-02 | Manifest/hash, metadata, audit, transformasi, sampling, dan bukti pipeline; lengkapi LOG-02/LOG-03 | [PERLU DILENGKAPI: nama pengumpul dan peninjau] | Bukti pipeline tersedia; percakapan lengkap dan atribusi masih kurang |
| DOK-03 | Konfigurasi tiap run, log training, metrik/grafik, checkpoint, contoh keluaran, debugging training | [PERLU DILENGKAPI: nama pengumpul dari Peran 3 dan peninjau] | Artefak belum ditemukan; bukan bukti bahwa pekerjaan di luar workspace tidak pernah dilakukan |
| DOK-04 | Penggabungan laporan, studi debugging, daftar pustaka, pemeriksaan tautan/status/rubrik | [PERLU DILENGKAPI: nama penggabung dan peninjau] | Draf kompilasi tersedia; pengesahan tim dan bukti wajib belum lengkap |

Setiap anggota menyerahkan nama/NIM, paket yang dikerjakan, tanggal sesi yang sebenarnya, lokasi bukti, perubahan dokumen, serta bagian yang belum dapat dikonfirmasi. Nama penyunting dicatat terpisah dari pembuat kode. Analisis pribadi atau keputusan bersama hanya dimasukkan jika teksnya benar-benar diserahkan tim.

## 3. Inventaris bukti dan batas verifikasi

Inventaris 31 Oktober 2026 membaca kode, seluruh daftar berkas workspace di luar metadata Git/cache, isi dokumentasi/bukti, manifest dataset, serta daftar anggota ZIP lokal. Tidak menjalankan training atau mengulang suite model.

| Bukti | Lokasi | Dukungan dan batas |
|---|---|---|
| Panduan tugas | Identitas/hash pada [lingkungan](bukti/lingkungan.json) | PDF lokal tersedia dan lima halaman dibaca; rubrik halaman 5: 30/25/25/20 |
| Prompt dan percakapan | [Prompt fondasi](bukti/prompt_pengguna.md), [prompt pipeline](bukti/prompt_orang2.md), [log](log_interaksi.md) | Prompt bukan ekspor seluruh respons/tools |
| Tautan percakapan | Tautan ChatGPT pada bagian 10 log | Akses browser 3 November berhasil, sementara layanan penelusuran memberi `Cache miss`; catatan kegagalan 31 Oktober tetap historis |
| Percakapan GPT/ChatGPT | [Ringkasan dan sumber](diskusi_llm.md) | Platform dan tiga rangkaian percakapan terlihat pada browser 3 November 2026; tidak menetapkan versi model, eksekusi alat lain atau review independen |
| Fondasi model | `cyclegan/config.py`, `cyclegan/models.py`, [tes fondasi](bukti/pemeriksaan.txt) | Generator residual/PatchGAN dan 10 tes pada tahap awal; bukan hasil training |
| Pipeline/data | `preprocessing/data.py`, [manifest](bukti/manifest.json), [preparation report](bukti/preparation_report.json) | Final 1.040/1.311/118/139, total 2.608; metadata terkoreksi memuat 2.661 sumber |
| Tes dan integrasi | [21 tes](bukti/tes_dataset_final_tanpa_zip.txt), [data/model](bukti/pemeriksaan_dataset_final_tanpa_zip.json) | Pemeriksaan 27 Oktober 2026, CPU; `training_performed=false` |
| Debugging | [Register debugging](debugging.md), [traceback fondasi](bukti/kegagalan_awal.txt), [traceback pipeline](bukti/tes_data_awal.txt) | Tiga kegagalan impor mempunyai traceback tersimpan; beberapa kejadian lain hanya memiliki ringkasan |
| Dataset aktif | `data/horse2zebra` | Semua path manifest ada; jumlah dan hash manifest cocok dengan bukti lama. Decode/hash piksel seluruh gambar tidak diulang pada sesi dokumentasi ini |
| Arsip lokal tambahan | `data.zip` | Ada 2.608 PNG final dan manifest yang cocok byte dengan folder, serta ZIP sumber Kaggle di dalamnya. Bukan checkpoint atau pengganti `preproc.zip`; integritas seluruh isi arsip tidak diuji ulang |
| Training | Tidak ditemukan pada kode, folder atau daftar isi ZIP yang diperiksa | Tidak ada loss/optimizer/training loop, log epoch, grafik metrik, checkpoint atau gambar keluaran yang dapat dikompilasi sebagai hasil training |

`data/` dan ZIP dikecualikan dari Git. [PERLU DILENGKAPI: lokasi distribusi dataset final yang ditetapkan tim]. Keberadaan `data.zip` lokal tidak dianggap sebagai tautan distribusi atau paket yang sudah disetujui tim. `preproc.zip` tetap tidak tersedia; hash dan audit historis tidak diubah.

Koreksi atribusi 3 November 2026: arsip `diskusi_claude.md`, tiga JSON berlabel model/session, klaim reviewer pada rencana/log dan permintaan alat di prompt tidak digunakan untuk membuktikan pelaksanaan Claude atau review independen. Seluruhnya tetap tersedia dengan alasan preservasi dan status pada register di diskusi LLM. Daftar bukti valid di atas tidak memasukkan putusan atau temuan reviewer tersebut. Tidak ada respons lama yang diganti atribusinya menjadi GPT.

## 4. Kebutuhan dari anggota berdasarkan lingkup kerja

| Sumber tim | Bahan yang perlu diserahkan |
|---|---|
| Jian Hazel Sitorus (103012300187) — arsitektur dan integrasi | [PERLU DILENGKAPI: ekspor lengkap sesi rancangan, keputusan yang disetujui anggota, dan catatan alasan/penjelasan pribadi] |
| Mikhael Restu Mahardhika (103012300277) — penanggung jawab pipeline; Jian Hazel Sitorus (103012300187) — bantuan sementara pada tahap awal | [PERLU DILENGKAPI: arsip percakapan implementasi, rincian tahap per anggota, traceback fixture gagal, versi fixture sebelum/sesudah, dan distribusi dataset final] |
| Reevanza Abel Desta Arifin (103012330104) — pelatihan dan tuning | [PERLU DILENGKAPI: konfigurasi/run ID, kode, log, grafik, checkpoint, keluaran, dan perbaikan eror terkait] |
| Setiap penyunting Peran 4 | [PERLU DILENGKAPI: nama/NIM, bagian dokumen yang disunting, sumber bukti, hasil pemeriksaan dan peninjau] |
| Koordinator kelompok | [PERLU DILENGKAPI: identitas akademik, daftar anggota, pemetaan penugasan awal/pelaksanaan/penyuntingan dan catatan pembagian yang benar-benar disepakati] |

Untuk eksperimen, sertakan konfigurasi arsitektur dan manifest/hash aktual, loss dan bobotnya, optimizer/scheduler, seed, device, epoch/step, protokol validasi/test, sumber metrik, checkpoint dan contoh keluaran yang dapat ditelusuri. Jika belum dikerjakan, catat status tersebut; jangan mengisi angka rancangan sebagai hasil run.

## 5. Checklist rubrik dan kelengkapan

Ini pemeriksaan kelengkapan, **bukan pemberian nilai atau prediksi skor**.

| Rubrik | Bobot | Bukti tersedia | Yang masih diperlukan | Status |
|---|---:|---|---|---|
| Kualitas dan orisinalitas vibe coding | 30% | Prompt, alasan arsitektur, kode modular, pengungkapan GPT/ChatGPT dan sumber percakapan | [PERLU DILENGKAPI: cakupan percakapan seluruh tahap dan penjelasan anggota tentang keputusan/variasi prompt] | Sebagian; bukan audit orisinalitas menyeluruh |
| Log interaksi dan dokumentasi LLM | 25% | Kronologi, indeks sesi, prompt lokal, satu tautan terdaftar, register debugging | [PERLU DILENGKAPI: ekspor asli yang belum ada, identitas peserta, analisis kelompok dan traceback yang hanya diringkas] | Sebagian |
| Fungsionalitas model dan pelatihan | 25% | Bukti tes model/pipeline, hash dataset dan forward batch nyata | [PERLU DILENGKAPI: kode/run training, konfigurasi, grafik/metrik valid, checkpoint/resume, contoh keluaran] | Fondasi terbukti; bukti pelatihan belum tersedia |
| Kerja sama dan manajemen integrasi | 20% | Kontrak v1.1, pemisahan peran, klarifikasi kontribusi dan usulan paket dokumentasi | [PERLU DILENGKAPI: nama/NIM, pembagian/hasil suntingan aktual, log kerja atau bukti perubahan dan penguasaan individu] | Sebagian |

- [x] Dasar teknis dan angka dataset mengacu bukti yang dapat ditelusuri.
- [x] Hasil pengujian dipisahkan dari hasil pelatihan.
- [x] Bukti asli dan riwayat lama dipertahankan.
- [ ] Identitas dan kontribusi anggota lengkap.
- [ ] Arsip percakapan seluruh tahap dan bukti debugging lengkap.
- [ ] Eksperimen, metrik/grafik, checkpoint dan keluaran tersedia.
- [ ] Analisis/refleksi kelompok serta pemeriksaan akhir oleh anggota tercatat.

Laporan tetap berstatus draf sampai kekurangan bukti wajib tersebut dilengkapi. Tidak ada perubahan perilaku model/pipeline/training yang dilakukan untuk memenuhi narasi laporan.
