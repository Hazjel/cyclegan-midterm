# Register Debugging dan Kekuatan Bukti

Dikompilasi 31 Oktober 2026 untuk Peran 4. Nama pelaksana/penyunting manusia: [PERLU DILENGKAPI: nama, NIM dan kasus yang ditangani]. Register ini melengkapi [log kronologis](log_interaksi.md); tidak menggantikan traceback, percakapan atau hasil tes asli.

Catatan atribusi 3 November 2026: proses proyek menggunakan GPT/ChatGPT menurut klarifikasi tim dan [sumber percakapan](diskusi_llm.md). Arsip klaim alat/review lain tidak membuktikan identitas pelaksana atau sebab eror. Pembacaan browser pada sesi koreksi berhasil; DBG-09 tetap mencatat kegagalan layanan pada tanggal kejadian lama.

## Aturan pencatatan

Tanggal kejadian mengikuti sumber; tanggal kompilasi tidak dipakai untuk menanggali ulang kejadian lama. **Terverifikasi dari output** dibedakan dari **dilaporkan dalam log** dan **hipotesis**. Kelulusan suite sesudah perbaikan tidak membuktikan sebab kegagalan lama jika fixture/traceback lama tidak tersedia. Test-first sebelum modul dibuat juga tidak dianggap sebagai bug training.

Semua prompt debugging di bawah adalah **draf yang disusun pada sesi kompilasi ini** untuk ditinjau tim, bukan rekonstruksi prompt yang pernah dikirim. Prompt asli dan analisis pribadi kelompok yang belum ditemukan ditandai. Tidak ada kode diperbaiki atau pengujian model diulang pada sesi dokumentasi ini.

Lingkungan historis yang tercatat: Windows, Python 3.12.3, PyTorch 2.5.1+cu121, Pillow 12.3.0; [inventaris](bukti/lingkungan.json). Tes model memakai CPU, empat thread dan seed 7. Informasi itu tidak otomatis menjadi konfigurasi setiap kegagalan: identitas versi/fixture persis saat gagal tetap perlu dilampirkan bila tidak tercatat.

## DBG-01 — Impor `cyclegan` sebelum paket dibuat

- **Kejadian/sumber:** tahap fondasi, 27 Oktober 2026; LOG-01. [Traceback lengkap tersimpan](bukti/kegagalan_awal.txt), blok perintah pertama.
- **Pemicu/gejala:** `python -m unittest discover -s tests -v`; loader gagal mengimpor `test_models`, berakhir `ModuleNotFoundError: No module named 'cyclegan'`.
- **Hipotesis dan bukti sebab:** modul belum ada atau path impor salah. Traceback membuktikan kegagalan impor; urutan kerja pada log mencatat tes ditulis sebelum paket dibuat. Tidak ada snapshot folder awal terpisah untuk membuktikan seluruh kondisi impor.
- **Usulan/tindakan aktual:** menyediakan paket dan ekspor publik; log mencatat pembuatan `cyclegan/__init__.py`, `config.py`, `models.py`. Ini penyelesaian tahap test-first, bukan perbaikan training.
- **Verifikasi/status:** [pemeriksaan fondasi](bukti/pemeriksaan.txt) memuat import berhasil dan 10 tes lulus. Status: terselesaikan pada tahap fondasi menurut output/log; tidak diulang kini.
- **Draf prompt:** “Pada tahap test-first, perintah unittest gagal mengimpor `cyclegan`; traceback lengkap terlampir. Periksa keberadaan paket dan root eksekusi sebelum mengubah path. Implementasi/ekspor minimal apa yang perlu tersedia, dan perintah apa yang membuktikan impor serta tes berhasil?”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: prompt debugging asli jika terpisah, identitas penangan dan analisis pribadi].

## DBG-02 — Impor utilitas audit sebelum modul dibuat

- **Kejadian/sumber:** tahap fondasi, 27 Oktober 2026; LOG-01. [Traceback lengkap](bukti/kegagalan_awal.txt), blok perintah kedua.
- **Pemicu/gejala:** `python -m unittest discover -s tests -p test_audit_archive.py -v`; `ModuleNotFoundError: No module named 'scripts.audit_archive'`.
- **Hipotesis dan bukti sebab:** utilitas belum ada atau eksekusi dari root salah. Output membuktikan modul tidak dapat diimpor; log mencatat script belum diimplementasikan saat tes dijalankan.
- **Usulan/tindakan aktual:** implementasi audit baca-saja dan CLI pada `scripts/audit_archive.py`; notebook diperlakukan sebagai data. Tes fixture memastikan sel berisi `raise RuntimeError(...)` tidak dieksekusi.
- **Verifikasi/status:** tiga tes audit tercantum lulus dalam [hasil suite fondasi](bukti/pemeriksaan.txt). Status: terselesaikan pada tahap test-first; bukan audit dataset final tanpa ZIP.
- **Draf prompt:** “Tes audit gagal pada impor `scripts.audit_archive` sebelum utilitas ditulis. Dari traceback ini, periksa struktur paket dan kontrak fungsi yang dipanggil tes. Buat audit baca-saja dan pastikan notebook tidak dieksekusi; verifikasi kembali tes audit.”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: prompt asli jika terpisah, identitas dan analisis anggota].

## DBG-03 — Impor `scripts.prepare_data` sebelum implementasi

- **Kejadian/sumber:** tahap pipeline, 27 Oktober 2026; LOG-02. [Keluaran awal dan traceback lengkap](bukti/tes_data_awal.txt).
- **Pemicu/gejala:** `python -m unittest discover -s tests -p test_data.py -v`; test loader gagal pada `from scripts.prepare_data import prepare_data`. `NativeCommandError` pada pembungkus PowerShell muncul bersama kegagalan impor Python, bukan bukti penyebab Python yang berbeda.
- **Hipotesis dan bukti sebab:** modul belum tersedia menurut urutan implementasi pada log. Traceback membuktikan `ModuleNotFoundError`; detail commit/versi saat gagal [PERLU DILENGKAPI: revisi kode atau snapshot saat kejadian].
- **Usulan/tindakan aktual:** log mencatat penulisan `scripts/prepare_data.py` dan pengujian menggunakan pasangan ZIP sintetis. Dataset final sekarang tidak memerlukan script persiapan ini untuk loading.
- **Verifikasi/status:** [tes pipeline](bukti/tes_orang2.txt) dan [suite berikutnya](bukti/tes_dataset_final_tanpa_zip.txt) mencatat 21 tes lulus, termasuk sebelas tes data. Status: terselesaikan menurut bukti pengujian lanjutan.
- **Draf prompt:** “Unittest `test_data.py` gagal mengimpor fungsi `prepare_data` seperti traceback terlampir. Periksa keberadaan modul dan root eksekusi. Implementasikan kontrak yang diminta tes tanpa menimpa sumber/output yang ada, lalu periksa ulang tes.”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: prompt asli, identitas pelaksana dan penjelasan kelompok].

## DBG-04 — Tes variasi augmentasi gagal karena fixture

- **Kejadian/sumber:** tahap pipeline, 27 Oktober 2026; LOG-02, bagian 7.3 log. Hanya pesan `AssertionError: False is not true` dan narasi perbaikan tersedia. [PERLU DILENGKAPI: traceback lengkap dan fixture gagal sebelum perubahan].
- **Pemicu/gejala:** tes variasi augmentasi train. Perintah persis saat gagal [PERLU DILENGKAPI: perintah asli]; nama tes pada kode sekarang adalah `test_augmentation_varies_on_train_but_not_on_test` di `tests/test_data.py`.
- **Hipotesis vs sebab dilaporkan:** variasi tidak terlihat dapat berasal dari transformasi, data yang salah terpilih, atau fixture seragam. Log melaporkan fixture gradient identik dengan sampel test sehingga deduplikasi mempertahankan test dan mengeluarkan fixture train. Sebab historis ini belum dapat dibuktikan ulang tanpa fixture gagal; bukan alasan mengubah perilaku augmentasi produksi.
- **Usulan/tindakan aktual:** log menyatakan fixture dibuat berbeda antara train/test dan deduplikasi dipertahankan. Kode sekarang memiliki fixture gradient train dan pemeriksaan test deterministik.
- **Verifikasi/status:** tes bernama tersebut tercatat lulus pada dua log 21 tes. Status fungsi kini didukung output lulus; penjelasan sebab lama masih bersandar pada narasi.
- **Draf prompt:** “Assertion variasi augmentasi gagal. Telusuri dulu apakah fixture train dipertahankan setelah deduplikasi, bandingkan hash pikselnya dengan fixture test, dan catat seed serta tensor sebelum/sesudah transformasi. Apakah yang salah fixture atau transformasi? Jangan menonaktifkan deduplikasi agar tes lulus.”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: versi fixture sebelum/sesudah, seed dan perintah saat gagal, prompt asli serta analisis anggota].

## DBG-05 — Helper fixture traversal gagal sebelum validasi produksi

- **Kejadian/sumber:** tahap pipeline, 27 Oktober 2026; LOG-02, bagian 7.3 log. Pesan yang tercatat: `ValueError: too many values to unpack (expected 2)`. [PERLU DILENGKAPI: traceback lengkap dan versi helper sebelum perbaikan].
- **Pemicu/gejala:** penambahan path fixture `trainA/../escape.png`; menurut log helper memakai `split, filename = name.split('/')`, sehingga tes gagal sebelum mencapai utilitas produksi.
- **Hipotesis vs sebab dilaporkan:** lebih dari satu separator menghasilkan lebih dari dua bagian. Narasi tersebut menjelaskan gejala; kode sekarang mengonfirmasi helper memakai `name.split('/', 1)`, tetapi tidak menyimpan reproduksi versi gagal.
- **Usulan/tindakan aktual:** batasi pemisahan pertama pada helper tes, pertahankan penolakan path traversal produksi. Tidak melonggarkan validasi trust boundary.
- **Verifikasi/status:** `test_archive_traversal_is_rejected` tercatat lulus pada [suite pipeline](bukti/tes_orang2.txt). Status tes terselesaikan; provenance perbaikan belum lengkap.
- **Draf prompt:** “Fixture traversal gagal pada unpacking helper, sebelum fungsi produksi diuji. Bedakan kegagalan helper dari validasi path. Bagaimana membuat helper dapat menulis path uji tersebut tanpa melemahkan penolakan traversal pada `image_members`?”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: perintah asli, snapshot gagal, prompt dan identitas penangan].

## DBG-06 — Kegagalan akses terminal historis

- **Kejadian/sumber:** 27 Oktober 2026, LOG-02; hanya log naratif menyimpan `CryptUnprotectData failed: 2148073483`. [PERLU DILENGKAPI: keluaran layanan lengkap, perintah pemicu dan konfigurasi runtime].
- **Gejala/hipotesis:** tool terminal gagal sebelum menjalankan perintah. Dugaan masalah dekripsi/kredensial lingkungan tidak dibuktikan hanya dengan satu pesan; tidak diatribusikan kepada model atau dataset.
- **Tindakan/status:** log menyatakan pembacaan kemudian berhasil melalui terminal di luar sandbox setelah pemeriksaan izin saat itu. Ini pemulihan akses yang dilaporkan, bukan perbaikan akar penyebab. Keadaan kebijakan/runtime historis tidak diasumsikan sama dengan sesi kini.
- **Draf prompt:** “Layanan terminal gagal sebelum perintah dijalankan dengan kode ini. Minta diagnosis runtime dari keluaran layanan lengkap dan kebijakan sesi; jangan mengubah kode model untuk mengatasi kegagalan tool. Langkah baca-saja apa yang dapat memastikan akses pulih?”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: bukti layanan, pemicu, status akar penyebab dan identitas pelapor].

## DBG-07 — Peringatan struktur PDF panduan

- **Kejadian/sumber:** LOG-01 dan pembacaan ulang LOG-06 pada 31 Oktober 2026. Pesan parser berupa `Ignoring wrong pointing object 6 0 (offset 0)` dan peringatan serupa untuk objek lain. Ini warning, bukan traceback exception.
- **Pemicu/lingkungan:** membaca PDF panduan dengan `pypdf.PdfReader` lalu mengekstrak teks. Hash PDF cocok dengan [inventaris historis](bukti/lingkungan.json); versi pypdf khusus kejadian awal tidak tercatat.
- **Hipotesis/bukti:** peringatan menunjukkan pointer objek PDF yang tidak cocok; penyebab dokumen menjadi demikian belum terbukti. Parser tetap menghasilkan lima halaman, termasuk rubrik dan template log. Tidak membuktikan semua elemen visual ter-render sempurna.
- **Usulan/tindakan aktual:** teks relevan diperiksa; pada sesi fondasi, log menyatakan gambar halaman yang tersedia juga dilihat. Sesi kompilasi hanya membaca teks, tidak merender ulang atau memperbaiki PDF sumber.
- **Status:** isi panduan dapat dibaca; warning struktur tetap muncul. [PERLU DILENGKAPI: salinan PDF sumber yang tervalidasi atau konfirmasi pemilik dokumen jika struktur perlu diperbaiki].
- **Draf prompt:** “PDF dengan hash yang sama menghasilkan warning pointer objek tetapi lima halaman teks terbaca. Bedakan kegagalan ekstraksi dari warning struktur. Bagian mana perlu dikonfirmasi melalui render atau salinan sumber tanpa mengubah bukti PDF asli?”

## DBG-08 — Konteks patch dokumentasi tidak ditemukan

- **Kejadian/sumber:** 27 Oktober 2026, LOG-03, bagian 8.4 log. Pesan tersedia sebagai ringkasan: `Failed to find expected lines`. [PERLU DILENGKAPI: keluaran patch lengkap bila masih tersimpan].
- **Pemicu/hipotesis:** edit memakai sebagian paragraf sebagai konteks yang seharusnya cocok dengan satu baris utuh, menurut log. Tidak ada gejala kerusakan model atau dataset yang dilaporkan.
- **Tindakan/verifikasi/status:** konteks diganti dengan judul dan patch berhasil menurut log; dokumen status tersedia. Status edit terselesaikan menurut narasi, tidak direproduksi ulang.
- **Draf prompt:** “Patch dokumentasi menolak konteks yang tidak cocok. Baca baris aktual dan susun konteks yang tepat tanpa penggantian global; verifikasi diff hanya menyentuh paragraf yang dimaksud.”
- **Kelengkapan kelompok:** [PERLU DILENGKAPI: patch gagal, prompt asli dan identitas penyunting].

## DBG-09 — Pengambilan ulang tautan percakapan gagal

- **Kejadian:** 31 Oktober 2026, LOG-06; layanan penelusuran memberi pesan utuh berikut saat membuka tautan yang sudah terdaftar:

```text
Failed to fetch https://chatgpt.com/s/cx_6ae56f2a66c4819196c604c1f9cd4f05: Cache miss
```

- **Gejala/hipotesis:** layanan tidak memperoleh halaman. Pesan `Cache miss` tidak membuktikan tautan tidak aktif, konten dihapus, atau hak akses salah. Bagian 10 log sebelumnya mencatat keberhasilan membuka tautan; catatan itu tidak ditimpa.
- **Tindakan aktual:** gunakan bukti lokal dan catatan cakupan terdahulu; tidak merekonstruksi percakapan yang tidak terbaca. Tidak ada perubahan kode atau login yang dilakukan.
- **Status:** verifikasi akses ulang belum berhasil melalui layanan ini; kelengkapan ekspor masih terbuka. [PERLU DILENGKAPI: ekspor percakapan asli dan konfirmasi akses tautan oleh anggota].
- **Draf prompt:** “Pengambilan tautan berbagi menghasilkan `Cache miss`, sedangkan catatan sebelumnya menyatakan halaman terbuka. Jangan menyimpulkan tautan mati. Bukti ekspor apa yang perlu diminta agar cakupan prompt/respons dapat diperiksa tanpa bergantung pada keberhasilan satu layanan?”

## Catatan yang bukan kasus debugging baru

Koreksi pin Pillow ke versi terpasang dan penolakan input invalid pada unit test sudah dijelaskan dalam log. Penjelasan lama yang mengaitkan perbedaan hitungan tes dengan urutan reviewer belum terverifikasi; klaim tersebut tidak dipakai untuk membuktikan eksekusi review. Angka hasil pengujian tetap mengacu output tes asli. Tidak ada bukti bahwa hal tersebut merupakan eror training. Metadata salah, grayscale, blur dan duplikasi adalah temuan kualitas data yang ditangani pada tahap persiapan, bukan kegagalan training yang boleh dikarang ulang.

**Tindak lanjut teknis:** pemilik pipeline perlu melampirkan fixture/traceback gagal untuk DBG-04/05; pemilik lingkungan menangani bukti DBG-06/07; tim dokumentasi meminta ekspor DBG-09. Tidak ada bug perilaku model atau training baru yang dibuktikan pada sesi ini. Perubahan kode hanya dilakukan oleh penanggung jawab modul melalui tugas terpisah jika diagnosis dengan bukti memang memerlukannya.
