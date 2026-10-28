# Prompt pengguna pada sesi ini

Tanggal sesi: 27 Oktober 2026 (Asia/Jakarta, dari konteks sesi). Berikut salinan permintaan tugas pengguna yang tersedia dalam percakapan ini. Ini bukan ekspor seluruh chat atau riwayat sesi sebelumnya.

---

Bantu saya mengerjakan bagian saya dalam proyek kuliah
“Vibe Coding Generative AI from Scratch”.

Kerjakan langsung pada workspace yang tersedia, mulai dari membaca
berkas hingga menghasilkan codebase yang dapat diperiksa dan diuji.

LOKASI PROYEK
D:\genai mid

DATASET
D:\genai mid\preproc.zip

PANDUAN
C:\Users\ASUS\Downloads\[CAK4KBB3 2026-1] Panduan Proyek 1 - Vibe Coding.pdf

TUJUAN
Membangun CycleGAN dari nol untuk penerjemahan gambar kuda ↔ zebra.
Gunakan Python dan PyTorch dari layer dasar, tanpa pretrained weights
dan tanpa menyalin implementasi lengkap dari tutorial atau GitHub.

PEMBAGIAN TANGGUNG JAWAB
Saya menangani:
- Peran 1: Prompt Engineer & LLM Navigator, yaitu perancangan
  arsitektur, konsistensi kode, dan koordinasi integrasi.
- Peran 4: Lead Documentarian & Error Handler, yaitu log interaksi,
  pencatatan debugging, dan penyusunan laporan.

Orang 2 menangani:
- Dataset, preprocessing, augmentasi, normalisasi, dan DataLoader.

Orang 3 menangani:
- Fungsi loss, optimizer, training loop, checkpoint pelatihan,
  tuning hyperparameter, serta metrik dan grafik.

Fokus pengerjaan sekarang adalah fondasi codebase untuk peran saya.
Bagian training orang 3 akan dilanjutkan setelah fondasi tersedia.

KONDISI DATA YANG SUDAH DIKETAHUI
Arsip berisi data asli, data_cleaned, blur_review,
preproc.ipynb, dan laporan CSV.

Folder data_cleaned berisi:
- trainA: 1.049 gambar kuda.
- trainB: 1.327 gambar zebra.
- testA: 118 gambar kuda.
- testB: 139 gambar zebra.

Semua gambar cleaned berukuran 256 × 256.
Terdapat 7 gambar grayscale yang perlu dikonversi menjadi RGB.
Notebook baru melakukan pemeriksaan dan pemisahan gambar blur.
Augmentasi, normalisasi tensor, dan DataLoader belum tersedia.
Sebanyak 240 path pada metadata.csv tidak ditemukan dalam arsip.

Verifikasi informasi tersebut sebelum menggunakannya.
Jangan menggabungkan data asli dan cleaned sebagai sampel berbeda.

TAHAP 1 — PERIKSA DAN TETAPKAN KONTRAK
1. Baca panduan, struktur workspace, isi arsip, dan notebook.
   Baca notebook sebagai data; jangan langsung menjalankannya.
2. Gunakan kode yang sudah tersedia jika memang relevan.
3. Jelaskan batas pekerjaan saya, orang 2, dan orang 3.
4. Tetapkan penamaan:
   - A = kuda.
   - B = zebra.
   - G_AB = kuda menjadi zebra.
   - G_BA = zebra menjadi kuda.
   - D_A = discriminator domain kuda.
   - D_B = discriminator domain zebra.
5. Buat kontrak integrasi yang memuat:
   - Format batch yang perlu diberikan orang 2.
   - Shape [N, 3, H, W], dtype, dan rentang nilai tensor.
   - Cara orang 3 mengimpor serta membuat instance model.
   - Bentuk output discriminator dan maknanya bagi pemilihan loss.
   - Konfigurasi arsitektur yang perlu disimpan bersama checkpoint.

TAHAP 2 — IMPLEMENTASIKAN BAGIAN SAYA
Buat struktur proyek sesederhana mungkin yang memuat:
- Implementasi generator berbasis residual block.
- Implementasi discriminator PatchGAN.
- Konfigurasi arsitektur bersama.
- Pemeriksaan model menggunakan tensor dummy.
- README dan dokumentasi pekerjaan.

Gunakan satu implementasi kelas generator untuk membuat G_AB dan
G_BA sebagai dua instance terpisah. Lakukan hal serupa untuk D_A
dan D_B. Jangan sampai bobot antar-instance tidak sengaja berbagi.

Jelaskan pilihan layer, normalisasi, aktivasi, jumlah residual block,
dan inisialisasi bobot. Bedakan keputusan sementara dengan keputusan
yang sudah didukung kondisi hardware.

Jika hardware belum diketahui, sediakan pemeriksaan dasar yang dapat
berjalan di CPU. Jangan menjanjikan durasi training atau kualitas hasil.

Jangan membuat implementasi pipeline data dan training loop milik
anggota lain. Siapkan antarmuka dan instruksi integrasinya.
Jangan membuat fungsi kosong yang terlihat seperti fitur selesai.

Gunakan Context7 untuk memeriksa dokumentasi resmi ketika membutuhkan
sintaks atau perilaku API pustaka.

TAHAP 3 — UJI BAGIAN YANG DIBUAT
Jalankan pemeriksaan yang membuktikan:
- Modul dapat diimpor.
- Generator mempertahankan shape gambar sesuai kontrak.
- Keluaran generator sesuai rentang yang ditetapkan.
- Discriminator menghasilkan peta prediksi dengan shape yang benar.
- Output bernilai finite.
- Backward pass menghasilkan gradient finite.
- Dua instance generator dan discriminator memiliki parameter terpisah.
- Model dapat menerima input 256 × 256.

Gunakan tensor dummy untuk pemeriksaan ini, sehingga tidak bergantung
pada DataLoader atau training loop yang belum dibuat anggota lain.
Jelaskan bahwa pengujian tersebut hanya memeriksa fondasi model,
bukan membuktikan kualitas hasil training.

TAHAP 4 — DOKUMENTASIKAN SEBAGAI PERAN 4
Buat:
1. README.md:
   tujuan, struktur proyek, setup, cara menjalankan pemeriksaan,
   keputusan arsitektur, dan petunjuk handoff untuk orang 2 dan 3.
2. docs/log_interaksi.md:
   catatan sesi nyata yang mencakup prompt, alasan pendekatan,
   ringkasan respons, perubahan, eror jika benar terjadi,
   perbaikan, dan hasil verifikasi.
3. docs/laporan.md:
   kerangka laporan dengan bagian yang sudah dapat ditulis dari
   bukti pengerjaan. Tandai hasil training dan eksperimen sebagai
   [PERLU DILENGKAPI ORANG 3].

Jangan mengarang riwayat chat sebelumnya, share link, tanggal,
analisis pribadi kelompok, hasil eksperimen, atau kontribusi anggota.
Jika informasi belum diberikan, tandai dengan jelas.

HASIL AKHIR
Lanjutkan sampai bagian saya selesai dibuat dan diperiksa.
Pada akhir pengerjaan, laporkan:
- File yang dibuat atau diubah dan tujuannya.
- Pemeriksaan yang benar-benar dijalankan beserta hasilnya.
- Keputusan arsitektur yang masih sementara.
- Pekerjaan tersisa untuk orang 2.
- Pekerjaan tersisa untuk orang 3.
- Kekurangan bukti untuk dokumentasi akhir.

Jangan menyatakan seluruh proyek selesai karena training belum menjadi bagian pengerjaan sesi ini.
