# CycleGAN kuda ↔ zebra: fondasi model

Fondasi untuk proyek kuliah **Vibe Coding Generative AI from Scratch**. Generator residual dan discriminator PatchGAN ditulis menggunakan layer dasar PyTorch dengan bobot acak. Tidak memakai pretrained weights atau menyalin implementasi CycleGAN lengkap dari tutorial/GitHub.

Status: fondasi model dan **pipeline orang 2 tersedia**. Dataset asli Kaggle telah dibandingkan dengan arsip lokal, metadata dikoreksi, RGB dan duplikasi dibereskan, serta batch nyata diperiksa pada empat model di CPU. Training milik orang 3 belum diimplementasikan. Lihat [status orang 2](docs/status_orang2.md).

**Root dataset aktif: `data/horse2zebra`.** Folder final ini langsung dipakai untuk loading, pemeriksaan, dan training orang 3 nantinya. Pengguna menyatakan telah menghapus `preproc.zip`; ketidakberadaannya sudah diverifikasi pada sesi penyesuaian. Arsip tersebut bukan prasyarat penggunaan dataset final. Riwayat persiapan dari ZIP dan hash sumber tetap dipertahankan.

## Pembagian pekerjaan

| Penanggung jawab | Cakupan |
|---|---|
| Pengguna, Peran 1 | Arsitektur, penamaan, konfigurasi bersama, konsistensi antarmuka, koordinasi integrasi |
| Pengguna, Peran 4 | Log sesi nyata, pencatatan masalah dan perbaikan, kerangka laporan, pengumpulan bukti |
| Orang 2 | Dataset, rekonsiliasi metadata/duplikasi, RGB, augmentasi, normalisasi, Dataset dan DataLoader |
| Orang 3 | Loss, optimizer, training loop, checkpoint pelatihan, tuning, metrik dan grafik |

Nama dan identitas anggota belum diberikan. Tabel ini adalah pembagian tanggung jawab, bukan klaim kontribusi yang telah mereka kerjakan.

## Struktur

```text
cyclegan/
  __init__.py              # ekspor publik
  config.py                # ArchitectureConfig, serializable dan immutable
  models.py                # Generator, ResidualBlock, PatchDiscriminator
  data.py                  # ImageDataset, UnpairedDataset, create_dataloaders
tests/
  test_models.py           # pemeriksaan model menggunakan tensor dummy
  test_audit_archive.py    # pemeriksaan audit dengan ZIP kecil buatan tes
  test_data.py             # persiapan data, augmentasi, normalisasi, worker Windows
scripts/
  audit_archive.py         # opsional: audit arsip sumber, bukan dataset final
  prepare_data.py          # opsional: persiapan dari dua ZIP sumber
  check_data.py            # seluruh berkas/epoch dan batch nyata pada empat model
docs/
  kontrak_integrasi.md
  log_interaksi.md
  laporan.md
  bukti/                   # prompt, audit JSON, lingkungan dan keluaran pemeriksaan
  superpowers/plans/       # rencana dan status pengerjaan
requirements.txt
data/horse2zebra/
  trainA/                  # 1.040 kuda
  trainB/                  # 1.311 zebra
  testA/                   # 118 kuda
  testB/                   # 139 zebra
  manifest.json
  metadata_corrected.csv
  preparation_report.json
```

`tmp/` sudah ada sebelum pengerjaan, berisi gambar inspeksi. Isinya bukan riwayat kontribusi kelompok. Dataset, environment, checkpoint, dan folder temporer dikecualikan dari Git melalui `.gitignore`.

## Setup dan pemeriksaan

Lingkungan yang benar-benar dipakai: Windows, Python 3.12.3, PyTorch 2.5.1+cu121, Pillow 12.3.0. Gunakan Python 3.12 untuk mengikuti lingkungan ini. Pipeline menggunakan DataLoader bawaan PyTorch dan Pillow, tanpa torchvision, notebook, NumPy, atau GPU. Audit ulang skor blur pada sesi ini memakai OpenCV/NumPy yang sudah terpasang; keduanya bukan dependensi pipeline.

Jalankan dari direktori proyek di PowerShell. Untuk environment baru:

```powershell
Set-Location -LiteralPath 'D:\genai mid'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Perintah setup environment baru belum dijalankan pada sesi ini; pemeriksaan menggunakan paket yang sudah terpasang. Untuk menggunakan instalasi yang telah diperiksa:

```powershell
python -c "from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator; print(ArchitectureConfig())"
python -m unittest discover -s tests -v
python scripts/check_data.py --root data/horse2zebra --output docs/bukti/pemeriksaan_dataset_final_tanpa_zip.json
```

Semua tes model secara eksplisit menggunakan CPU, empat thread, dan seed 7. Baseline penuh diuji pada batch satu gambar 256 × 256; konfigurasi kecil hanya dipakai untuk pemeriksaan tambahan. `check_data.py` membaca folder final, memeriksa hash dan seluruh gambar, mengiterasi train/testA/testB, lalu memeriksa satu batch nyata pada empat model. Tidak membuka arsip ZIP. Bukti baru disimpan terpisah dari [bukti fondasi historis](docs/bukti/pemeriksaan.txt) dan [pemeriksaan data sebelumnya](docs/bukti/pemeriksaan_data.json): [hasil data/model tanpa ZIP](docs/bukti/pemeriksaan_dataset_final_tanpa_zip.json) dan [log pemeriksaan sesi penyesuaian](docs/bukti/tes_dataset_final_tanpa_zip.txt).

Sebagian unit test membuat ZIP sintetis di folder temporer untuk menguji utilitas persiapan/audit; tes tersebut tidak memerlukan `preproc.zip` sumber yang sudah dihapus dan tidak membuat arsip pengganti dari dataset final.

## Temuan awal arsip, sebelum pipeline orang 2

Bagian ini adalah **riwayat audit sebelum dataset final disiapkan**, bukan petunjuk loading saat ini. Kondisi dan tindakan yang disebut di sini berlaku pada tahap awal tersebut; penggunaan saat ini dijelaskan pada bagian pipeline final di bawah.

| Split | Asli | Cleaned | Review blur |
|---|---:|---:|---:|
| trainA: kuda | 1.067 | 1.049 | 18 |
| trainB: zebra | 1.334 | 1.327 | 7 |
| testA: kuda | 120 | 118 | 2 |
| testB: zebra | 140 | 139 | 1 |
| Total | 2.661 | 2.633 | 28 |

Semua 2.633 cleaned dapat didekode dan berukuran 256 × 256: 2.626 RGB dan tujuh mode `L` (grayscale), seluruhnya di trainB. Daftar path tersedia dalam [audit JSON](docs/bukti/audit_dataset.json).

Seluruh cleaned identik byte dengan berkas asalnya. Pada tahap persiapan, **hanya `data_cleaned` menjadi sumber kandidat sampel**, tidak digabung dengan data asli atau `blur_review`. Jumlah tersebut adalah jumlah berkas sebelum keputusan deduplikasi lebih lanjut; loader sekarang memakai `data/horse2zebra`.

Metadata awal memiliki 2.661 baris; 240 path tidak ditemukan jika di-resolve terhadap root `data/` dalam arsip: 120 `testB/...` ditemukan berdasarkan nama berkas di `testA`, dan 120 `trainA/...` ditemukan di `trainB`. Ini bukti ketidakcocokan path/domain pada metadata, bukan 240 gambar yang pasti hilang. Rekonsiliasi dilakukan pada tahap orang 2 dan dicatat dalam metadata terkoreksi.

Audit awal menemukan **25 kelompok berkas cleaned identik berdasarkan SHA-256**, termasuk satu pasangan lintas train/test: `testA/n02381460_1160.jpg` dan `trainA/n02381460_3811.jpg`. Temuan ini ditangani pada persiapan dataset final. Audit awal mendeteksi duplikasi byte, belum mencakup gambar yang sama dengan encoding atau crop berbeda. Pada sesi fondasi awal, berkas sumber tidak dihapus atau direlokasi oleh AI.

Notebook dibaca sebagai JSON. Kodenya menghitung variansi Laplacian dan menyalin gambar dengan skor `<100` ke review; hasil partisi CSV cocok dengan isi ZIP. Tulisan `Image blur: 0` pada output notebook sebenarnya menghitung skor kosong/gambar tak terbaca, bukan jumlah skor di bawah threshold. Terdapat 28 gambar dalam review. `convert("RGB")` di notebook hanya untuk tampilan plot, bukan konversi dataset tersimpan. Pada tahap notebook tersebut, belum ada augmentasi, normalisasi tensor, atau DataLoader.

## Pipeline orang 2: penggunaan dan hasil final

Provenance persiapan: [Horse2zebra Dataset, Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), versi 1. Unduhan sumber masih tersedia di `data/source/horse2zebra-kaggle.zip`; loader tidak membutuhkannya. Pada sesi persiapan sebelumnya, seluruh 2.661 gambar dibandingkan byte demi byte dengan `data/` pada `preproc.zip`. CSV metadata Kaggle identik dengan CSV lokal, termasuk 240 baris yang salah path/domain. Pipeline memperbaikinya pada salinan `metadata_corrected.csv`, tanpa mengubah sumber pada saat itu. `preproc.zip` sekarang sudah tidak tersedia.

Dataset siap pakai berada di `data/horse2zebra/`. Semua 2.608 berkas akhir berupa PNG RGB 256 x 256; tujuh grayscale dipertahankan dengan kanal RGB identik, sehingga tampilannya tetap abu-abu. Deduplikasi membandingkan piksel RGB, memprioritaskan test sebelum train, lalu nama path. Sebanyak 28 gambar blur dan 25 salinan duplikat dikecualikan. Tidak ada piksel identik lintas train/test yang tersisa.

| Split final | Gambar |
|---|---:|
| trainA: kuda | 1.040 |
| trainB: zebra | 1.311 |
| testA: kuda | 118 |
| testB: zebra | 139 |

Langsung muat folder final dengan API yang sudah tersedia; tidak perlu menjalankan persiapan ulang atau mengunduh sumber. `manifest.json` menentukan sampel tiap split. `metadata_corrected.csv` mencatat seluruh 2.661 gambar sumber, sehingga bukan daftar 2.608 sampel final untuk loader.

Contoh integrasi training (simpan sebagai skrip dengan main guard pada Windows):

```python
import torch
from cyclegan import create_dataloaders

def main():
    torch.manual_seed(7)
    loaders = create_dataloaders("data/horse2zebra", batch_size=1, num_workers=0, seed=7)
    batch = next(iter(loaders["train"]))
    real_A, real_B = batch["A"], batch["B"]
    test_A = next(iter(loaders["testA"]))
    test_B = next(iter(loaders["testB"]))
    print(real_A.shape, real_B.shape, test_A.shape, test_B.shape)

if __name__ == "__main__":
    main()
```

Augmentasi berlangsung **saat loading di `cyclegan/data.py`**, bukan disimpan sebagai gambar baru. Train melakukan resize bicubic 286, crop acak 256 (offset 0..30), flip horizontal peluang 0,5, lalu `2*(pixel/255)-1`, CHW float32. Batch train berupa dictionary `A`/`B`, masing-masing float32 `[N,3,256,256]` dalam [-1,1]. Transform domain A/B independen. Panjang epoch 1.311: domain lebih besar dikunjungi sekali, domain lebih kecil disampling acak dengan pengembalian. Batch terakhir tetap disertakan. Loader `testA` dan `testB` masing-masing menghasilkan tensor domain tunggal, tanpa transform acak; jangan digabung memakai `zip` karena ukuran keduanya berbeda. Orang 3 dapat langsung meneruskan batch ini ke model dan training loop yang akan dibuatnya.

Parameter `seed` mengatur shuffle dan seed worker DataLoader. Untuk `num_workers=0`, panggil `torch.manual_seed` sebelum iterasi agar augmentasi/sampling dapat diulang. Dengan jumlah worker dan urutan pemanggilan sama, alur RNG dapat direproduksi; kelanjutan training harus menyimpan state RNG dan konfigurasi loader.

Pemeriksaan nyata mencakup seluruh 2.608 file, satu epoch train serta masing-masing test, dan forward batch `[1,3,256,256]` ke G_AB/G_BA/D_A/D_B default. Generator mengeluarkan `[1,3,256,256]`, discriminator `[1,1,30,30]`, semuanya finite. Bukti: [pemeriksaan data](docs/bukti/pemeriksaan_data.json), [manifest](docs/bukti/manifest.json), [audit persiapan](docs/bukti/preparation_report.json), dan [penghitungan ulang blur](docs/bukti/verifikasi_blur.json).

Hasil evaluasi kelak harus disebut **subset test cleaned**, bukan benchmark Kaggle penuh. Tidak ada split validasi yang dipilih dalam handoff ini: orang 3 harus menyisihkan validasi dari train sebelum tuning/pemilihan checkpoint. Deduplikasi piksel belum membuktikan bebas gambar mirip dengan encoding/crop berbeda.

## Utilitas arsip opsional dan bukti historis

`scripts/prepare_data.py` membutuhkan **dua berkas ZIP sumber**: `--preproc` untuk arsip preprocessing lama dan `--source` untuk ZIP Kaggle. CLI mem-pin SHA-256 ZIP Kaggle versi 1 ke `0655d8feb8eb83911ac08c1d230a0464794418c107412a28adc47e1eb69e90a0` dan menolak output yang sudah ada. `scripts/audit_archive.py` membutuhkan arsip preprocessing lama untuk mengaudit `data`, `data_cleaned`, `blur_review`, CSV dan notebook di dalamnya. Keduanya dipertahankan untuk riwayat/reproduksi opsional jika arsip asli tersedia lagi; **bukan langkah wajib penggunaan folder final**. Folder `data/horse2zebra` bukan pengganti argumen ZIP kedua utilitas tersebut.

Karena `preproc.zip` telah dihapus oleh pengguna, audit ZIP dan persiapan dari pasangan sumber asli tidak dapat diulang sepenuhnya saat ini. Hash dan laporan lama tetap menjadi catatan provenance; penghapusan tidak mengubah riwayat bahwa dataset final dibuat dari ZIP. Dataset final tetap dapat diperiksa terhadap manifest dan preparation report tanpa arsip. Sesi penyesuaian tidak mengulang cleaning, deduplikasi, atau augmentasi offline dan tidak membuat arsip pengganti.

## Arsitektur dan pemakaian

```python
from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator

config = ArchitectureConfig()
G_AB = Generator(config)          # kuda -> zebra
G_BA = Generator(config)          # zebra -> kuda
D_A = PatchDiscriminator(config) # domain kuda
D_B = PatchDiscriminator(config) # domain zebra
```

Konstruktor mengalokasikan dan menginisialisasi parameter baru setiap kali dipanggil. Berbagi `config` immutable aman; jangan memakai assignment `G_BA = G_AB` atau `D_B = D_A`.

| Bagian | Keputusan awal dan alasan |
|---|---|
| Generator | Conv 7×7, dua downsampling Conv 3×3, sembilan residual block di resolusi 64×64, dua ConvTranspose 3×3, Conv 7×7 dan Tanh |
| Kanal generator | 3 → 64 → 128 → 256 → 128 → 64 → 3; fitur bertambah ketika resolusi mengecil |
| Residual block | Dua Conv 3×3 dengan InstanceNorm, ReLU hanya setelah conv pertama; penjumlahan `x + F(x)` mempertahankan jalur identitas fitur |
| Padding generator | Reflection padding pada conv 7×7 dan residual, untuk mengurangi ketergantungan pada border nol; downsampling memakai padding 1 |
| Upsampling | ConvTranspose dengan stride 2, padding 1, output_padding 1 mengembalikan ukuran tepat untuk H/W kelipatan empat |
| Normalisasi | InstanceNorm2d, eps 1e-5, affine=False, track_running_stats=False; statistik per gambar, mendukung batch satu; tetap memakai statistik input saat eval |
| Aktivasi | ReLU generator, LeakyReLU(0.2) discriminator; Tanh terakhir generator memetakan output ke [-1,1] |
| Discriminator | Lima Conv 4×4, padding 1, stride 2/2/2/1/1; kanal 3/64/128/256/512/1; tanpa normalisasi di conv pertama/terakhir |
| Output discriminator | `[N,1,30,30]` untuk 256×256; skor mentah, tanpa sigmoid dan tanpa flatten; receptive field konvolusional nominal 70×70 |
| Inisialisasi | Semua Conv/ConvTranspose menggunakan normal(mean=0,std=0.02), bias 0; InstanceNorm non-affine tidak mempunyai parameter scale/bias |

InstanceNorm menghitung statistik spasial, sehingga ukuran patch nominal bukan klaim bahwa setiap skor sepenuhnya independen dari bagian gambar lain. Semua convolution memakai bias. Tidak ada dropout, pretrained backbone, atau pemilihan device tersembunyi. Jumlah parameter: **11.378.179 per generator**, **2.764.737 per discriminator**.

Pilihan 64 filter, sembilan block, normalisasi, inisialisasi, dan metode upsampling adalah baseline arsitektur, belum hasil tuning. Inventaris mendeteksi RTX 3060 Laptop dengan VRAM 6 GiB, tetapi training empat model beserta optimizer dan graph cycle belum diukur. Batch satu pada tes adalah pilihan pemeriksaan, bukan rekomendasi kapasitas training. Orang 3 dapat mengevaluasi filter/block lebih kecil melalui konfigurasi dan harus menyimpan konfigurasi aktual. Tidak ada janji durasi training atau kualitas hasil.

## Handoff

Baca [kontrak integrasi](docs/kontrak_integrasi.md) sebelum menghubungkan modul.

- **Orang 2:** pipeline selesai dan diperiksa; gunakan `create_dataloaders` serta manifest final di atas. Batch train `{"A": tensor, "B": tensor}`, float32 RGB `[N,3,256,256]`, finite dalam [-1,1]. Test terpisah per domain. Simpan hash manifest dalam checkpoint orang 3.
- **Orang 3:** gunakan empat konstruktor di atas, pilih loss yang sesuai skor mentah, implementasikan optimizer/training/checkpoint dan evaluasi. Simpan `dataclasses.asdict(config)` bersama empat state_dict. Verifikasi penggunaan memori training nyata sebelum menetapkan hyperparameter.
- **Peran 4:** lanjutkan log dengan sesi nyata berikutnya; kumpulkan ekspor chat asli, identitas kelompok, bukti kontribusi, grafik, checkpoint dan contoh hasil dari anggota yang mengerjakannya.

## Referensi yang dipakai

Panduan lokal: `C:\Users\ASUS\Downloads\[CAK4KBB3 2026-1] Panduan Proyek 1 - Vibe Coding.pdf`, lima halaman. Halaman 1 mengatur penggunaan dokumentasi resmi hanya untuk fungsi dasar; halaman 2 pembagian peran; halaman 3 memuat CycleGAN; halaman 5 rubrik dan log. Hash dokumen tercatat di [lingkungan](docs/bukti/lingkungan.json).

Context7 digunakan untuk memeriksa API dasar, bukan mengambil implementasi model lengkap: [InstanceNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html), [ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html), [inisialisasi](https://docs.pytorch.org/docs/stable/nn.init.html), [serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), dan [Pillow file lifecycle](https://pillow.readthedocs.io/en/stable/reference/open_files.html). Sebagian hasil indeks PyTorch menunjuk dokumentasi main/2.14; kompatibilitas kode yang diserahkan diperiksa langsung pada versi terpasang 2.5.1. Detail penelusuran dicatat dalam log.
