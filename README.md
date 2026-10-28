# CycleGAN untuk Penerjemahan Gambar Kuda ↔ Zebra

Proyek tugas kelompok mata kuliah dengan tema **Vibe Coding Generative AI from Scratch**. CycleGAN dibangun dari layer dasar PyTorch untuk menerjemahkan gambar kuda dan zebra tanpa pasangan gambar satu-ke-satu.

| Identitas akademik | Keterangan |
|---|---|
| Mata kuliah | [NAMA MATA KULIAH] |
| Program studi / institusi | [PROGRAM STUDI DAN INSTITUSI] |
| Kelas / kelompok | [KELAS] / [NAMA ATAU NOMOR KELOMPOK] |
| Dosen pengampu | [NAMA DOSEN] |
| Anggota | [NAMA DAN NIM SETIAP ANGGOTA] |

## Tujuan dan metode

Tujuan proyek adalah membangun, mengintegrasikan, dan menguji model penerjemahan gambar dua arah dari nol. Domain A adalah kuda dan domain B adalah zebra. Dua generator, G_AB dan G_BA, menangani arah penerjemahan yang berlawanan; D_A dan D_B menilai citra pada domain masing-masing.

Pengembangan menggunakan metode *vibe coding*: kebutuhan dan rancangan dirumuskan melalui prompt, kode disusun dengan bantuan LLM, lalu diperiksa melalui pengujian dan pencatatan masalah. Dokumentasi resmi menjadi acuan API dasar. Implementasi tidak memakai pretrained weights atau menyalin implementasi CycleGAN lengkap dari tutorial. Pernyataan proses ini bukan hasil audit orisinalitas menyeluruh. Prompt, respons, debugging, dan penggunaan LLM disimpan dalam [log interaksi](docs/log_interaksi.md) serta [diskusi teknis](docs/diskusi_claude.md).

## Status implementasi

| Bagian | Status |
|---|---|
| Model | Generator residual, PatchGAN, dan konfigurasi bersama tersedia |
| Pipeline data | Dataset, augmentasi saat loading, normalisasi, dan DataLoader unpaired tersedia |
| Pengujian | Bukti 21 tes lulus serta pemeriksaan seluruh dataset dan batch nyata tersedia |
| Pelatihan dan tuning | Belum diimplementasikan; tanggung jawab Peran 3 |
| Hasil eksperimen | Belum tersedia: grafik, metrik kualitas, checkpoint pelatihan, dan gambar hasil |

Generator menggunakan filter dasar 64, sembilan residual block, InstanceNorm, ReLU, dan output Tanh. PatchGAN menghasilkan skor mentah tanpa sigmoid. Bobot convolution diinisialisasi normal(0, 0.02) dengan bias nol. Dua instance masing-masing kelas mempunyai parameter terpisah. Kapasitas ini masih berupa baseline, belum hasil tuning; rincian dan alasan rancangan terdapat dalam [laporan](docs/laporan.md) dan [kontrak integrasi](docs/kontrak_integrasi.md).

## Struktur repository

```text
cyclegan/
  config.py               # konfigurasi arsitektur
  models.py               # Generator, ResidualBlock, PatchDiscriminator
  data.py                 # ImageDataset, UnpairedDataset, create_dataloaders
  __init__.py             # ekspor API publik
scripts/
  check_data.py           # pemeriksaan dataset final dan batch nyata
  prepare_data.py         # utilitas opsional persiapan dari ZIP sumber
  audit_archive.py        # utilitas opsional audit ZIP sumber
tests/                    # tes model, pipeline, dan utilitas arsip
docs/
  kontrak_integrasi.md
  laporan.md
  status_orang2.md         # status teknis Peran 2; bukan atribusi anggota
  log_interaksi.md
  diskusi_claude.md
  bukti/                  # bukti asli; tidak ditimpa saat pemeriksaan baru
  superpowers/plans/      # arsip proses perancangan
requirements.txt
```

## Setup

Jalankan dari root repository. Lingkungan pengujian yang tercatat menggunakan Python 3.12.3, PyTorch 2.5.1+cu121, dan Pillow 12.3.0. Untuk environment baru di PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Gunakan `python` dari environment yang telah disiapkan. Jika environment belum diaktifkan, ganti `python` pada perintah berikut dengan `.\.venv\Scripts\python.exe`. Pipeline memakai PyTorch dan Pillow tanpa torchvision atau NumPy. GPU tidak diperlukan untuk pemeriksaan CPU yang tersedia. Instalasi environment baru belum menjadi bagian bukti pengujian.

## Penempatan dataset

**Dataset tidak disertakan dalam Git dan dibagikan secara terpisah.** Lokasi atau tautan distribusi final: [BELUM DIBERIKAN]. Setelah menerima dataset, tempatkan seluruh paket pada struktur berikut:

```text
data/horse2zebra/
  trainA/                  # 1.040 gambar kuda
  trainB/                  # 1.311 gambar zebra
  testA/                   # 118 gambar kuda
  testB/                   # 139 gambar zebra
  manifest.json
  metadata_corrected.csv
  preparation_report.json
```

Dataset final berisi 2.608 PNG RGB 256 × 256. Loader menggunakan `manifest.json` untuk memilih sampel. `metadata_corrected.csv` mencatat 2.661 gambar sumber, termasuk yang dikecualikan; berkas ini bukan daftar sampel akhir untuk loader. `preparation_report.json` menyimpan hasil persiapan dan hash manifest. Rincian sumber, koreksi metadata, konversi RGB, dan pengecualian tersedia dalam [status pipeline data](docs/status_orang2.md).

`preproc.zip` merupakan arsip persiapan historis yang telah dihapus, bukan prasyarat penggunaan dataset final. `prepare_data.py` membutuhkan ZIP preprocessing dan ZIP Kaggle; `audit_archive.py` membutuhkan ZIP preprocessing. Keduanya opsional dan tidak menerima folder final sebagai pengganti ZIP. Audit serta hash sumber lama tetap dipertahankan.

## Penggunaan

Contoh memuat ketiga loader; simpan sebagai skrip dengan main guard pada Windows:

```python
import torch
from cyclegan import create_dataloaders

def main():
    torch.manual_seed(7)
    loaders = create_dataloaders("data/horse2zebra", batch_size=1, num_workers=0, seed=7)
    train_batch = next(iter(loaders["train"]))
    test_A = next(iter(loaders["testA"]))
    test_B = next(iter(loaders["testB"]))
    print(train_batch["A"].shape, train_batch["B"].shape, test_A.shape, test_B.shape)

if __name__ == "__main__":
    main()
```

Batch train adalah dictionary `A`/`B` dengan tensor RGB `float32 [N,3,256,256]`, finite, dalam `[-1,1]`. Augmentasi dilakukan saat loading di `cyclegan/data.py`: resize bicubic 286, crop acak 256, dan flip horizontal dengan peluang 0,5, diikuti normalisasi `2*(pixel/255)-1`. Gambar hasil augmentasi tidak disimpan sebagai sampel baru.

Satu epoch train berisi 1.311 pasangan unpaired: domain lebih besar dikunjungi sekali, domain lebih kecil disampling acak dengan pengembalian. Loader testA dan testB menghasilkan tensor domain tunggal, deterministik, dan tetap menyertakan batch terakhir. Keduanya dievaluasi terpisah agar tidak ada sampel terpotong oleh `zip`.

Parameter `seed` mengatur shuffle dan seed worker. Untuk `num_workers=0`, panggil `torch.manual_seed` sebelum iterasi. Peran 3 dapat langsung menggunakan loader ini serta API model berikut:

```python
from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator

config = ArchitectureConfig()
G_AB = Generator(config)
G_BA = Generator(config)
D_A = PatchDiscriminator(config)
D_B = PatchDiscriminator(config)
```

Format output, pemilihan loss, perpindahan device, dan konfigurasi checkpoint dijelaskan dalam [kontrak integrasi](docs/kontrak_integrasi.md). Training loop belum tersedia.

## Pengujian dan hasil aktual

```powershell
python -m unittest discover -s tests -v
python scripts/check_data.py --root data/horse2zebra --output tmp/pemeriksaan_data_baru.json
```

Gunakan output baru agar bukti historis tidak tertimpa. Pemeriksaan data membutuhkan paket dataset final, sedangkan unit test memakai fixture sintetis dan tidak membutuhkan arsip sumber.

Bukti pemeriksaan 27 Oktober 2026 mencatat 21 tes lulus: tujuh tes model, tiga audit, dan sebelas data. Seluruh 2.608 gambar diperiksa mode, ukuran, hash file/piksel, serta kecocokannya dengan manifest. Semua loader diiterasi; batch nyata diterima empat model di CPU. Output generator adalah `[1,3,256,256]`, discriminator `[1,1,30,30]`, seluruhnya finite. Bukti: [log tes](docs/bukti/tes_dataset_final_tanpa_zip.txt), [pemeriksaan data/model](docs/bukti/pemeriksaan_dataset_final_tanpa_zip.json), dan [preservasi dataset](docs/bukti/preservasi_dataset_final_tanpa_zip.json). Hasil ini memverifikasi fondasi dan pipeline, bukan keberhasilan training.

## Pembagian tugas dan kontribusi aktual

| Peran awal | Tanggung jawab | Penanggung jawab awal | Kontribusi aktual yang dilaporkan |
|---|---|---|---|
| 1 — Prompt Engineer & LLM Navigator | Rancangan, prompt, konsistensi kode, integrasi | [NAMA/NIM PELAKSANA PERAN 1 DAN 4] | Menangani fondasi model dan koordinasi teknis dengan bantuan LLM |
| 2 — Data Pipeline & Preprocessing Engineer | Dataset, preprocessing, augmentasi, normalisasi, DataLoader | [NAMA/NIM PENANGGUNG JAWAB AWAL PERAN 2] | Pengerjaan teknis diambil alih oleh pelaksana Peran 1 dan 4 |
| 3 — Training Loop & Hyperparameter Tuning Engineer | Loss, optimizer, training, tuning, checkpoint, metrik | [NAMA/NIM PENANGGUNG JAWAB PERAN 3] | Implementasi dan hasil training belum tersedia |
| 4 — Lead Documentarian & Error Handler | Log interaksi, debugging, laporan | Pelaksana yang sama dengan Peran 1 | Menyusun dokumentasi dan mencatat verifikasi dengan bantuan LLM |

Pemetaan kontribusi ini mengikuti klarifikasi pelaksana proyek. Nama berkas `status_orang2.md` dan label “orang 2” pada bukti lama menunjukkan lingkup peran, bukan bukti bahwa anggota yang semula ditugaskan pada Peran 2 menulis implementasinya. Kontribusi anggota lain yang belum dilaporkan tidak diasumsikan.

## Keterbatasan dan pekerjaan berikutnya

Pelatihan, tuning, penggunaan memori training, checkpoint/resume, dan evaluasi kualitas belum dilakukan. Inventaris terdahulu mencatat RTX 3060 Laptop 6 GiB; kapasitas training penuh belum diukur. Peran 3 perlu menentukan konfigurasi pelatihan dan menyisihkan validasi dari train sebelum memilih hyperparameter atau checkpoint.

Test merupakan subset yang telah melalui penyaringan blur, bukan benchmark Kaggle penuh. Pemeriksaan piksel identik belum menjamin tidak ada gambar mirip atau crop berbeda. Tanpa ZIP preprocessing asli, audit dan persiapan ulang terhadap arsip tersebut tidak dapat diulang sepenuhnya. Identitas akademik, distribusi dataset final, bukti kontribusi anggota, ekspor chat lengkap, dan refleksi kelompok masih perlu dilengkapi.

## Referensi dan dokumentasi pendukung

- Panduan tugas: *[CAK4KBB3 2026-1] Panduan Proyek 1 — Vibe Coding*. Identitas dokumen dan hash historis tersimpan dalam [inventaris](docs/bukti/lingkungan.json).
- Sumber dataset: [Horse2zebra Dataset, Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), versi 1 menurut [metadata yang diarsipkan](docs/bukti/kaggle_metadata.json). Tautan ini adalah sumber publik, bukan tautan paket dataset final kelompok.
- API dasar resmi: [PyTorch](https://docs.pytorch.org/docs/stable/index.html) dan [Pillow](https://pillow.readthedocs.io/en/stable/). Rincian penelusuran API melalui Context7 tersimpan dalam log.
- Dokumen proyek: [laporan akademik](docs/laporan.md), [kontrak integrasi](docs/kontrak_integrasi.md), [status pipeline](docs/status_orang2.md), dan [log interaksi LLM](docs/log_interaksi.md).
