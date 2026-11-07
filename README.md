# CycleGAN untuk Penerjemahan Gambar Kuda ↔ Zebra

Proyek tugas kelompok mata kuliah dengan tema **Vibe Coding Generative AI from Scratch**. CycleGAN dibangun dari layer dasar PyTorch untuk menerjemahkan gambar kuda dan zebra tanpa pasangan gambar satu-ke-satu.

| Identitas akademik | Keterangan |
|---|---|
| Mata kuliah | Generative AI |
| Program studi / institusi | S1 Informatika / Telkom University |
| Kelas / kelompok | GENERATIVE AI BS1IF-47-GAB03 / Kelompok 2 |
| Anggota | Jian Hazel Sitorus (103012300187); Mikhael Restu Mahardhika (103012300277); Reevanza Abel Desta Arifin (103012330104) |

## Tujuan dan metode

Tujuan proyek adalah membangun, mengintegrasikan, dan menguji model penerjemahan gambar dua arah dari nol. Domain A adalah kuda dan domain B adalah zebra. Dua generator, G_AB dan G_BA, menangani arah penerjemahan yang berlawanan; D_A dan D_B menilai citra pada domain masing-masing.

Pengembangan menggunakan metode *vibe coding* melalui **GPT/ChatGPT**, sesuai klarifikasi tim dan [percakapan yang diberikan](https://chatgpt.com/s/cx_6ae56f2a66c4819196c604c1f9cd4f05). Prompt digunakan untuk merumuskan kebutuhan, menyusun kode dan meninjau hasil pengujian. Nama atau versi model GPT tidak ditetapkan. Dokumentasi resmi menjadi acuan API dasar; implementasi tidak memakai pretrained weights. Uraian proses ini bukan audit orisinalitas menyeluruh. Cakupan percakapan, perbedaan permintaan alat dengan pelaksanaan yang terbukti, serta koreksi atribusi tersedia dalam [diskusi GPT/ChatGPT](docs/diskusi_llm.md) dan [log interaksi](docs/log_interaksi.md). Klaim penggunaan alat lain atau review independen yang belum terverifikasi dikeluarkan dari dasar kesimpulan.

## Status implementasi

| Bagian | Status |
|---|---|
| Model | Generator residual, PatchGAN, dan konfigurasi bersama tersedia |
| Pipeline data | Dataset, augmentasi saat loading, normalisasi, dan DataLoader unpaired tersedia |
| Pengujian | Bukti 21 tes lulus serta pemeriksaan seluruh dataset dan batch nyata tersedia |
| Pelatihan dan tuning | Skrip training/evaluasi tersedia dari commit anggota lain `5fe225b`, digabungkan 3 November 2026; hasil eksperimen belum terverifikasi |
| Hasil eksperimen | Belum ditemukan log run, grafik, metrik kualitas, checkpoint pelatihan atau gambar hasil yang dapat diverifikasi |

Generator menggunakan filter dasar 64, sembilan residual block, InstanceNorm, ReLU, dan output Tanh. PatchGAN menghasilkan skor mentah tanpa sigmoid. Bobot convolution diinisialisasi normal(0, 0.02) dengan bias nol. Dua instance masing-masing kelas mempunyai parameter terpisah. Kapasitas ini masih berupa baseline, belum hasil tuning; rincian dan alasan rancangan terdapat dalam [laporan](docs/laporan.md) dan [kontrak integrasi](docs/kontrak_integrasi.md).

## Struktur repository

```text
cyclegan/
  config.py               # konfigurasi arsitektur
  models.py               # Generator, ResidualBlock, PatchDiscriminator
  __init__.py             # ekspor API publik
preprocessing/
  __init__.py             # paket kode preprocessing
  data.py                 # ImageDataset, UnpairedDataset, create_dataloaders
  check_data.py           # pemeriksaan dataset final dan batch nyata
  prepare_data.py         # utilitas opsional persiapan dari ZIP sumber
  audit_archive.py        # utilitas opsional audit ZIP sumber
scripts/
  train_cyclegan.py        # training, validasi, checkpoint dan evaluasi
  evaluate_cyclegan.py     # evaluasi checkpoint terpilih tanpa training
tests/                    # tes model, pipeline, dan utilitas arsip
docs/
  training_lokal.md        # penggunaan skrip training/evaluasi
  kontrak_integrasi.md
  laporan.md
  status_orang2.md         # status teknis Peran 2; bukan atribusi anggota
  log_interaksi.md
  diskusi_llm.md           # sumber GPT/ChatGPT dan register atribusi
  diskusi_claude.md        # arsip belum terverifikasi; bukan bukti valid
  koordinasi_dokumentasi.md # paket kerja, inventaris bukti, checklist rubrik
  debugging.md             # gejala, hipotesis, perbaikan dan batas bukti
  bukti/                  # bukti asli; tidak ditimpa saat pemeriksaan baru
  superpowers/plans/      # arsip proses perancangan
requirements.txt
```

`preprocessing/` berisi **kode** persiapan, audit, loading dan pemeriksaan data. `data/horse2zebra/` berisi **hasil dataset** beserta manifest/metadata; memindahkan kode tidak memindahkan atau memproses ulang gambar. `cyclegan/` tetap berfokus pada model dan konfigurasi.

Penataan 3 November 2026 memindahkan implementasi tanpa salinan atau wrapper:

| Lokasi lama | Lokasi aktif |
|---|---|
| `cyclegan/data.py` | `preprocessing/data.py` |
| `scripts/prepare_data.py` | `preprocessing/prepare_data.py` |
| `scripts/audit_archive.py` | `preprocessing/audit_archive.py` |
| `scripts/check_data.py` | `preprocessing/check_data.py` |

API publik `from cyclegan import create_dataloaders` tetap berlaku melalui re-export langsung; `ImageDataset` dan `UnpairedDataset` juga tetap diekspor. Import internal menggunakan `preprocessing.data`; tidak ada shim untuk path modul lama. Referensi lama pada prompt, traceback, hasil tes dan respons LLM tetap menjadi riwayat.

## Setup

Jalankan dari root repository. Lingkungan pengujian yang tercatat menggunakan Python 3.12.3, PyTorch 2.5.1+cu121, dan Pillow 12.3.0. Untuk environment baru di PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Gunakan `python` dari environment yang telah disiapkan. Jika environment belum diaktifkan, ganti `python` pada perintah berikut dengan `.\.venv\Scripts\python.exe`. Pipeline memakai PyTorch dan Pillow tanpa torchvision atau NumPy. GPU tidak diperlukan untuk pemeriksaan CPU yang tersedia. Instalasi environment baru belum menjadi bagian bukti pengujian.

## Penempatan dataset

**Dataset tidak disertakan dalam Git dan dibagikan secara terpisah.** Lokasi atau tautan distribusi final: [PERLU DILENGKAPI: bahan atau lokasi distribusi yang belum diserahkan]. Setelah menerima dataset, tempatkan seluruh paket pada struktur berikut:

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

`preproc.zip` merupakan arsip persiapan historis yang telah dihapus, bukan prasyarat penggunaan dataset final. `preprocessing/prepare_data.py` membutuhkan ZIP preprocessing dan ZIP Kaggle; `preprocessing/audit_archive.py` membutuhkan ZIP preprocessing. Keduanya opsional dan tidak menerima folder final sebagai pengganti ZIP. Audit serta hash sumber lama tetap dipertahankan.

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

Batch train adalah dictionary `A`/`B` dengan tensor RGB `float32 [N,3,256,256]`, finite, dalam `[-1,1]`. Augmentasi dilakukan saat loading di `preprocessing/data.py`: resize bicubic 286, crop acak 256, dan flip horizontal dengan peluang 0,5, diikuti normalisasi `2*(pixel/255)-1`. Gambar hasil augmentasi tidak disimpan sebagai sampel baru.

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

Training dapat dimulai dengan `Set-Location E:\midterm\cyclegan-midterm` lalu `python scripts\train_cyclegan.py --device auto --batch-size 1`. Default dataset sekarang `E:\midterm\data_cleaned_v2`. Folder ini tanpa manifest, sehingga loader memakai berkas dalam empat folder split secara langsung. Rincian loss, tuning, output, validasi dan resume ada di [panduan training lokal](docs/training_lokal.md); API dan kontrak tetap di [kontrak integrasi](docs/kontrak_integrasi.md).

## Pengujian dan hasil aktual

```powershell
python -m unittest discover -s tests -v
python -m preprocessing.check_data --root data/horse2zebra --output tmp/pemeriksaan_data_baru.json
```

Gunakan output baru agar bukti historis tidak tertimpa. Pemeriksaan data membutuhkan paket dataset final, sedangkan unit test memakai fixture sintetis dan tidak membutuhkan arsip sumber.

Bukti pemeriksaan 27 Oktober 2026 mencatat 21 tes lulus: tujuh tes model, tiga audit, dan sebelas data. Seluruh 2.608 gambar diperiksa mode, ukuran, hash file/piksel, serta kecocokannya dengan manifest. Semua loader diiterasi; batch nyata diterima empat model di CPU. Output generator adalah `[1,3,256,256]`, discriminator `[1,1,30,30]`, seluruhnya finite. Bukti: [log tes](docs/bukti/tes_dataset_final_tanpa_zip.txt), [pemeriksaan data/model](docs/bukti/pemeriksaan_dataset_final_tanpa_zip.json), dan [preservasi dataset](docs/bukti/preservasi_dataset_final_tanpa_zip.json). Hasil ini memverifikasi fondasi dan pipeline, bukan keberhasilan training.

Verifikasi penataan 3 November 2026 mengulang seluruh 21 tes, termasuk worker Windows, identitas re-export publik dan subprocess dari lokasi baru. Pemeriksaan dataset melalui `preprocessing.check_data` juga berhasil; hasil numerik data/loader/model sama dengan bukti sebelumnya. Keluaran baru disimpan terpisah di `tmp/preprocessing_2026-11-03/` (`tes_sesudah.txt`, `pemeriksaan_data.json`, `pemeriksaan_cli.txt`, `verifikasi_struktur.json`). Hash membuktikan keempat implementasi yang dipindahkan identik byte dan dataset/bukti historis tidak berubah. Tidak ada training yang dijalankan.

## Pembagian tugas dan kontribusi aktual

| Peran awal | Tanggung jawab | Penanggung jawab awal | Kontribusi aktual yang dilaporkan |
|---|---|---|---|
| 1 — Prompt Engineer & LLM Navigator | Rancangan, prompt, konsistensi kode, integrasi | Jian Hazel Sitorus (103012300187) | Menangani fondasi model dan koordinasi teknis; membantu prapemrosesan awal untuk sementara |
| 2 — Data Pipeline & Preprocessing Engineer | Dataset, preprocessing, augmentasi, normalisasi, DataLoader | Mikhael Restu Mahardhika (103012300277) | Bertanggung jawab melanjutkan prapemrosesan setelah bantuan sementara dari Jian Hazel Sitorus; rincian implementasi mengikuti bukti pekerjaan |
| 3 — Training Loop & Hyperparameter Tuning Engineer | Loss, optimizer, training, tuning, checkpoint, metrik | Reevanza Abel Desta Arifin (103012330104) | Menangani pelatihan dan penyetelan hiperparameter; skrip tersedia dari commit remote `5fe225b`, sedangkan artefak hasil run perlu dilengkapi |
| 4 — Lead Documentarian & Error Handler | Log interaksi, debugging, laporan | Dokumentasi awal: Jian Hazel Sitorus (103012300187); kelanjutan bersama anggota tim | [PERLU DILENGKAPI: nama penyunting dan bagian yang dikerjakan masing-masing] |

Nama dan NIM penanggung jawab Peran 1–3 diselaraskan dengan laporan Word pada 7 November 2026. Bantuan Jian Hazel Sitorus pada prapemrosesan awal bersifat sementara; kelanjutannya menjadi tanggung jawab Mikhael Restu Mahardhika. Dokumentasi Peran 4 dikerjakan bersama anggota tim. Nama berkas `status_orang2.md` dan label “orang 2” dalam bukti lama menunjukkan lingkup tugas. Rincian pelaksana setiap implementasi dan penyuntingan mengikuti bukti aktual dalam [koordinasi dokumentasi](docs/koordinasi_dokumentasi.md) dan [register debugging](docs/debugging.md).

## Keterbatasan dan pekerjaan berikutnya

Skrip training, validasi, checkpoint/resume dan evaluasi dari commit `5fe225b` dipertahankan pada integrasi 3 November 2026. Log run, checkpoint, grafik dan keluaran training belum tersedia sebagai bukti hasil. Kapasitas memori, tuning dan kualitas terjemahan belum diverifikasi. Inventaris terdahulu mencatat RTX 3060 Laptop 6 GiB; panduan anggota lain menyebut target RTX 4050 Laptop 6 GB, yang belum diperiksa pada sesi integrasi. Jangan memakai test untuk memilih hyperparameter atau checkpoint. [PERLU DILENGKAPI: konfigurasi run aktual dan bukti hasil dari Peran 3].

Test merupakan subset yang telah melalui penyaringan blur, bukan benchmark Kaggle penuh. Pemeriksaan piksel identik belum menjamin tidak ada gambar mirip atau crop berbeda. Tanpa ZIP preprocessing asli, audit dan persiapan ulang terhadap arsip tersebut tidak dapat diulang sepenuhnya. Distribusi dataset final, bukti kontribusi anggota, ekspor chat lengkap, dan refleksi kelompok masih perlu dilengkapi.

## Referensi dan dokumentasi pendukung

- Panduan tugas: *[CAK4KBB3 2026-1] Panduan Proyek 1 — Vibe Coding*. Identitas dokumen dan hash historis tersimpan dalam [inventaris](docs/bukti/lingkungan.json).
- Sumber dataset: [Horse2zebra Dataset, Balraj Ashwath](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset), versi 1 menurut [metadata yang diarsipkan](docs/bukti/kaggle_metadata.json). Tautan ini adalah sumber publik, bukan tautan paket dataset final kelompok.
- API dasar resmi: [PyTorch](https://docs.pytorch.org/docs/stable/index.html) dan [Pillow](https://pillow.readthedocs.io/en/stable/). Rincian penelusuran API melalui Context7 tersimpan dalam log.
- Kompilasi Peran 4: [paket dokumentasi dan checklist rubrik](docs/koordinasi_dokumentasi.md), [register debugging](docs/debugging.md). Laporan tetap draf karena identitas, ekspor dan hasil eksperimen belum lengkap.
- Atribusi LLM: [ringkasan percakapan GPT/ChatGPT](docs/diskusi_llm.md). Permintaan alat dalam prompt tidak membuktikan eksekusi; arsip atribusi bermasalah dipertahankan dan ditandai, bukan diubah menjadi respons GPT.
- Dokumen proyek: [laporan akademik](docs/laporan.md), [kontrak integrasi](docs/kontrak_integrasi.md), [status pipeline](docs/status_orang2.md), dan [log interaksi LLM](docs/log_interaksi.md).
