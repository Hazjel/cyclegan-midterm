# Kontrak integrasi fondasi CycleGAN

Kontrak teknis v1.1 untuk integrasi proyek kelompok. Ketentuan batch dan API berasal dari tahap implementasi pipeline tanggal 27 Oktober 2026; penyuntingan dokumentasi 28 Oktober 2026 tidak mengubah ketentuan tersebut. Loader evaluasi dipisah per domain. Arah domain tetap, sedangkan kapasitas arsitektur masih berupa baseline. Perubahan interface atau makna checkpoint perlu dicatat untuk koordinasi kelompok.

Root dataset aktif adalah **`data/horse2zebra`**. Dataset dibagikan terpisah dan tidak disertakan dalam Git. Paketnya harus mencakup trainA/trainB/testA/testB, `manifest.json`, `metadata_corrected.csv`, dan `preparation_report.json`; lokasi distribusi final [PERLU DILENGKAPI: bahan atau lokasi distribusi yang belum diserahkan]. Penghapusan `preproc.zip` dilaporkan pelaksana proyek dan dikonfirmasi pada sesi 27 Oktober 2026. Loader dan pemeriksaan folder final tidak membutuhkan ZIP.

## Penamaan dan kepemilikan

| Simbol | Makna |
|---|---|
| A | kuda |
| B | zebra |
| G_AB | mengubah kuda menjadi zebra |
| G_BA | mengubah zebra menjadi kuda |
| D_A | menilai real/fake dalam domain kuda |
| D_B | menilai real/fake dalam domain zebra |

Pembagian awal menempatkan arsitektur/API dan koordinasi pada Peran 1, pipeline pada Peran 2, pelatihan pada Peran 3, serta dokumentasi pada Peran 4. Kontribusi aktual yang diklarifikasi: pelaksana Peran 1 dan 4 juga mengambil alih pengerjaan teknis Peran 2. Nama/NIM pelaksana dan penanggung jawab awal tiap peran [PERLU DILENGKAPI: nama/NIM pelaksana dan penanggung jawab awal setiap peran]; kontribusi anggota lain yang belum dilaporkan tidak diasumsikan. Implementasi training, tuning dan checkpoint tetap menjadi tanggung jawab Peran 3; pada inventaris 31 Oktober 2026 belum ditemukan kode atau artefaknya di workspace. Peran 4 kini dikerjakan bersama beberapa anggota; identitas penyunting dan paket kerja perlu dicatat terpisah dari pelaksana kode dalam [koordinasi dokumentasi](koordinasi_dokumentasi.md). Rincian kontribusi dicatat dalam README dan laporan.

## Pipeline data → model dan pelatihan

Satu batch adalah dictionary dengan dua key wajib:

```python
batch = {"A": tensor_kuda, "B": tensor_zebra}
```

Struktur tersebut dihasilkan `create_dataloaders(root)["train"]` yang diimplementasikan pada `preprocessing/data.py` sejak penataan 3 November 2026. Ekspor publik dari `cyclegan` tetap berupa re-export langsung, tanpa wrapper. Folder `preprocessing/` adalah kode; `data/horse2zebra` adalah hasil dataset yang tetap. Metadata/path pada manifest bukan input model. Loader `testA` dan `testB` menghasilkan tensor domain tunggal `[N,3,256,256]` secara terpisah; semua sampel dikunjungi sekali dan batch terakhir tetap disertakan. Jangan menggabungkan loader test dengan `zip` atau menyamakan panjangnya dengan mengulang sampel.

Contoh integrasi untuk Peran 3 (dari root repository, `num_workers=0`):

```python
import torch
from cyclegan import create_dataloaders

torch.manual_seed(7)
loaders = create_dataloaders("data/horse2zebra", batch_size=1, num_workers=0, seed=7)
train_batch = next(iter(loaders["train"]))  # dictionary A/B
test_A = next(iter(loaders["testA"]))       # tensor kuda
test_B = next(iter(loaders["testB"]))       # tensor zebra
```

Loader membaca gambar dan `manifest.json` pada folder final secara langsung. Augmentasi saat loading, normalisasi dan sampling unpaired berada di `preprocessing/data.py`; gambar, manifest dan split tidak diubah. Peran 3 dapat langsung memakai pipeline tanpa persiapan arsip ulang. Jika menambah worker di Windows, bungkus pembuatan/iterasi loader dalam main guard sebagaimana contoh README.

- Setiap tensor mempunyai shape `[N,3,H,W]`, urutan NCHW, urutan kanal **RGB**, dtype `torch.float32`, nilai finite dalam **[-1,1]**. Batch tidak kosong.
- Kontrak data proyek sekarang: H=W=256. API model juga mendukung H/W >=32 dan kelipatan 4; bukan dukungan untuk ukuran arbitrer. Kedua tensor dalam batch proyek memakai N/H/W yang sama agar integrasi sederhana.
- Pasangan A[i] dan B[i] **tidak harus gambar objek yang sama**; CycleGAN memakai data unpaired. Panjang epoch = max(trainA,trainB) = 1.311. Domain lebih besar dikunjungi sekali dalam urutan shuffle; yang lebih kecil diambil acak dengan pengembalian sehingga tidak selalu ada awalan yang diulang. Batch terakhir tidak dibuang. Transform domain dilakukan independen.
- DataLoader memberikan tensor CPU. Kode pelatihan Peran 3 memindahkan tensor dan keempat model ke device yang sama. Tensor tidak perlu `requires_grad=True`; autograd menghitung gradient parameter secara normal.
- Setelah decoding, seluruh gambar dikonversi RGB, termasuk tujuh grayscale yang ditemukan. Normalisasi yang disepakati adalah `x = 2*(pixel_float/255)-1`; jangan menormalkan ulang output Tanh.
- Tidak memakai mean/std ImageNet. Hasil untuk tampilan dikembalikan dengan `(x+1)/2` lalu dibatasi [0,1] jika perlu, bukan untuk input training yang dinormalisasi ulang.
- Gunakan hanya dataset final `data/horse2zebra`, seluruhnya PNG RGB 256 x 256. Total final: trainA=1.040, trainB=1.311, testA=118, testB=139. Riwayat persiapan: `data_cleaned` dalam arsip dahulu menjadi kandidat; deduplikasi piksel RGB memprioritaskan test, sehingga `trainA/n02381460_3811.jpg` dikecualikan karena identik dengan testA. Sebanyak 28 blur dan 25 salinan duplikat dikecualikan, dicatat dalam `preparation_report.json`. Loading sekarang tidak mengulang proses tersebut.
- Pada sesi persiapan sebelumnya, seluruh gambar asli dibandingkan byte demi byte dengan unduhan Kaggle. CSV Kaggle juga memiliki 240 kesalahan path/domain yang sama; koreksi berdasarkan folder asli dan cek silang prefix nama disimpan di `metadata_corrected.csv`. Sumber tidak diubah saat persiapan; penghapusan `preproc.zip` dilaporkan setelah tahap tersebut. Manifest final SHA-256 `fe9d4e3a940fb9167fead81b21123ed4b12e09487e3a336a695f7d0681e38f2f`.
- Train memakai resize bicubic 286, random crop 256 (offset 0..30), horizontal flip 0,5. Test hanya resize bicubic 256 deterministik. Semua RNG sampling/augmentasi memakai torch. `seed` factory mengatur sampler/worker; panggil `torch.manual_seed(seed)` sebelum iterasi bila num_workers=0. Pada Windows, pembuatan/iterasi worker harus dilindungi main guard.
- Test terfilter oleh kebijakan blur awal, sehingga hasil kelak adalah evaluasi pada subset cleaned. Peran 3 harus mengambil validasi dari train sebelum tuning atau memilih checkpoint; jangan memakai test untuk keputusan tersebut. Deduplikasi belum mendeteksi semua crop/encoding berbeda.

Model memvalidasi rank, batch, kanal, dimensi, dtype, rentang, dan finite pada setiap forward. Tidak ada resize, convert, normalisasi, atau clipping diam-diam. Pemeriksaan nilai menggunakan `.item()` sehingga dapat menyinkronkan CPU/GPU. AMP dan dtype selain float32 belum menjadi kontrak; penambahan AMP pada Peran 3 memerlukan peninjauan validasi dan pengujian seluruh jalur cycle.

Pemeriksaan aktif: `python -m preprocessing.check_data --root data/horse2zebra --output tmp/pemeriksaan_data_baru.json`. Gunakan output baru agar bukti historis tidak tertimpa. Script membutuhkan folder final beserta manifest/preparation report, bukan ZIP. `preprocessing/prepare_data.py` dan `preprocessing/audit_archive.py` tetap merupakan utilitas opsional berbasis ZIP; tanpa `preproc.zip` asli, reproduksi persiapan/audit arsip lengkap belum dapat diulang. Hash/provenance dan bukti lama dipertahankan. Pemetaan lokasi lama ke baru tersedia pada README; kontrak batch dan arsitektur v1.1 tidak berubah.

## Pelatihan Peran 3 → model

```python
from dataclasses import asdict
from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator

config = ArchitectureConfig()
G_AB = Generator(config)
G_BA = Generator(config)
D_A = PatchDiscriminator(config)
D_B = PatchDiscriminator(config)
architecture_for_checkpoint = asdict(config)
```

Pemanggilan empat konstruktor ini membuat parameter baru yang terpisah. Konfigurasi adalah dataclass frozen, aman dipakai bersama. Jangan menyamakan referensi instance. `.to(device)`, `.train()`/`.eval()` serta pengaturan seed dimiliki caller; konstruktor tidak memilih GPU dan tidak mereset seed global.

Aliran yang perlu dipertahankan:

```text
fake_B = G_AB(real_A)       D_B menilai real_B dan fake_B
fake_A = G_BA(real_B)       D_A menilai real_A dan fake_A
reconstructed_A = G_BA(fake_B)
reconstructed_B = G_AB(fake_A)
```

Untuk pembaruan discriminator, pemisahan graph fake dengan `detach()` menjadi tanggung jawab kode pelatihan Peran 3. Pada pembaruan generator dan jalur cycle, gradient harus dapat melewati model yang dibutuhkan; jangan memutus graph tanpa alasan. Tidak ada detach di dalam forward model.

## Output dan pemilihan loss

Generator mengeluarkan float32 `[N,3,H,W]`, dibatasi Tanh ke [-1,1]. Discriminator mengeluarkan **peta skor mentah** `[N,1,floor(H/8)-2,floor(W/8)-2]`, menjadi `[N,1,30,30]` pada 256×256. Peta itu bukan satu label per gambar atau probabilitas 0–1.

Dengan kernel 4 dan stride 2/2/2/1/1, ukuran spasial berturut-turut 256→128→64→32→31→30; receptive field konvolusional nominal 4→10→22→46→70. Normalisasi InstanceNorm menggunakan statistik seluruh fitur spasial; nominal 70×70 mengacu pada jalur konvolusi.

Peran 3 menetapkan adversarial loss dan reduction secara eksplisit. Skor mentah cocok untuk least-squares berbasis MSE dengan target peta real/fake, atau `BCEWithLogitsLoss` bila memilih formulasi BCE. `BCEWithLogitsLoss` sudah menggabungkan sigmoid; jangan menambahkan sigmoid di model atau sebelum loss tersebut. `BCELoss` langsung pada skor mentah tidak sesuai. Target harus mempunyai shape sama dengan prediksi, misalnya dibuat dengan `ones_like`/`zeros_like`, bukan tensor `[N]` yang mengandalkan broadcasting. Cycle/identity loss beserta bobotnya dan pilihan perlu/tidaknya identity loss masih [PERLU DILENGKAPI: konfigurasi, bukti dan hasil aktual dari Peran 3].

## Konfigurasi dan checkpoint

Semua nilai `asdict(config)` wajib disimpan:

```json
{
  "architecture_version": "resnet_patchgan_v1",
  "channels": 3,
  "generator_filters": 64,
  "discriminator_filters": 64,
  "residual_blocks": 9,
  "norm_eps": 0.00001,
  "init_std": 0.02
}
```

`architecture_version` mewakili keputusan tetap dalam kode: kernel/stride/padding, dua down/up sampling, normalisasi non-affine tanpa running stats, bias convolution, aktivasi dan ketiadaan dropout/sigmoid. Jika keputusan tetap berubah, ubah versi dan dokumentasikan; jangan menganggap kesamaan shape state_dict menjamin kesamaan perilaku model. Nilai konfigurasi divalidasi saat konstruksi; versi yang tidak didukung ditolak.

Kontrak payload checkpoint yang perlu dibuat **Peran 3**, bukan fitur penyimpanan yang sudah selesai:

| Key wajib | Isi |
|---|---|
| `architecture` | `asdict(config)` aktual, termasuk versi |
| `model_state_dicts` | dictionary dengan empat key `G_AB`, `G_BA`, `D_A`, `D_B`, masing-masing state_dict sendiri |
| Metadata pelatihan | epoch/global step, konfigurasi loss/optimizer/scheduler, preprocessing dan manifest split, seed/RNG, versi paket/kode |
| State untuk resume | state optimizer/scheduler dan state lain yang dipakai, termasuk scaler bila kelak AMP disepakati |

Saat restore, buat `ArchitectureConfig(**checkpoint["architecture"])`, buat ulang empat instance dari konfigurasi tersebut, lalu muat masing-masing state_dict dengan `strict=True`. Inisialisasi konstruktor akan ditimpa bobot checkpoint; jangan melakukan inisialisasi ulang sesudah restore. Simpan format state_dict, bukan pickle objek model lengkap. Tes sesi ini hanya memverifikasi round-trip konfigurasi dan state_dict model dalam memori, **belum resume training**.

## Kriteria penerimaan handoff

Pipeline: satu batch nyata harus memenuhi seluruh kontrak di atas dan split final harus bebas duplikasi lintas train/test yang diketahui. Bukti pemeriksaan tersedia pada [hasil data/model](bukti/pemeriksaan_dataset_final_tanpa_zip.json). Peran 3 perlu memeriksa integrasi, satu langkah training nyata, penggunaan memori, checkpoint/resume, serta evaluasi terpisah. Pemeriksaan fondasi dan pipeline yang tersedia belum merupakan bukti langkah training.
