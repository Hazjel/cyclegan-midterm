# Training dan tuning lokal CycleGAN

`scripts/train_cyclegan.py` memakai package lokal `cyclegan/` dan API model serta loader yang sudah ada. Tidak ada pretrained weights. Dataset aktif ditempatkan pada `data/horse2zebra` relatif terhadap root repository. Default kode menunjuk `REPO_ROOT.parent / "data" / "horse2zebra"`, sehingga seluruh perintah di bawah memberikan `--dataset-root data/horse2zebra` secara eksplisit. Perilaku skrip dari commit anggota lain `5fe225b` dipertahankan pada integrasi 3 November 2026. Script hanya membaca manifest/gambar melalui loader; ia tidak menulis ke dataset.

## Setup dan mulai menjalankan

Gunakan environment dengan dependensi `requirements.txt` dan PyTorch yang mendukung CUDA pada driver setempat. Perintah dari PowerShell:

```powershell
# Jalankan dari root repository
python -m pip install -r requirements.txt
python scripts\train_cyclegan.py --dataset-root data/horse2zebra --device auto --epochs 100 --batch-size 1 --lr 0.0002 --seed 7 --num-workers 0
```

`auto` memilih CUDA bila PyTorch mendeteksinya, selain itu CPU. GPU target RTX 4050 Laptop 6 GB memakai float32 dan batch awal 1. Kesesuaian memori harus dipastikan saat dijalankan; OOM memberi saran untuk tetap/mengurangi ke batch 1 atau memilih CPU. Jika import gagal, periksa `Get-Location` (harus root repo) dan keberadaan `cyclegan\__init__.py`; script juga menambahkan root repo berdasarkan lokasi file script.

Argumen utama: `--dataset-root`, `--output-root`, `--epochs`, `--batch-size`, `--lr`, `--seed`, `--device`, `--resume`, `--num-workers`, `--lambda-cycle`, `--lambda-identity`, `--identity/--no-identity`, `--val-fraction`, `--checkpoint-every`. Script memberi setiap run folder waktu/PID baru di `runs\`; tidak menimpa bukti run lama. `--limit-train-batches` hanya untuk debugging alur dan bukan epoch atau hasil eksperimen penuh.

Pada pemeriksaan integrasi 3 November 2026, `--help` yang ditangkap melalui pipe Windows gagal mencetak simbol panah pada encoding `cp1252`. Pemanggilan `python -X utf8 scripts/train_cyclegan.py --help` dan `python -X utf8 scripts/evaluate_cyclegan.py --help` berhasil. Gunakan opsi `-X utf8` bila mengalihkan keluaran pada lingkungan serupa; kode training tidak diubah untuk penyesuaian ini.

## Objective dan update

Adversarial memakai LSGAN dengan `MSELoss` atas peta skor mentah PatchDiscriminator. Target real dan fake dibuat tepat sebesar keluaran dengan `ones_like` dan `zeros_like`. Tiap generator menjumlahkan MSE real-target untuk fake_B di D_B dan fake_A di D_A. Cycle loss adalah `L1(G_BA(G_AB(A)), A) + L1(G_AB(G_BA(B)), B)`, bobot awal λ_cycle=10. Identity aktif secara default: `L1(G_BA(A),A) + L1(G_AB(B),B)`, bobot λ_identity=5. Total generator `L_adv + λ_cycle L_cycle + λ_identity L_identity`. Discriminator domain masing-masing memakai setengah jumlah MSE(real, ones) dan MSE(fake.detach(), zeros); objective D total menjumlahkan D_A dan D_B. Nilai bobot serta identity dapat diubah lewat CLI.

Optimizer awal Adam dengan lr=2e-4, betas=(0.5,0.999), lazim sebagai baseline GAN dan konsisten untuk dua generator bersama dan dua discriminator bersama. Secara default scheduler tidak digunakan. Fine-tuning dapat mengaktifkan linear decay melalui `--lr-decay-start` dan `--lr-decay-end`; LR sama dengan nilai dasar pada epoch start, lalu turun linear hingga 0 pada epoch end. Semua jalur float32. Saat update G, parameter discriminator dibekukan tetapi forward discriminator tetap mengalirkan gradient ke fake/generator; tidak ada pemutusan pada cycle. Saat update D, fake dilepas memakai `detach()`.

## Validasi, test, dan tuning

Sebelum epoch dibuat holdout deterministik (seed+1) per domain dari objek dataset trainA/trainB yang dibuat oleh loader proyek. Default 5% tiap domain menjadi validasi; gambar holdout tidak digunakan oleh train loader. Loader validasi memakai resize deterministik, tanpa crop/flip acak, dan memasangkan sampel hanya di dalam subsetnya. Empat pasangan pertama holdout dikunci; `validation_samples.json` mencatat path relatifnya. Preview disimpan setiap lima epoch (serta epoch pertama resume dan epoch akhir) di `validation_samples/`; setiap citra dijalankan sendiri sebagai batch 1 pada 256×256. Karena pasangan tidak punya makna semantik, skor validasi adalah monitoring loss gabungan, bukan metrik kualitas terjemahan. Checkpoint terbaik meminimalkan total generator validation loss. Saat membandingkan konfigurasi, tambahkan `--skip-test`; testA/testB tidak boleh dipakai untuk memilih bobot/parameter/checkpoint. Jalankan evaluasi test sekali pada konfigurasi final tanpa opsi tersebut. Test domain dibaca mandiri tanpa `zip`. `test_metrics.json` menyimpan jumlah gambar dan MAE rekonstruksi siklus per domain. MAE ini hanya deskriptif dan bukan skor kualitas terjemahan. Gambar contoh menampilkan sampai empat baris real, translated, reconstructed; konversi [-1,1] ke [0,1] hanya untuk tampilan.

Untuk membandingkan konfigurasi, jalankan tiap eksperimen baru dari awal dengan seed dan prosedur sama serta parameter eksplisit. `--log-every-batches 100` (default) mencetak progres di tengah epoch; gunakan 0 untuk menonaktifkan. Contoh tuning (test dilewati):

```powershell
python scripts\train_cyclegan.py --dataset-root data/horse2zebra --epochs 100 --batch-size 1 --lr 0.0002 --seed 7 --skip-test --output-root runs
python scripts\train_cyclegan.py --dataset-root data/horse2zebra --epochs 100 --batch-size 1 --lr 0.0001 --seed 7 --skip-test --output-root runs
```

Folder output berisi `config.json`, `metrics.csv` (loss train/validasi, lr, waktu, epoch/global step), `best.pt`, checkpoint `epoch-NNNN.pt`, `test_metrics.json`, gambar contoh, dan `training_complete.json`. Checkpoint menyimpan architecture `asdict`, empat state_dict terpisah, optimizer, epoch/global step, loss/optimizer/scheduler, seed, RNG Python/PyTorch/CUDA, hash manifest, versi PyTorch/Python dan revision Git. Untuk melanjutkan, berikan file checkpoint melalui `--resume`; `--epochs` adalah target total epoch, mis. checkpoint epoch 20 dilanjutkan dengan `--epochs 100`.

Untuk melanjutkan checkpoint epoch 50 dengan LR 0.0002 yang turun linear pada epoch 51–100:

```powershell
python scripts\train_cyclegan.py --dataset-root data/horse2zebra --device cuda --batch-size 1 --num-workers 0 --epochs 100 --lr 0.0002 --seed 7 --skip-test --lr-decay-start 51 --lr-decay-end 100 --resume runs\cyclegan-20261031-114034-14384\epoch-0050.pt
```

LR yang benar-benar dipakai dicatat di setiap baris `metrics.csv`; informasi jadwal dan state-nya ikut disimpan dalam checkpoint. Script hanya mengizinkan penambahan jadwal saat resume tepat pada epoch batasnya dan setelahnya mengharuskan konfigurasi scheduler tetap sama.

Untuk tuning, lewati test dengan `--skip-test`. Setelah memilih konfigurasi berdasarkan validasi, evaluasi checkpoint terbaik satu kali pada test tanpa training ulang:

```powershell
python scripts\evaluate_cyclegan.py --dataset-root data/horse2zebra --checkpoint runs\NAMA_RUN\best.pt --device cuda --batch-size 1 --num-workers 0
```

Script evaluasi membuat folder baru di `runs\evaluations\` berisi metrik testA/testB dan sampai empat contoh translasi per arah. Jangan memakai angka test untuk memilih konfigurasi.

Default batch 1 konservatif untuk GPU laptop 6 GB, tetapi pemakaian aktual dipengaruhi driver dan aplikasi lain. Tidak dijanjikan waktu, konvergensi, atau hasil tuning. Belum ada metrik/checkpoint training yang dihasilkan dalam repository ini.
