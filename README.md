# papaya-cli

Aplikasi CLI berbasis Python 3.10+ untuk membangun dataset deteksi objek buah pepaya California (*Carica papaya*) pada 5 kelas tingkat kematangan:

- `unripe` (mentah / hijau)
- `early_ripening` (mengkal / mulai menguning)
- `ripe` (matang)
- `overripe` (terlalu matang)
- `rotten` (busuk)

## Fitur Utama

1. **Scraping gambar (`scrape`)**: Mendukung Bing Images & Google Images (via `icrawler`) serta API opsional Unsplash & Pexels, query multibahasa (Inggris + Indonesia) dari `config.yaml`, filter ukuran minimum, rate-limit delay, multi-thread, retry + exponential backoff, kepatuhan `robots.txt`, dan **auto-resume**.
2. **Pembersihan dataset (`clean`)**: Menghapus file rusak dan gambar di bawah ukuran minimum, deduplikasi SHA-256 & perceptual hash (pHash), filter zero-shot CLIP untuk memastikan gambar berisi buah pepaya, serta menyimpan `metadata.csv`.
3. **Auto-annotasi (`annotate`)**: Deteksi bounding box multi-objek menggunakan model open-vocabulary (**Grounding DINO** atau **OWL-ViT**) dengan prompt `"papaya"`, verifikasi kelas kematangan menggunakan **CLIP** (menandai ketidakcocokan atau tanpa deteksi sebagai `needs_review`), dukungan otomatis GPU CUDA / CPU, dan rendering gambar ke folder `preview/`.
4. **Ekspor dataset (`export`)**: Pembagian deterministik `train/val/test` (default `70/20/10`, seed tetap `42`) terstratifikasi per kelas ke format **YOLO** (`.txt` ternormalisasi + `classes.txt` + `data.yaml`), **COCO JSON**, atau **Pascal VOC XML**.

---

## Instalasi

Pastikan menggunakan **Python 3.10+**.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Jika menggunakan GPU NVIDIA, pasang `torch` sesuai versi CUDA Anda dari https://pytorch.org/get-started/locally/ — `papaya-cli` akan mendeteksi CUDA secara otomatis.

---

## Menjalankan dari Nol sampai Dataset Jadi

### Opsi 1: Jalankan seluruh pipeline sekaligus

```bash
papaya-cli run-all --config config.yaml
```

### Opsi 2: Jalankan tahap demi tahap

```bash
# 1. Scrape gambar (resume otomatis jika dihentikan dan dijalankan ulang)
papaya-cli scrape --config config.yaml --per-class 200

# 2. Bersihkan duplikat, file rusak, dan gambar non-pepaya -> data/clean/metadata.csv
papaya-cli clean --input data/raw

# 3. Auto-annotasi bounding box + verifikasi kematangan CLIP + simpan preview/
papaya-cli annotate --input data/clean --conf 0.35 --preview

# 4. Ekspor ke YOLO (atau --format coco / voc / all)
papaya-cli export --format yolo --split 70 20 10
```

Gunakan `--help` pada perintah utama maupun subcommand untuk melihat semua opsi:

```bash
papaya-cli --help
papaya-cli annotate --help
```

### Sumber Gambar & API Keys Opsional

Sumber tanpa API key (`bing`, `google`, `duckduckgo`, `wikimedia`, `inaturalist`) aktif langsung. Sumber berbasis API (`unsplash`, `pexels`, `pixabay`, `flickr`, `serper`) otomatis aktif jika key-nya diisi di `.env` (atau `config.yaml`) dan dilewati jika kosong:

```bash
export UNSPLASH_ACCESS_KEY="your_unsplash_key"
export PIXABAY_API_KEY="your_pixabay_key"
export FLICKR_API_KEY="your_flickr_key"
export SERPER_API_KEY="your_serper_key"
export PEXELS_API_KEY="your_pexels_key"
```

---

## Struktur Output

```text
data/
├── raw/                     # Gambar hasil scrape per kelas + sources.jsonl
├── clean/                   # Gambar bersih per kelas + metadata.csv
├── annotated/               # annotations.json + preview/<class>/*.jpg
└── export/
    ├── classes.txt
    ├── data.yaml
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

Setelah ekspor YOLO selesai, dataset siap dilatih misalnya dengan Ultralytics YOLO:

```bash
yolo detect train data=data/export/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

## Pengujian (Unit Tests)

```bash
pytest -q
# atau tanpa pytest:
python3 -m unittest discover -s tests -v
```

---

## Catatan Lisensi Gambar

Dataset yang dikumpulkan melalui web scraping **ditujukan untuk keperluan riset dan edukasi**. Hak cipta masing-masing gambar tetap dimiliki oleh pemegang hak ciptanya. Sebelum menggunakan gambar atau model hasil pelatihan untuk keperluan komersial, periksa lisensi tiap gambar sumber (tercatat di `data/clean/metadata.csv` pada kolom `source_url`) atau gunakan sumber berlisensi terbuka yang sesuai.

