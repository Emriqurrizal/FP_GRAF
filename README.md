#  MediGraph — Knowledge Graph Medis dengan Neo4j + LLM

> **Final Project — Graf Pengetahuan**

MediGraph membangun sebuah **Knowledge Graph** yang menghubungkan penyakit, gejala, obat-obatan, diet, olahraga, dan tindakan pencegahan menggunakan **Neo4j**, lalu mengintegrasikan **Large Language Model (LLM)** melalui OpenRouter untuk analisis cerdas dan tanya-jawab berbasis graf.

---

##  Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────────┐
│                          MediGraph AI                            │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│   Tier 1     │   Tier 2     │   Tier 3     │   Tier 4            │
│  Text-to-    │  ML on Graph │  LLM Graph   │  Graph-Augmented    │
│  Cypher +    │  (FastRP +   │  Builder     │  RAG                │
│  GDS Algos   │  KNN)        │              │                     │
├──────────────┴──────────────┴──────────────┴─────────────────────┤
│                      Neo4j Graph Database                        │
│               (6 Tipe Node, 5 Tipe Relasi)                       │
├──────────────────────────────────────────────────────────────────┤
│                      OpenRouter LLM API                          │
│                  (Moonshot Kimi K2 / Gratis)                     │
└──────────────────────────────────────────────────────────────────┘
```

### Tipe Node & Relasi

```cypher
(Disease) -[:HAS_SYMPTOM]----------> (Symptom)
(Disease) -[:TREATED_WITH]----------> (Medication)
(Disease) -[:HAS_PRECAUTION]--------> (Precaution)
(Disease) -[:RECOMMENDED_WORKOUT]--> (Workout)
(Disease) -[:RECOMMENDED_DIET]------> (Diet)
```

---

##  Teknologi yang Digunakan

| Komponen       | Teknologi                                                              |
|----------------|------------------------------------------------------------------------|
| **Database**   | Neo4j 5.x + Plugin GDS (Graph Data Science)                          |
| **LLM**        | llama-3.3-70b-instruct via OpenRouter (tier gratis)                        |
| **Bahasa**     | Python 3.10+                                                           |
| **Library**    | `neo4j`, `openai`, `langchain`, `langchain-neo4j`, `pandas`, `python-dotenv` |
| **Notebook**   | Jupyter Notebook / Google Colab                                       |

---

##  Struktur Proyek

```
sympscan-knowledge-graph/
├── config.py                  ← Loader kredensial Neo4j & LLM dari file .env
├── 01_setup_constraints.py    ← Buat uniqueness constraint di semua node
├── 02_ingest_data.py          ← ETL: muat semua CSV → node & relasi di Neo4j
├── 03_text_to_cypher.py       ← Tier 1: Bahasa alami → Cypher → Neo4j
├── 04_gds_analytics.py        ← Tier 1: PageRank, Louvain, Shortest Path
├── 05_ml_on_graph.py          ← Tier 2: Embedding FastRP + similaritas KNN
├── 06_llm_graph_builder.py    ← Tier 3: Ekstraksi entitas LLM → Graf
├── 07_graph_augmented_rag.py  ← Tier 4: Retrieve → Augment → Generate
├── merge_all_csv.py           ← Utilitas preprocessing & penggabungan dataset
├── FP_GRAF_MediGraph.ipynb    ← Notebook utama (semua tier terintegrasi)
├── Dataset/
│   ├── Diseases_and_Symptoms_dataset.csv   ← 230 kolom gejala biner
│   ├── description.csv                     ← Deskripsi tiap penyakit
│   ├── medications.csv                     ← Rekomendasi obat
│   ├── precautions.csv                     ← Tindakan pencegahan
│   ├── workout.csv                         ← Rekomendasi olahraga
│   └── diets.csv                           ← Rekomendasi diet
├── .env.example               ← Template kredensial (salin ke .env)
└── README.md
```

---

##  Referensi Modul

| File | Tier | Fungsi Utama | Fungsi Kunci |
|------|------|-------------|--------------|
| `config.py` | — | Memuat variabel lingkungan untuk Neo4j & OpenRouter | — |
| `01_setup_constraints.py` | Setup | Membuat uniqueness constraint pada 6 tipe node | `setup_constraints()` |
| `02_ingest_data.py` | Setup | Memuat 6 file CSV → node & relasi ke Neo4j | `ingest_diseases_symptoms()`, `ingest_medications()`, `run_all()` |
| `03_text_to_cypher.py` | Tier 1 | LLM + LangChain terjemahkan bahasa alami → Cypher → eksekusi | `get_graph_schema()`, `text_to_cypher()` |
| `04_gds_analytics.py` | Tier 1 | Algoritma GDS: PageRank, Louvain, Shortest Path, Degree Centrality | `setup_gds_graph()`, `run_pagerank()`, `run_community_detection()`, `run_shortest_path_and_degree()` |
| `05_ml_on_graph.py` | Tier 2 | Embedding FastRP 64-dim + KNN + similaritas gejala bersama | `run_fastrp_embeddings()`, `run_knn_similarity()`, `run_shared_symptoms_similarity()` |
| `06_llm_graph_builder.py` | Tier 3 | LLM ekstrak entitas dari teks klinis → MERGE ke dalam graf | `extract_entities_from_text()`, `populate_graph_from_extraction()` |
| `07_graph_augmented_rag.py` | Tier 4 | `GraphCypherQAChain` RAG: generate Cypher + sintesis jawaban | `ask_medigraph()` |
| `merge_all_csv.py` | Utilitas | Left-join 6 CSV → `merged_sympscan.csv` (236 kolom) | — |

---

##  Dataset

| File | Deskripsi | Kolom Utama |
|------|-----------|-------------|
| `Diseases_and_Symptoms_dataset.csv` | Indikator gejala biner per penyakit | `diseases` + 230 kolom gejala |
| `description.csv` | Deskripsi medis singkat per penyakit | `Disease`, `Description` |
| `medications.csv` | Rekomendasi obat/pengobatan | `Disease`, `Medication` |
| `precautions.csv` | Tindakan pencegahan | `Disease`, `Precaution_1..4` |
| `workout.csv` | Rekomendasi olahraga | `Disease`, `Workouts` |
| `diets.csv` | Rekomendasi diet | `Disease`, `Diet` |

---

##  Instalasi & Konfigurasi

### Prasyarat
1. **Python 3.10+** dengan pip
2. **Neo4j 5.x** dengan **Plugin GDS** aktif
   - Unduh: https://neo4j.com/download/
   - Plugin GDS: https://neo4j.com/docs/graph-data-science/current/installation/
3. **API Key OpenRouter** (gratis): https://openrouter.ai/

### 1. Clone & Install Dependensi

```bash
git clone <repo-url>
cd sympscan-knowledge-graph
pip install neo4j openai langchain langchain-neo4j pandas python-dotenv
```

### 2. Konfigurasi Variabel Lingkungan

Salin `.env.example` ke `.env` dan isi kredensial Anda:

```bash
cp .env.example .env
```

```env
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password_anda_di_sini
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
MODEL_NAME=moonshotai/kimi-k2:free
```

### 3. Konfigurasi Neo4j

1. Jalankan Neo4j Desktop atau Neo4j Community Server
2. Aktifkan Plugin GDS di pengaturan database
3. Pastikan database berjalan di `bolt://localhost:7687`

---

##  Menjalankan Pipeline

Jalankan skrip secara berurutan:

```bash
# (Opsional) Pra-proses: gabungkan semua CSV → merged_sympscan.csv (236 kolom)
python merge_all_csv.py

# Langkah 1: Bootstrap Neo4j — buat uniqueness constraint
python 01_setup_constraints.py

# Langkah 2: ETL — muat 6 CSV ke dalam knowledge graph
python 02_ingest_data.py

# Tier 1 — Text-to-Cypher (LangChain + Kimi K2)
python 03_text_to_cypher.py

# Tier 1 — Analitik Graf GDS (membutuhkan plugin GDS)
python 04_gds_analytics.py

# Tier 2 — ML: FastRP (64-dim) + KNN (jalankan setelah 04)
python 05_ml_on_graph.py

# Tier 3 — LLM Graph Builder dari teks klinis bebas
python 06_llm_graph_builder.py

# Tier 4 — RAG Interaktif (REPL, ketik 'quit' atau 'exit' untuk berhenti)
python 07_graph_augmented_rag.py
```

> **Alternatif:** Jalankan `FP_GRAF_MediGraph.ipynb` di Google Colab untuk pengalaman all-in-one dengan semua tier terintegrasi dalam satu notebook.

---

##  Google Colab — Cara Cepat

1. Upload `FP_GRAF_MediGraph.ipynb` ke [Google Colab](https://colab.research.google.com/)
2. Buka **Pengaturan → Secrets**, tambahkan:
   - `OPENROUTER_API_KEY` — API key dari OpenRouter
3. Di **Cell 2**, konfigurasikan kredensial Neo4j Anda
4. Jalankan semua cell secara berurutan (`Runtime → Run all`)

---

##  Ringkasan Tier

| Tier | Skrip | Fitur | Deskripsi |
|------|-------|-------|-----------|
| Setup | `01` + `02` | Ingesti Data | CSV → node & relasi di Neo4j |
| **Tier 1** | `03_text_to_cypher.py` | Text-to-Cypher | Bahasa alami → Cypher → hasil query |
| **Tier 1** | `04_gds_analytics.py` | Analitik Graf | PageRank, Louvain, Degree Centrality, Shortest Path |
| **Tier 2** | `05_ml_on_graph.py` | ML pada Graf | Embedding FastRP + pencarian similaritas KNN |
| **Tier 3** | `06_llm_graph_builder.py` | Pembangun Graf | Ekstraksi entitas LLM → populasi graf secara langsung |
| **Tier 4** | `07_graph_augmented_rag.py` | Graph RAG | Ambil konteks graf → augmentasi prompt → jawaban LLM |

---

> **Catatan:** Skrip `.py` modular dan notebook `FP_GRAF_MediGraph.ipynb` mengimplementasikan pipeline yang sama. Gunakan skrip untuk pengembangan lokal dan notebook untuk demonstrasi di Google Colab.

---

## Penggunaan AI & Modifikasi Manual

Pengembangan proyek dan basis kode ini sangat dibantu oleh **Claude Sonnet 4.6 (Thinking Mode)** sebagai asisten *coding* AI utama.

**Peran AI dalam Proyek:**
- Merancang fondasi kode untuk arsitektur 4-Tier (Text-to-Cypher, GDS, ML on Graph, LLM Builder, dan Graph RAG).
- Melakukan refaktorisasi file notebook besar menjadi skrip Python modular (`01` s/d `07`).
- Menulis draf *query* Cypher dasar dan algoritma analisis Neo4j.
- Menyusun dokumentasi, penjelasan struktur direktori, serta narasi presentasi (README, draf metodologi).

**Modifikasi & Penyesuaian Manual:**
Meski kode dasar di-*generate* oleh AI, kami tetap melakukan modifikasi dan penyetelan manual yang esensial agar sistem dapat berjalan optimal:
1. **Penyempurnaan Prompt (Prompt Engineering):** Memodifikasi template *prompt* LangChain pada tahap *Text-to-Cypher* dan RAG secara manual untuk memastikan model (khususnya model OpenRouter *free-tier*) merespons dengan struktur Cypher dan JSON yang valid tanpa berhalusinasi.
2. **Koreksi Relasi & Algoritma GDS:** Memperbaiki arah panah relasi Neo4j pada kode AI yang terkadang terbalik, serta menyetel ulang konfigurasi parameter *node/relationship projection* di fungsi FastRP, KNN, dan PageRank.
3. **Standarisasi Dataset & Lokalisasi:** Melakukan pra-pemrosesan data secara manual menggunakan Pandas (misal: `merge_all_csv.py`), serta menerjemahkan/menyesuaikan istilah medis agar seragam antara format pertanyaan (Inggris/Indonesia) dan isi graf.
4. **Integrasi & Error Handling:** Menambahkan mekanisme *try-except* berlapis, mengelola muatan kredensial `.env` yang aman, dan menyederhanakan *output* panjang di terminal menjadi ringkas agar nyaman dibaca.
