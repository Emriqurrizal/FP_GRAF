#  MediGraph — Medical Knowledge Graph with Neo4j + LLM

> **Final Project — Graph Database (Semester 6)**

MediGraph membangun **Knowledge Graph** penyakit–gejala–obat–diet–olahraga–pencegahan menggunakan **Neo4j**, lalu mengintegrasikan **LLM (Large Language Model)** via OpenRouter untuk analisis dan tanya-jawab cerdas.

---

##  Arsitektur

```
┌─────────────────────────────────────────────────────────────────┐
│                        MediGraph AI                             │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Tier 1      │  Tier 2      │  Tier 3      │  Tier 4            │
│  Text-to-    │  ML on Graph │  LLM Graph   │  Graph-Augmented   │
│  Cypher +    │  (FastRP +   │  Builder     │  RAG               │
│  Analytics   │  KNN)        │              │                    │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                     Neo4j Graph Database                        │
│              (6 Node Types, 5 Relationship Types)               │
├─────────────────────────────────────────────────────────────────┤
│                    OpenRouter LLM API                           │
│                 (Moonshot Kimi K2.6 / Free)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Node Types & Relationships
```
(Disease) -[:HAS_SYMPTOM]-> (Symptom)
(Disease) -[:TREATED_WITH]-> (Medication)
(Disease) -[:HAS_PRECAUTION]-> (Precaution)
(Disease) -[:RECOMMENDED_WORKOUT]-> (Workout)
(Disease) -[:RECOMMENDED_DIET]-> (Diet)
```

---

##  Instalasi & Konfigurasi

### Prerequisites
1. **Neo4j 5.x** dengan **GDS (Graph Data Science) Plugin** aktif
   - Download: https://neo4j.com/download/
   - GDS Plugin: https://neo4j.com/docs/graph-data-science/current/installation/
2. **Google Colab** (untuk menjalankan notebook)
3. **OpenRouter API Key** (gratis): https://openrouter.ai/

### Setup Neo4j
1. Install & jalankan Neo4j Desktop atau Neo4j Community
2. Aktifkan GDS Plugin di database settings
3. Catat URI, username, dan password

### Setup API Key
1. Buat akun di [OpenRouter](https://openrouter.ai/)
2. Generate API key
3. Di Google Colab: **Settings  Secrets**  tambahkan `OPENROUTER_API_KEY`

---

##  Cara Run

### Langkah 1: Upload Notebook ke Colab
Upload file `FP_GRAF_MediGraph.ipynb` ke [Google Colab](https://colab.research.google.com/)

### Langkah 2: Setup Secrets
- Buka **Settings (️)  Secrets**
- Tambahkan `OPENROUTER_API_KEY` dengan API key dari OpenRouter

### Langkah 3: Konfigurasi Neo4j
Di **Cell 2**, ganti kredensial Neo4j:
```python
NEO4J_URI      = "neo4j://127.0.0.1:7687"  # atau neo4j+s://xxx.databases.neo4j.io
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password_here"        # ← GANTI INI
```

### Langkah 4: Run Semua Cell
Jalankan cell secara sequential (Ctrl+Enter atau Runtime  Run all):
1. **Cell 1** — Install dependencies
2. **Cell 2** — Setup API key & koneksi
3. **Cell 3** — Upload 6 file CSV (klik tombol "Choose Files")
4. **Cell 4–7** — Load data, setup constraints, ingest ke Neo4j
5. **Cell 8–15** — Semua analisis (Tier 1–4)

---

##  Dataset

| File | Deskripsi | Kolom |
|------|-----------|-------|
| `Diseases_and_Symptoms_dataset.csv` | Binary symptom indicators | `diseases` + 230 symptom columns |
| `description.csv` | Deskripsi penyakit | `Disease`, `Description` |
| `medications.csv` | Obat per penyakit | `Disease`, `Medication` |
| `precautions.csv` | Tindakan pencegahan | `Disease`, `Precaution_1..4` |
| `workout.csv` | Rekomendasi olahraga | `Disease`, `Workouts` |
| `diets.csv` | Rekomendasi diet | `Disease`, `Diet` |

---

##  Komponen & Tier

| Cell | Komponen | Tier | Deskripsi |
|------|----------|------|-----------|
| 8 | **LLM Text-to-Cypher** |  Tier 1 | Natural language  Cypher query  Neo4j |
| 9 | **PageRank** |  Tier 1 | Identifikasi node paling sentral |
| 10 | **Community Detection (Louvain)** |  Tier 1 | Clustering penyakit berdasarkan koneksi |
| 11 | **Shortest Path + Degree Centrality** |  Tier 1 | Path finding & analisis konektivitas |
| 12 | **FastRP Embeddings + KNN** |  Tier 2 | ML: node embeddings & similarity search |
| 13 | **LLM Graph Builder** |  Tier 3 | Ekstraksi entitas dari teks  populate graph |
| 14 | **Graph-Augmented RAG** |  Tier 4 | Retrieve + Augment + Generate |
| 15 | **Interactive Demo** | Demo | Full pipeline Q&A interaktif |

---

## ️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| **Database** | Neo4j 5.x + GDS Plugin |
| **LLM** | Moonshot Kimi K2.6 via OpenRouter (free tier) |
| **Language** | Python 3 (Jupyter Notebook / Google Colab) |
| **Libraries** | `neo4j`, `openai`, `pandas` |
| **Platform** | Google Colab |

---

##  Struktur File

```
FP_GRAF/
├── FP_GRAF_MediGraph.ipynb    ← Notebook utama (semua kode dalam 1 file)
├── README.md                  ← Dokumentasi ini
├── Dataset/
│   ├── Diseases_and_Symptoms_dataset.csv
│   ├── description.csv
│   ├── medications.csv
│   ├── precautions.csv
│   ├── workout.csv
│   └── diets.csv
├── config.py                  ← (legacy) konfigurasi versi script
├── 01_setup_constraints.py    ← (legacy) versi script terpisah
├── 02_ingest_data.py          ← (legacy) versi script terpisah
├── 03_llm_text_cleaner.py     ← (legacy) versi script terpisah
├── 04_graphrag_agent.py       ← (legacy) versi script terpisah
├── 05_gds_analytics.py        ← (legacy) versi script terpisah
└── merge_all_csv.py           ← (legacy) merge utility
```

> **Note:** File `.py` legacy disertakan sebagai referensi. Semua fungsionalitas sudah terintegrasi di `FP_GRAF_MediGraph.ipynb`.
