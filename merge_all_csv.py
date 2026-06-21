import pandas as pd
import os

DATA_DIR    = r"C:\Users\Emriqurrizal\Downloads\symtomps"
OUTPUT_FILE = r"C:\Users\Emriqurrizal\Downloads\symtomps\merged_sympscan.csv"

print("[LOAD] Loading semua file CSV...\n")

df_main = pd.read_csv(os.path.join(DATA_DIR, "Diseases_and_Symptoms_dataset.csv"))
print(f"  OK  Diseases_and_Symptoms_dataset.csv  -> {len(df_main):>6} baris, {len(df_main.columns)} kolom")

df_main["diseases"] = df_main["diseases"].str.strip().str.lower()

df_desc = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
df_desc["Disease"] = df_desc["Disease"].str.strip().str.lower()
df_desc = df_desc.rename(columns={"Disease": "diseases", "Description": "description"})
print(f"  OK  description.csv                    -> {len(df_desc):>6} baris")

df_meds = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
df_meds["Disease"] = df_meds["Disease"].str.strip().str.lower()
df_meds = df_meds.rename(columns={"Disease": "diseases", "Medication": "medications"})
print(f"  OK  medications.csv                    -> {len(df_meds):>6} baris")

df_prec = pd.read_csv(os.path.join(DATA_DIR, "precautions.csv"))
df_prec["Disease"] = df_prec["Disease"].str.strip().str.lower()
# Gabungkan Precaution_1..4 menjadi 1 kolom list
df_prec["precautions"] = df_prec[["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]]\
    .apply(lambda row: str([x for x in row if pd.notna(x)]), axis=1)
df_prec = df_prec[["Disease", "precautions"]].rename(columns={"Disease": "diseases"})
print(f"  OK  precautions.csv                    -> {len(df_prec):>6} baris")

df_work = pd.read_csv(os.path.join(DATA_DIR, "workout.csv"))
df_work["Disease"] = df_work["Disease"].str.strip().str.lower()
df_work = df_work.rename(columns={"Disease": "diseases", "Workouts": "workouts"})
print(f"  OK  workout.csv                        -> {len(df_work):>6} baris")

df_diet = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))
df_diet["Disease"] = df_diet["Disease"].str.strip().str.lower()
df_diet = df_diet.rename(columns={"Disease": "diseases", "Diet": "diets"})
print(f"  OK  diets.csv                          -> {len(df_diet):>6} baris")

print("\n[MERGE] Merging semua file...")

df_merged = df_main.copy()
df_merged = df_merged.merge(df_desc, on="diseases", how="left")
df_merged = df_merged.merge(df_meds, on="diseases", how="left")
df_merged = df_merged.merge(df_prec, on="diseases", how="left")
df_merged = df_merged.merge(df_work, on="diseases", how="left")
df_merged = df_merged.merge(df_diet, on="diseases", how="left")

symptom_cols = [c for c in df_main.columns if c != "diseases"]
info_cols    = ["diseases", "description", "medications", "precautions", "workouts", "diets"]
final_cols   = info_cols + symptom_cols

df_merged = df_merged[final_cols]

df_merged.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*55}")
print(f"MERGE SELESAI!")
print(f"{'='*55}")
print(f"  Output file : {OUTPUT_FILE}")
print(f"  Total baris : {len(df_merged):,}")
print(f"  Total kolom : {len(df_merged.columns)} (6 info + 230 symptoms)")
print(f"\n  Struktur kolom:")
print(f"    - diseases       : nama penyakit")
print(f"    - description    : deskripsi penyakit")
print(f"    - medications    : daftar obat")
print(f"    - precautions    : tindakan pencegahan")
print(f"    - workouts       : rekomendasi olahraga")
print(f"    - diets          : rekomendasi diet")
print(f"    - [230 kolom]    : binary symptom indicators (0/1)")
print(f"\nPreview baris pertama:")
print(df_merged[info_cols].head(3).to_string())
