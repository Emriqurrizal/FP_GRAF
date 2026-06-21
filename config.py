#config buat ingest dataset ke neo4j (file 01 dan 02)
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "emriq123")   

DATA_DIR = r"C:\Users\Emriqurrizal\Downloads\symtomps"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
