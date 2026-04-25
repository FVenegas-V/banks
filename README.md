# RAG Banco Central

Sistema de retrieval-augmented generation para documentos financieros de
banco central. Ingesta PDFs (Comunicados BCCh, Minutas, Reportes JPMorgan,
Fed Statements) y los sirve como corpus semántico con citación por página.

- **Embeddings**: `intfloat/multilingual-e5-small` (384-dim, multilingüe).
- **Storage**: PostgreSQL + pgvector con índice HNSW.
- **Retrieval**: híbrido — vector semántico (HNSW) + full-text (tsvector
  + ts_rank_cd), fusionados con RRF y diversificados con MMR.
- **Taxonomía**: 14 variables económicas, 10 secciones canónicas, entidades,
  patterns robustos a tildes y mayúsculas.

Para arquitectura detallada ver [`CLAUDE.md`](CLAUDE.md). Para instalar la
extensión de PostgreSQL ver [`SETUP_PGVECTOR.md`](SETUP_PGVECTOR.md).

## Pipeline

5 pasos, todos en Python puro (sin shell scripts):

```
Datos_prueba/*.pdf
       │
       ▼
[0] 00_generate_jsons.py     → logs/documents.json + logs/chunks.json
       │                       (chunking semántico por párrafos, overlap 100)
       ▼
[1] 01_enrich_metadata.py    → logs/chunks_enriched.json
       │                       (sección, variables, numerics, importance)
       ▼
[2] 02_vectorize.py          → logs/chunks_vectorized.json
       │                       (embeddings normalizados L2)
       ▼
[3] 03_database.py setup     → PostgreSQL schema + HNSW + GIN indices
    03_database.py load      → insert docs + chunks
       │
       ▼
[4] 04_search.py "<query>"   → top-k resultados híbridos
```

## Quick start

```bash
# 1. Dependencias Python
pip install -r requirements.txt

# 2. PostgreSQL + pgvector (ver SETUP_PGVECTOR.md)
#    Verifica que 'vector' aparezca en pg_available_extensions.

# 3. Pipeline completo
python3 run.py full

# 4. Búsqueda
python3 04_search.py "tasa de política monetaria 2022" 5
python3 04_search.py "riesgos inflación" 10 --json
```

## Comandos útiles

```bash
python3 run.py                      # menú interactivo
python3 run.py full                 # pipeline completo
python3 run.py step 0               # solo extracción
python3 run.py step 1               # solo enriquecimiento
python3 run.py step 2               # solo vectorización
python3 run.py step 3 setup         # crear schema
python3 run.py step 3 load          # cargar datos
python3 run.py step 3 reset         # drop + recrear + cargar
python3 run.py step 3 stats         # métricas de la BD
python3 run.py step 4 "<query>"     # búsqueda one-shot
python3 run.py search               # búsqueda interactiva
```

## Variables de entorno

| Variable                      | Default                              | Propósito                           |
|-------------------------------|--------------------------------------|-------------------------------------|
| `PGHOST` / `PGPORT` / `PGUSER` / `PGPASSWORD` | stack estándar       | Conexión PostgreSQL                 |
| `PGDATABASE`                  | `rag_banco`                          | Nombre de la BD                     |
| `RAG_EMBEDDING_MODEL`         | `intfloat/multilingual-e5-small`     | Modelo de embeddings                |
| `RAG_PURE_TEXT`               | `0`                                  | `1` = no prefijar metadata al texto |
| `SENTENCE_TRANSFORMERS_HOME`  | `~/.cache/...`                       | Caché de modelos HuggingFace        |

## Agregar nuevos PDFs

```bash
# Colocar en la carpeta que corresponde al tipo
cp nuevo_comunicado.pdf Datos_prueba/Comunicados/

# Re-ejecutar pipeline (idempotente)
python3 run.py full
```

El chunker respeta páginas (para citación) y preserva secciones cuando son
detectables. El enriquecimiento es robusto a tildes y mayúsculas.

## Desarrollo

Este repo tiene un gemelo en `/Users/leandrovenegas/Desktop/Proyecto_rag/`
(sandbox Mac). Mantener paridad entre ambos: cambios en uno se copian al
otro. El orquestador `run.py` es el mismo, y `taxonomy.py` es la fuente de
verdad para patrones (pasos 1 y 4 lo importan).
