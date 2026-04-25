# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Sistema RAG para banco central chileno. Procesa PDFs financieros (Comunicados
BCCh, Minutas del Consejo, reportes JPMorgan, Fed Statements) en un corpus
semántico consultable con citación por página.

Stack: `pypdf` + `sentence-transformers` (multilingual-e5-small, 384-dim) +
PostgreSQL + pgvector (HNSW) + búsqueda híbrida BM25/vector con RRF y MMR.

## Comandos esenciales

```bash
# Pipeline completo (extracción → enriquecimiento → embeddings → BD)
python3 run.py full

# Pasos individuales
python3 run.py step 0          # extracción PDF → chunks.json
python3 run.py step 1          # enriquecimiento → chunks_enriched.json
python3 run.py step 2          # embeddings → chunks_vectorized.json
python3 run.py step 3 reset    # drop + recrear schema PostgreSQL + cargar
python3 run.py step 3 stats    # métricas de la BD sin tocar nada

# Búsqueda
python3 04_search.py "tasa de interés 2022" 5
python3 04_search.py "commodities riesgos" 10 --json
python3 04_search.py "inflación" 5 --no-mmr

# Generar prompt RAG listo para LLM local (Ollama, llama.cpp)
python3 05_query.py "política monetaria 2022" 3

# Validación de que el sistema funciona correctamente
python3 04_search.py "política monetaria 2022" 3
# → top-1 debe ser comunicado1.pdf, sección DECISION, importance ≥ 0.90
```

## Arquitectura

### Flujo de datos

```
Datos_prueba/**/*.pdf
  → [00] chunks.json + documents.json        (extracción + chunking)
  → [01] chunks_enriched.json               (taxonomía semántica)
  → [02] chunks_vectorized.json             (embeddings normalizados)
  → [03] PostgreSQL rag_banco               (schema + HNSW + GIN)
  → [04] búsqueda híbrida                   (HNSW + BM25 → RRF → MMR)
  → [05] prompt para LLM                    (contexto delimitado + citas)
```

### Módulos y responsabilidades

| Archivo | Responsabilidad única |
|---|---|
| `models.py` | Dataclasses `Document`, `Chunk`, `EnrichedChunk` — única fuente de verdad de estructuras |
| `taxonomy.py` | Patterns de variables, secciones, entidades. Compartido por paso 1 y paso 4 |
| `00_generate_jsons.py` | PDF → chunks semánticos (por párrafos, target 600 chars, overlap 100) |
| `01_enrich_metadata.py` | Scoring de sección, variables económicas, importance_score |
| `02_vectorize.py` | Embeddings con E5-multilingual, prefijo contextual, normalización L2 |
| `03_database.py` | Schema PostgreSQL, HNSW index, bulk upsert, stats |
| `04_search.py` | Parse query NL, recall dual (vector+BM25), RRF, MMR, importance boost |
| `05_query.py` | Formatea contexto RAG + few-shot prompt para LLM local |
| `run.py` | Orquestador — delega a los scripts numerados vía subprocess |

### Invariantes críticos

- **`taxonomy.py` es la fuente de verdad**: si agregas una variable o sección,
  edita solo este archivo. El paso 1 y el parser del paso 4 la leen en tiempo
  de ejecución — no hay duplicación.

- **doc_type se hereda del filepath, nunca del contenido del chunk**: la función
  `detect_doc_type()` en `00_generate_jsons.py` opera sobre la ruta relativa del
  PDF. Un chunk de una minuta que menciona "comunicado" no se re-clasifica.

- **Embeddings normalizados L2 + `vector_cosine_ops`**: los embeddings se
  normalizan en `02_vectorize.py`. El índice HNSW usa `vector_cosine_ops`.
  Cambiar uno sin el otro rompe la similitud.

- **models_cache/ se auto-detecta**: `02_vectorize.py` y `04_search.py` setean
  `SENTENCE_TRANSFORMERS_HOME=./models_cache` automáticamente si la carpeta
  existe junto al script. En el servidor Windows sin internet, el modelo ya
  está en `models_cache/` y no requiere configuración adicional.

- **Filtro de variables incluye DECISION**: en `04_search.py`,
  `build_filters_sql()` siempre incluye chunks con `section_type='DECISION'`
  aunque no tengan la variable etiquetada, porque las decisiones de política
  monetaria son siempre relevantes para queries sobre variables críticas.

## Agregar nuevos tipos de documento

Dos cambios necesarios:

1. **`00_generate_jsons.py` → `detect_doc_type()`**: agregar rama `if` que
   reconozca la carpeta nueva (ej. `if "bce" in p: return "BCE_STATEMENT"`).

2. **`taxonomy.py` → `SECTION_KEYWORDS`**: agregar los patrones de texto
   característicos de los documentos del nuevo tipo para que el paso 1
   los clasifique correctamente (ej. secciones de comunicados del BCE).

Luego correr `python3 run.py full` — el pipeline es idempotente.

## Ajustar pesos de búsqueda

| Parámetro | Archivo | Qué controla |
|---|---|---|
| `IMPORTANCE_WEIGHTS` | `01_enrich_metadata.py` | Peso de cada señal en importance_score |
| `RECALL_N` | `04_search.py` | Candidatos por rama (vector + BM25) antes de RRF |
| `RRF_K` | `04_search.py` | Hiperparámetro de RRF (60 = estándar) |
| `MMR_LAMBDA` | `04_search.py` | 1.0 = solo relevancia, 0.0 = solo diversidad |
| `IMPORTANCE_BOOST` | `04_search.py` | Peso de importance en el re-rank final |

## Schema PostgreSQL (referencia rápida)

```sql
-- Dos tablas normalizadas
documents (document_id PK, doc_type_category, institution,
           document_date, document_year, extraction_warnings JSONB)

chunks    (chunk_id PK, document_id FK,
           text, text_tsv TSVECTOR,           -- BM25
           embedding VECTOR(384),             -- HNSW cosine
           section_type, section_confidence,
           economic_variables JSONB,          -- GIN jsonb_path_ops
           importance_score, is_policy_decision, is_forward_looking,
           tags TEXT[])                       -- GIN
```

## Variables de entorno

| Variable | Default | Cuándo cambiar |
|---|---|---|
| `PGDATABASE` | `rag_banco` | Usar otra BD |
| `PGUSER` / `PGPASSWORD` | SO / vacío | Servidor con auth |
| `RAG_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Cambiar modelo |
| `RAG_PURE_TEXT` | `0` | `1` = no inyectar metadata context en embedding |
| `SENTENCE_TRANSFORMERS_HOME` | auto-detectado | Solo si models_cache/ no está junto al script |
