# CLAUDE.md

Guía para Claude Code cuando trabaje en este repositorio.

## Overview

Sistema RAG (Retrieval-Augmented Generation) para banco central. Ingesta PDFs
financieros (Comunicados BCCh, Minutas, Reportes JPM, Fed Statements) y los
sirve como corpus semántico auditable con citación por página.

- **Dataset actual**: 8 PDFs → 361 chunks con embeddings de 384 dimensiones.
- **Storage**: PostgreSQL con extensión **pgvector** (HNSW + cosine).
- **Retrieval**: híbrido (vector semántico + BM25 vía tsvector) con RRF y MMR.
- **Enriquecimiento**: taxonomía compartida de 14 variables económicas, 10
  secciones canónicas, detección de entidades y datos numéricos.

## Hermano

Existe un directorio gemelo **`/Users/leandrovenegas/Desktop/Proyecto_rag/`**
que se usa como sandbox en el Mac (testing). Este repositorio
(`Proyecto_RAG_Windows/`) es la copia que se despliega al servidor Windows.
Mantener ambos en paridad — cualquier cambio se copia al otro.

## Pipeline (5 pasos, todos .py)

| Paso | Script                  | Entrada                    | Salida                       |
|------|-------------------------|----------------------------|------------------------------|
| 0    | `00_generate_jsons.py`  | `Datos_prueba/**/*.pdf`    | `logs/documents.json`, `logs/chunks.json`, `logs/extraction_report.json` |
| 1    | `01_enrich_metadata.py` | `logs/chunks.json` + `logs/documents.json` | `logs/chunks_enriched.json` |
| 2    | `02_vectorize.py`       | `logs/chunks_enriched.json`| `logs/chunks_vectorized.json`, `logs/vectorization_report.json` |
| 3    | `03_database.py`        | JSONs de pasos 0 y 2       | PostgreSQL `rag_banco`       |
| 4    | `04_search.py`          | BD + query en NL           | Resultados top-k             |

Orquestador: `run.py` (menú interactivo o subcomandos). Ejecutable de un shot:

```bash
python3 run.py full
python3 run.py step 4 "tasa de interés 2022" 5
```

## Arquitectura de cada paso

### Paso 0 — Extracción y chunking semántico

- Usa `pypdf` (fallback a PyPDF2 si hace falta).
- Detecta PDFs protegidos (Azure IP, encrypted) y los reporta como warning sin
  abortar el pipeline.
- **Chunker**: agrupa párrafos (split por `\n\s*\n+`) hasta un tamaño objetivo
  de ~600 chars, con overlap de 100 chars entre chunks consecutivos. Divide
  párrafos >1200 chars por oraciones.
- Preserva `page_start` / `page_end` en cada chunk para citación auditable.
- Normaliza texto: une guiones de corte de línea, colapsa whitespace, elimina
  page-numbers huérfanos.
- Schema de salida **separa** documentos y chunks (no todo en un JSON plano):
  `documents.json` con metadata del PDF, `chunks.json` con chunks crudos.

Configuración (constantes al inicio del archivo): `TARGET_CHUNK_CHARS`,
`MIN_CHUNK_CHARS`, `MAX_CHUNK_CHARS`, `OVERLAP_CHARS`.

### Paso 1 — Enriquecimiento

- Importa patrones de `taxonomy.py` (módulo compartido con paso 4).
- **doc_type_category e institution se HEREDAN** del documento (paso 0, por
  filepath). No se re-detectan por chunk.
- Detección de **sección canónica** (10 tipos: ENCABEZADO, RESUMEN, DECISION,
  VOTACION, CONTEXTO_EXTERNO/INTERNO, ANALISIS, PROYECCION, RIESGOS, DATOS,
  CONTENIDO) vía scoring ponderado, no "primer match". Devuelve confidence.
- Detección de **14 variables económicas** con niveles CRITICAL/HIGH/MEDIUM.
  Patterns case/accent-insensitive con word-boundaries.
- Extracción de **numeric_values** (porcentajes, puntos base, millones, USD).
- Detección de **entidades** (bancos centrales, países).
- Cálculo de `importance_score` calibrado (ver `IMPORTANCE_WEIGHTS` en 01).
  Un chunk tipo "decisión de política con variables críticas + datos" llega
  a ~0.9; chunks narrativos sin variables se quedan cerca de 0.

### Paso 2 — Vectorización

- **Modelo por defecto**: `intfloat/multilingual-e5-small` (384 dim, 512
  tokens, multilingüe).
- Fallback: `all-MiniLM-L6-v2` si el multilingüe no puede cargarse.
- Override: `RAG_EMBEDDING_MODEL=<hf-id>`.
- Embeddings **normalizados L2** → cosine = dot product.
- Texto embebido incluye un prefijo corto de contexto
  `[DOC_TYPE | SECTION | top_vars]` para inyectar señal del enriquecimiento
  al vector (desactivable con `RAG_PURE_TEXT=1`).
- Para modelos E5 se añade el prefijo requerido `passage: ` en documentos
  (y `query: ` en queries, ver paso 4).
- Reporta truncamiento real usando el tokenizer del modelo, no heurística.

### Paso 3 — PostgreSQL + pgvector

Schema **normalizado**:

```
documents (document_id PK, filename, filepath, doc_type_category,
           institution, document_date, document_year, total_pages,
           total_chunks, char_count, extraction_warnings JSONB)

chunks (chunk_id PK, document_id FK → documents ON DELETE CASCADE,
        text, text_tsv TSVECTOR, page_start, page_end,
        section_type, section_confidence,
        economic_variables JSONB, numeric_values JSONB, entities JSONB,
        temporal_refs JSONB, tags TEXT[],
        importance_score, is_policy_decision, is_forward_looking,
        embedding VECTOR(384), embedding_model)
```

Índices:

- **HNSW** `vector_cosine_ops` sobre `embedding` (m=16, ef_construction=64).
- **GIN** sobre `text_tsv` (config `simple` para multilingüe), `tags`,
  `economic_variables` (jsonb_path_ops), `entities`.
- BTree sobre `importance_score DESC`, `section_type`, `document_id`,
  `doc_type_category`, `institution`, `document_year`.
- Índices parciales sobre `is_policy_decision=TRUE` y `is_forward_looking=TRUE`.

Subcomandos: `setup`, `load`, `reset` (drop+setup+load), `stats`.

Conexión por variables de entorno: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`,
`PGDATABASE` (default `rag_banco`).

### Paso 4 — Búsqueda híbrida

Pipeline de retrieval:

1. **Parse query** (lenguaje natural) → extrae filtros: año/rango, doc_type,
   variables económicas, secciones. Usa la misma `taxonomy.py` que el paso 1.
2. **Vector recall** (`RECALL_N=50`) vía HNSW con filtros pushdown en SQL.
3. **Lexical recall** (`RECALL_N=50`) vía `ts_rank_cd` sobre `text_tsv` con
   los mismos filtros.
4. **RRF fusion** (`k=60`) combina ambos rankings sin calibrar pesos.
5. **MMR** (`λ=0.65`) para diversidad final (opcional con `--no-mmr`).
6. **Importance boost** (`weight=0.15`) como tie-breaker suave.

Output: texto legible o JSON (`--json`). Incluye citas por página.

Si los filtros estrictos (variables/secciones) dejan el resultset vacío, el
pipeline automáticamente relaja esos filtros y reintenta.

## Taxonomía compartida (`taxonomy.py`)

Módulo usado por pasos 1 y 4. Contiene:

- `ECONOMIC_VARIABLES` — dict `{nombre: (keywords, importance)}`.
- `SECTION_KEYWORDS` — dict `{sección: [keywords]}`.
- `ENTITY_KEYWORDS` — dict `{entidad: [keywords]}`.
- `NUMERIC_PATTERNS` — lista de `(regex, unit_label)`.
- `normalize_text()` — lowercase + sin tildes + whitespace colapsado. Todos
  los patterns operan sobre texto normalizado para ser robustos a variaciones.
- `compile_word_pattern(words)` — compila alternación con lookarounds de
  word-boundary compatibles con Unicode.
- `derive_tags()` — tags derivados (DECISION_POLITICA, FORWARD_LOOKING,
  DATOS_NUMERICOS, VARIABLE_CRITICA, VOTACION).

**Para agregar una variable económica** o una sección nueva: editar
`taxonomy.py` y re-ejecutar pasos 1 (enrich), 3 (reload). No hace falta
re-vectorizar salvo que se agregue metadata context nuevo.

## Agregar nuevos PDFs

1. Copiar el PDF a `Datos_prueba/<TipoDocumento>/<filename>.pdf`. La carpeta
   de primer nivel determina `doc_type_category` (ver `detect_doc_type` en 00):
   `Comunicados/`, `Minutas/`, `Fed/`, `Researchs/`.
2. Ejecutar `python3 run.py full`.

## Decisiones de diseño importantes

1. **Separación documents vs chunks**: evita redundancia de metadata por
   chunk (fechas, tipos, institución) y permite queries `JOIN` eficientes.

2. **Heredar doc_type del filepath**: es más determinista y menos propenso
   a error que re-detectarlo por contenido del chunk. Si un chunk de una
   minuta referencia "comunicado", no queremos que se re-clasifique.

3. **RRF sobre weighted sum**: las escalas de cosine similarity (0-1) y BM25
   (no acotado) no son comparables directamente. RRF rankea por posición
   y funde sin calibrar.

4. **Importance como re-rank, no filtro**: no es un pre-filtro (perderíamos
   chunks relevantes por semántica). Se aplica como peso pequeño (0.15) sobre
   el score RRF final.

5. **`text_tsv` config 'simple'**: sin stemming, preserva números y términos
   literales (ej. "9,75%"). Los documentos mezclan español e inglés, así que
   un config localizado a un solo idioma degradaría el otro.

6. **Embeddings normalizados + `vector_cosine_ops`**: equivalente numérico a
   inner product, pero más legible en queries (`<=>` vs `<#>`). HNSW optimiza
   igual.

7. **pgvector `vector(384)` en lugar de `FLOAT8[]`**: obligatorio para usar
   HNSW. Búsqueda O(log n) vs O(n) en Python.

## Deployment

### Mac (desarrollo — `Proyecto_rag/`)
```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
pip install -r requirements.txt
python3 run.py full
```

### Servidor Windows (producción — `Proyecto_RAG_Windows/`)
1. Instalar PostgreSQL 14+ y compilar pgvector. Ver `SETUP_PGVECTOR.md`.
2. `pip install -r requirements.txt`.
3. Pre-descargar modelo offline (si no hay internet en el servidor):
   ```bash
   SENTENCE_TRANSFORMERS_HOME=./models_cache \
     python -c "from sentence_transformers import SentenceTransformer; \
                SentenceTransformer('intfloat/multilingual-e5-small')"
   ```
   Transferir `models_cache/` al servidor y setear
   `SENTENCE_TRANSFORMERS_HOME=C:\Proyecto_RAG_Windows\models_cache`.
4. `python run.py full`.

## Estructura de archivos

```
Proyecto_RAG_Windows/
├── run.py                      # orquestador
├── taxonomy.py                 # patrones compartidos (pasos 1 y 4)
├── 00_generate_jsons.py        # extracción + chunking
├── 01_enrich_metadata.py       # enriquecimiento
├── 02_vectorize.py             # embeddings
├── 03_database.py              # PostgreSQL setup/load/reset/stats
├── 04_search.py                # búsqueda híbrida
├── requirements.txt
├── SETUP_PGVECTOR.md           # instalación de la extensión
├── CLAUDE.md                   # este archivo
├── Datos_prueba/               # PDFs de entrada
└── logs/                       # artefactos generados (no versionar)
```

## Validación rápida

```bash
python3 run.py step 0     # ~1s
python3 run.py step 1     # ~1s
python3 run.py step 2     # ~5-10s (CPU)
python3 run.py step 3 reset  # ~1s
python3 04_search.py "política monetaria 2022" 3
```

Los tres resultados top para esa query deberían incluir
`comunicado1.pdf` (julio 2022, decisión de política con TASA_INTERES e
INFLACION como variables críticas).

---

**Última actualización**: 2026-04-24 — arquitectura rediseñada (chunking
semántico, pgvector/HNSW, RRF híbrido, taxonomía compartida, schema
normalizado).
