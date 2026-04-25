# Instalación de pgvector

El pipeline usa [pgvector](https://github.com/pgvector/pgvector) para almacenar
embeddings de 384 dimensiones con un índice HNSW. No es un paquete Python — se
compila e instala en el servidor PostgreSQL.

Verificar si ya está instalada antes de hacer cualquier cosa:

```sql
psql -U postgres -c "SELECT * FROM pg_available_extensions WHERE name='vector';"
```

Si devuelve una fila, ya está lista. Salta directo a [Variables de conexión](#variables-de-conexión).

---

## Windows sin permisos de Administrador (método recomendado)

Este es el método cuando no puedes abrir el Command Prompt como Administrador,
que es el caso más común en servidores corporativos.

### Paso 1 — Verificar acceso a los directorios de PostgreSQL

```cmd
icacls "C:\Program Files\PostgreSQL\14\lib"
icacls "C:\Program Files\PostgreSQL\14\share\extension"
```

Busca tu usuario en la salida. Si aparece con **(W)**, **(M)** o **(F)** tienes
permiso de escritura y puedes continuar. Si no aparece, ve al
[método con Administrador](#windows-con-permisos-de-administrador).

### Paso 2 — Compilar pgvector

Abrir **"x64 Native Tools Command Prompt for VS 2022"** (no hace falta como
Administrador para compilar, solo para copiar archivos al paso siguiente).

```cmd
cd C:\ruta\al\repositorio\pgvector

set "PGROOT=C:\Program Files\PostgreSQL\14"

nmake /F Makefile.win
```

Cuando termine verás los archivos generados en la carpeta actual:
`vector.dll`, `vector.control`, `vector--*.sql`.

### Paso 3 — Copiar los archivos manualmente

```cmd
copy vector.dll "C:\Program Files\PostgreSQL\14\lib\"

copy vector.control "C:\Program Files\PostgreSQL\14\share\extension\"

copy vector--*.sql "C:\Program Files\PostgreSQL\14\share\extension\"
```

Si algún `copy` da error de acceso denegado, necesitas el método con Administrador.

### Paso 4 — Reiniciar PostgreSQL

```cmd
net stop postgresql-x64-14
net start postgresql-x64-14
```

Si no tienes permisos para reiniciar el servicio, pídele al administrador solo
ese paso (no necesita ser él quien copie los archivos).

### Paso 5 — Verificar

```cmd
psql -U postgres -c "SELECT name, default_version FROM pg_available_extensions WHERE name='vector';"
```

Debe aparecer una fila. Si aparece, ya puedes correr el pipeline.

---

## Windows con permisos de Administrador

Si tienes acceso de Administrador, el proceso es más simple:

1. Abrir **"x64 Native Tools Command Prompt for VS 2022"** como **Administrador**.
2. Ejecutar:

```cmd
cd C:\ruta\al\repositorio\pgvector

set "PGROOT=C:\Program Files\PostgreSQL\14"

nmake /F Makefile.win
nmake /F Makefile.win install
```

3. Reiniciar PostgreSQL:

```cmd
net stop postgresql-x64-14
net start postgresql-x64-14
```

---

## macOS (Homebrew)

```bash
brew install pgvector
```

---

## Linux (compilar desde fuente)

```bash
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

---

## Variables de conexión

El pipeline lee las credenciales de las variables de entorno estándar de PostgreSQL.
Setearlas antes de correr `python run.py full`:

```cmd
set PGHOST=localhost
set PGPORT=5432
set PGUSER=postgres
set PGPASSWORD=tu_contraseña
set PGDATABASE=rag_banco
```

| Variable | Default |
|---|---|
| `PGHOST` | `localhost` |
| `PGPORT` | `5432` |
| `PGUSER` | usuario actual del SO |
| `PGPASSWORD` | vacío |
| `PGDATABASE` | `rag_banco` |

Para no tener que setear las variables cada vez, créalas en
**Panel de Control → Sistema → Variables de entorno del sistema**.

---

## Troubleshooting

**`nmake` no se reconoce como comando**
No estás en el "x64 Native Tools Command Prompt". Búscalo en el menú inicio
buscando "Native Tools".

**`copy` da acceso denegado en el Paso 3**
Tu usuario no tiene escritura en la carpeta de PostgreSQL. Opciones:
- Pedir al administrador que ejecute solo los comandos `copy` del Paso 3
- Pedir al administrador que ejecute el método completo con Administrador

**La extensión no aparece tras reiniciar**
Verificar que los archivos estén en el lugar correcto:
```cmd
dir "C:\Program Files\PostgreSQL\14\share\extension\vector*"
dir "C:\Program Files\PostgreSQL\14\lib\vector.dll"
```

**Error al crear la extensión en el pipeline**
```
ERROR: could not open extension control file
```
El archivo `vector.control` no está en la carpeta `extension`. Repetir el Paso 3.
