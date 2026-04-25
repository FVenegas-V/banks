# Instalación de pgvector

El pipeline usa la extensión [pgvector](https://github.com/pgvector/pgvector)
de PostgreSQL para almacenar embeddings de 384 dimensiones con un índice HNSW.
`pgvector` NO es un paquete Python; se compila y se instala en el servidor
PostgreSQL.

Verificar si ya está instalada:

```sql
SELECT * FROM pg_available_extensions WHERE name = 'vector';
```

Si devuelve una fila, la extensión está disponible y `python3 03_database.py setup`
la habilitará automáticamente con `CREATE EXTENSION vector`.

---

## Linux / macOS (compilar desde fuente)

Prerequisitos: `make`, `gcc`, `postgresql-server-dev-<version>`.

```bash
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

En sistemas con múltiples versiones de PostgreSQL, fijar el `PG_CONFIG`:

```bash
make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config
sudo make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install
```

En macOS con Homebrew ya viene disponible (Postgres 14+):

```bash
brew install pgvector
```

---

## Windows (servidor de producción)

La compilación en Windows requiere Visual Studio Build Tools y el entorno
`x64 Native Tools Command Prompt for VS 20XX`.

1. Descargar el código fuente de pgvector (ya hecho):
   `git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git`
2. Abrir **x64 Native Tools Command Prompt for VS 2022**.
3. Fijar la ruta a la instalación de PostgreSQL (ejemplo para PG 16):
   ```cmd
   set "PGROOT=C:\Program Files\PostgreSQL\16"
   ```
4. Compilar e instalar:
   ```cmd
   cd pgvector
   nmake /F Makefile.win
   nmake /F Makefile.win install
   ```
5. Reiniciar el servicio PostgreSQL:
   ```powershell
   Restart-Service postgresql-x64-16
   ```
6. Verificar:
   ```sql
   SELECT * FROM pg_available_extensions WHERE name = 'vector';
   ```

Si `nmake` falla por permisos, ejecutar el Command Prompt **como Administrador**.

---

## Variables de conexión esperadas

El pipeline lee las credenciales de las variables de entorno estándar de
PostgreSQL:

| Variable     | Default     |
|--------------|-------------|
| `PGHOST`     | `localhost` |
| `PGPORT`     | `5432`      |
| `PGUSER`     | usuario actual del SO |
| `PGPASSWORD` | vacío       |
| `PGDATABASE` | `rag_banco` |

Para el servidor Windows, la forma recomendada es un archivo `.pgpass`
(o `%APPDATA%\postgresql\pgpass.conf`) con los credentials del usuario de
la aplicación. Evitar pasar contraseñas por línea de comandos.
