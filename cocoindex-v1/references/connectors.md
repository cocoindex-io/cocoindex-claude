# CocoIndex v1 Connectors Reference

This document provides a comprehensive reference for all CocoIndex v1 connectors.

## Overview

CocoIndex provides connectors for databases (PostgreSQL, SQLite, LanceDB, Qdrant) and file systems (LocalFS). Connectors support:
- **Source operations**: Reading data from external systems
- **Target operations**: Declaring target states that CocoIndex syncs to external systems

## Common Patterns

### Target State Management
All write-capable connectors follow this pattern:
1. **Register** - Register connection/database with stable key
2. **Declare Parent** - Declare table/collection/directory as target state
3. **Declare Children** - Declare rows/points/files within the parent
4. **Lifecycle Management** - `managed_by="system"` (Coc oIndex manages) or `"user"` (assumes exists)

### Vector Support
Most connectors support vector embeddings via `VectorSchemaProvider`:
- Use embedding model as schema provider (recommended - auto-infers dimensions)
- Or use explicit `VectorSchema(dim=X)` when not using embedder

---

## PostgreSQL Connector

**Import**: `from cocoindex.connectors import postgres`

**Capabilities**: Source (read) and Target (write)

**Extension support**: pgvector for vector operations

### Connection Setup

```python
import cocoindex.asyncio as coco_aio
from cocoindex.connectors import postgres

PG_DB = coco.ContextKey[postgres.PgDatabase]("pg_db")

@coco_aio.lifespan
async def coco_lifespan(builder: coco_aio.EnvironmentBuilder):
    async with await postgres.create_pool(DATABASE_URL) as pool:
        builder.provide(PG_DB, postgres.register_db("my_db", pool))
        yield
```

### As Source

```python
from dataclasses import dataclass

@dataclass
class Record:
    id: int
    name: str

# Read from table
source = postgres.PgTableSource(
    pool,
    table_name="my_table",
    row_type=Record,  # Auto-convert to dataclass
)

# Iterate
async for record in source.iterate_async():
    print(record.id, record.name)
```

### As Target

```python
from dataclasses import dataclass
from typing import Annotated
from numpy.typing import NDArray

@dataclass
class Embedding:
    id: int
    text: str
    vector: Annotated[NDArray, embedder]  # Auto-infer dimensions

# Declare table
target_table = coco.mount_run(
    coco.component_subpath("setup", "table"),
    target_db.declare_table_target,
    table_name="embeddings",
    table_schema=postgres.TableSchema(
        Embedding,
        primary_key=["id"],
    ),
).result()

# Declare rows
target_table.declare_row(
    row=Embedding(id=1, text="hello", vector=embedding_array)
)
```

### Type Mapping

| Python | PostgreSQL |
|--------|-----------|
| `bool` | `boolean` |
| `int` | `bigint` |
| `float` | `double precision` |
| `str` | `text` |
| `bytes` | `bytea` |
| `UUID` | `uuid` |
| `datetime` | `timestamp with time zone` |
| `list`, `dict` | `jsonb` |
| `NDArray` + vector schema | `vector(n)` or `halfvec(n)` |

---

## SQLite Connector

**Import**: `from cocoindex.connectors import sqlite`

**Capabilities**: Target (write) only

**Extension support**: sqlite-vec for vector operations

### Connection Setup

```python
import cocoindex as coco
from cocoindex.connectors import sqlite

SQLITE_DB = coco.ContextKey[sqlite.SqliteDatabase]("sqlite_db")

@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    conn = sqlite.connect("./data.db", load_vec="auto")
    builder.provide(SQLITE_DB, sqlite.register_db("my_db", conn))
    yield
    conn.close()
```

### As Target

```python
from dataclasses import dataclass
from typing import Annotated
from numpy.typing import NDArray

@dataclass
class Embedding:
    id: int
    text: str
    vector: Annotated[NDArray, embedder]

# Declare table
target_table = coco.mount_run(
    coco.component_subpath("setup", "table"),
    target_db.declare_table_target,
    table_name="embeddings",
    table_schema=sqlite.TableSchema(
        Embedding,
        primary_key=["id"],
    ),
).result()

# Declare rows
target_table.declare_row(
    row=Embedding(id=1, text="hello", vector=embedding_array)
)
```

### Type Mapping

| Python | SQLite |
|--------|--------|
| `bool` | `INTEGER` (0/1) |
| `int` | `INTEGER` |
| `float` | `REAL` |
| `str` | `TEXT` |
| `bytes` | `BLOB` |
| `datetime` | `TEXT` (ISO format) |
| `list`, `dict` | `TEXT` (JSON) |
| `NDArray` + vector schema | `BLOB` (sqlite-vec format) |

**Note**: Use Homebrew Python on macOS for extension support.

---

## LanceDB Connector

**Import**: `from cocoindex.connectors import lancedb`

**Capabilities**: Target (write) only

**Storage**: Local or cloud (S3, etc.)

### Connection Setup

```python
import cocoindex.asyncio as coco_aio
from cocoindex.connectors import lancedb

LANCE_DB = coco.ContextKey[lancedb.LanceDatabase]("lance_db")

@coco_aio.lifespan
async def coco_lifespan(builder: coco_aio.EnvironmentBuilder):
    conn = await lancedb.connect_async("./lancedb_data")
    builder.provide(LANCE_DB, lancedb.register_db("my_db", conn))
    yield
```

### As Target

```python
from dataclasses import dataclass
from typing import Annotated
from numpy.typing import NDArray

@dataclass
class Embedding:
    id: int
    text: str
    vector: Annotated[NDArray, embedder]

# Declare table
target_table = coco.mount_run(
    coco.component_subpath("setup", "table"),
    target_db.declare_table_target,
    table_name="embeddings",
    table_schema=lancedb.TableSchema(
        Embedding,
        primary_key=["id"],
    ),
).result()

# Declare rows
target_table.declare_row(
    row=Embedding(id=1, text="hello", vector=embedding_array)
)
```

### Type Mapping

| Python | PyArrow |
|--------|---------|
| `bool` | `bool` |
| `int` | `int64` |
| `float` | `float64` |
| `str` | `string` |
| `bytes` | `binary` |
| `list`, `dict` | `string` (JSON) |
| `NDArray` + vector schema | `fixed_size_list<float>` |

---

## Qdrant Connector

**Import**: `from cocoindex.connectors import qdrant`

**Capabilities**: Target (write) only

**Model**: Point-oriented with schemaless payloads

### Connection Setup

```python
import cocoindex as coco
from cocoindex.connectors import qdrant

QDRANT_DB = coco.ContextKey[qdrant.QdrantDatabase]("qdrant_db")

@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    client = qdrant.create_client("http://localhost:6333")
    builder.provide(QDRANT_DB, qdrant.register_db("my_db", client))
    yield
```

### As Target (Single Vector)

```python
from qdrant_client.models import PointStruct

# Declare collection
collection = coco.mount_run(
    coco.component_subpath("setup", "collection"),
    target_db.declare_collection_target,
    collection_name="embeddings",
    schema=qdrant.CollectionSchema(
        vectors=qdrant.QdrantVectorDef(
            schema=embedder,  # Auto-infer dimensions
            distance="cosine",
        ),
    ),
).result()

# Declare points
collection.declare_point(
    point=PointStruct(
        id="point-1",
        vector=embedding_array.tolist(),
        payload={"text": "hello", "metadata": {...}},
    )
)
```

### As Target (Named Vectors)

```python
# Declare collection with named vectors
collection = coco.mount_run(
    coco.component_subpath("setup", "collection"),
    target_db.declare_collection_target,
    collection_name="multimodal",
    schema=qdrant.CollectionSchema(
        vectors={
            "text": qdrant.QdrantVectorDef(schema=text_embedder),
            "image": qdrant.QdrantVectorDef(schema=image_embedder),
        },
    ),
).result()

# Declare points
collection.declare_point(
    point=PointStruct(
        id="point-1",
        vector={
            "text": text_embedding.tolist(),
            "image": image_embedding.tolist(),
        },
        payload={"title": "example"},
    )
)
```

### Distance Metrics

- `"cosine"` - Cosine similarity (default)
- `"dot"` - Dot product similarity
- `"euclid"` - Euclidean distance (L2)

---

## LocalFS Connector

**Import**: `from cocoindex.connectors import localfs`

**Capabilities**: Source (read) and Target (write)

**Key feature**: Stable memoization across project moves

### Stable Memoization

```python
from cocoindex.connectors import localfs

# Register base directory with stable key
data_dir = localfs.register_base_dir("data", pathlib.Path("./data"))

# Build paths using / operator
file_path = data_dir / "subdir" / "file.txt"

# Resolve to absolute path when needed
absolute_path = file_path.resolve()
```

### As Source

```python
from cocoindex.resources.file import PatternFilePathMatcher
from cocoindex.connectors import localfs

# Walk directory
files = localfs.walk_dir(
    sourcedir,
    recursive=True,
    path_matcher=PatternFilePathMatcher(
        included_patterns=["*.py", "*.md"],
        excluded_patterns=[".*/**", "__pycache__/**"],
    ),
)

# Process files
for file in files:
    content = file.read_text()
    # Process content...
```

### As Target (Single File)

```python
from cocoindex.connectors import localfs

# Declare single file
localfs.declare_file(
    outdir / "output.txt",
    "file content",
    create_parent_dirs=True,
)
```

### As Target (Directory)

```python
# Declare directory target
output_dir = coco.mount_run(
    coco.component_subpath("setup", "output"),
    localfs.declare_dir_target,
    path=pathlib.Path("./output"),
    create_parent_dirs=True,
).result()

# Declare files within directory
output_dir.declare_file("file1.txt", "content 1")
output_dir.declare_file("file2.txt", "content 2")
```

### Pattern Matching

```python
from cocoindex.resources.file import PatternFilePathMatcher

matcher = PatternFilePathMatcher(
    included_patterns=[
        "*.py",           # Python files
        "*.md",           # Markdown files
        "src/**/*.ts",    # TypeScript in src/
    ],
    excluded_patterns=[
        ".*",             # Hidden files/dirs
        "**/__pycache__", # Python cache
        "**/node_modules", # Node modules
        "**/test_*",      # Test files
    ],
)
```

---

## Connector Comparison

| Connector | Source | Target | Vectors | Use Case |
|-----------|--------|--------|---------|----------|
| **PostgreSQL** | ✅ | ✅ | pgvector | Production SQL + vectors |
| **SQLite** | ❌ | ✅ | sqlite-vec | Local SQL + vectors |
| **LanceDB** | ❌ | ✅ | ✅ | Cloud-native vector DB |
| **Qdrant** | ❌ | ✅ | ✅ | Specialized vector DB |
| **LocalFS** | ✅ | ✅ | N/A | File-based pipelines |

## See Also

- [Programming Guide - Target State](../programming_guide/target_state.md)
- [API Reference](./api_reference.md)
- [Patterns](./patterns.md)
