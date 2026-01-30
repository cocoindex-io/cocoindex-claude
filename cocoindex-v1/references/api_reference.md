# CocoIndex v1 API Reference

This document provides a quick reference for the most commonly used CocoIndex v1 APIs.

## Core Decorators

### `@coco.function`

Mark a function as a CocoIndex function (processing component or transform).

```python
@coco.function(memo=True, version=None)
def my_function(arg1: str, arg2: int) -> None:
    ...
```

**Parameters:**
- `memo: bool = False` - Enable memoization (skip if inputs/code unchanged)
- `version: str | None = None` - Explicit version to force re-execution

**Usage:**
- Use on processing component functions
- Use on transform functions
- Add `memo=True` for expensive operations (embeddings, LLM calls, heavy computation)
- Set `version` when code changes but function signature doesn't

### `@coco.lifespan` / `@coco_aio.lifespan`

Define setup and cleanup logic for the app.

```python
@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    # Setup
    resource = create_resource()
    builder.provide(CONTEXT_KEY, resource)
    yield
    # Cleanup
    resource.close()
```

**Usage:**
- Load expensive resources once (models, database pools)
- Register resources with `builder.provide()`
- Cleanup in finally block or after yield
- Use async version for async resources

---

## Component Management

### `coco.mount()`

Mount a processing component.

```python
coco.mount(
    path: ComponentPath,
    func: Callable,
    *args,
    **kwargs,
)
```

**Parameters:**
- `path` - Component path (from `component_subpath()`)
- `func` - Function to execute
- `*args, **kwargs` - Arguments to pass to function

**Returns:** `ComponentHandle`

**Usage:**
- Mount one component per item for fine-grained updates
- Use stable paths for proper memoization

### `coco.mount_run()`

Mount and immediately wait for component result.

```python
result = coco.mount_run(
    path: ComponentPath,
    func: Callable,
    *args,
    **kwargs,
).result()
```

**Usage:**
- Use for setup operations that return values (e.g., table targets)
- Returns `ComponentHandle` with `.result()` method

### `coco.component_subpath()`

Create a stable component path.

```python
path = coco.component_subpath(*keys: StableKey)
```

**Parameters:**
- `*keys` - Stable keys (str, int, bool, bytes, UUID, or tuples)

**Returns:** `ComponentPath`

**Usage:**
```python
# Single key
coco.component_subpath("setup")

# Multiple keys
coco.component_subpath("file", str(file_path))
coco.component_subpath("record", record.id)

# Hierarchical
coco.component_subpath("thread", thread_id, "comment", comment_id)
```

### Context Manager Form

Use `with` for hierarchical organization:

```python
with coco.component_subpath("setup"):
    # All mounts here are under "setup" path
    table = coco.mount_run(...)

with coco.component_subpath("processing"):
    # All mounts here are under "processing" path
    for item in items:
        coco.mount(coco.component_subpath(item.id), ...)
```

---

## Context System

### `coco.ContextKey[T]`

Type-safe key for storing/retrieving resources.

```python
MY_RESOURCE = coco.ContextKey[MyType]("my_resource")
```

**Type parameter:**
- `T` - Type of resource (for type checking)

### `builder.provide()`

Register a resource in the context (used in lifespan).

```python
@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    builder.provide(CONTEXT_KEY, resource)
    yield
```

### `coco.use_context()`

Retrieve a resource from context.

```python
@coco.function
def my_function() -> None:
    resource = coco.use_context(CONTEXT_KEY)
    # Use resource...
```

---

## App Creation

### Synchronous App

```python
import cocoindex as coco

app = coco.App(
    config: AppConfig,
    main_func: Callable,
    **params,
)

# Run
app.update(report_to_stdout=True)
```

### Asynchronous App

```python
import cocoindex.asyncio as coco_aio
import asyncio

app = coco_aio.App(
    config: AppConfig,
    main_func: Callable,
    **params,
)

# Run
asyncio.run(app.update(report_to_stdout=True))
```

### AppConfig

```python
coco.AppConfig(
    name: str,              # App name (required)
    db_path: Path | None,   # Override default DB path
)
```

---

## CLI Commands

### Initialize Project

```bash
cocoindex init [PROJECT_NAME] [--dir DIRECTORY]
```

Creates a new project with starter files.

### Update (Run) App

```bash
cocoindex update APP_TARGET
```

**Examples:**
```bash
cocoindex update main.py
cocoindex update main.py:app_name
cocoindex update my_module:app_name
```

### Drop App

```bash
cocoindex drop APP_TARGET [-f|--force]
```

Drops all target states and internal state.

### List Apps

```bash
cocoindex ls [APP_TARGET]
cocoindex ls --db ./cocoindex.db
```

### Show App Structure

```bash
cocoindex show APP_TARGET
```

Shows component paths for the app.

---

## Text Operations

**Import:** `from cocoindex.ops.text import ...`

### `detect_code_language()`

```python
language = detect_code_language(
    filename: str | None = None,
    content: str | None = None,
)
```

**Returns:** Language name (e.g., `"python"`, `"rust"`) or `None`

### `RecursiveSplitter`

```python
splitter = RecursiveSplitter()

chunks = splitter.split(
    text: str,
    chunk_size: int,
    min_chunk_size: int,
    chunk_overlap: int,
    language: str | None = None,
)
```

**Returns:** `list[Chunk]`

**Chunk attributes:**
- `text: str` - Chunk content
- `start: ChunkPosition` - Start position (char_offset, line)
- `end: ChunkPosition` - End position (char_offset, line)

---

## Embedding Operations

**Import:** `from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder`

### `SentenceTransformerEmbedder`

```python
embedder = SentenceTransformerEmbedder(
    model_name: str,
    device: str | None = None,  # "cuda", "cpu", or None for auto
)

# Sync
embedding = embedder.embed(text: str) -> NDArray

# Async
embedding = await embedder.embed_async(text: str) -> NDArray
```

**Usage as VectorSchemaProvider:**
```python
from typing import Annotated
from numpy.typing import NDArray

@dataclass
class Record:
    text: str
    vector: Annotated[NDArray, embedder]  # Auto-infer dimensions
```

---

## File Operations

**Import:** `from cocoindex.connectors import localfs`

### Reading Files

```python
from cocoindex.resources.file import PatternFilePathMatcher

files = localfs.walk_dir(
    path: Path,
    recursive: bool = False,
    path_matcher: PatternFilePathMatcher | None = None,
)

# Iterate
for file in files:  # Returns File objects
    content = file.read_text()
    # or: bytes_content = file.read_bytes()
```

**PatternFilePathMatcher:**
```python
matcher = PatternFilePathMatcher(
    included_patterns: list[str] = ["*"],
    excluded_patterns: list[str] = [],
)
```

### Writing Files

```python
# Single file
localfs.declare_file(
    path: Path,
    content: str | bytes,
    create_parent_dirs: bool = False,
)

# Directory target
dir_target = coco.mount_run(
    coco.component_subpath("output"),
    localfs.declare_dir_target,
    path=Path("./output"),
    create_parent_dirs=True,
).result()

dir_target.declare_file("file.txt", "content")
```

### Stable Paths

```python
# Register base directory
base = localfs.register_base_dir("data", Path("./data"))

# Build paths
file_path = base / "subdir" / "file.txt"

# Resolve to absolute path
absolute = file_path.resolve()
```

---

## Database Target State

### Common Pattern (all DB connectors)

```python
# 1. Register database
@coco.lifespan
async def coco_lifespan(builder):
    async with await connector.create_pool(URL) as pool:
        db = connector.register_db("db_key", pool)
        builder.provide(DB_CONTEXT_KEY, db)
        yield

# 2. Declare table/collection target
@coco.function
def app_main():
    db = coco.use_context(DB_CONTEXT_KEY)

    target = coco.mount_run(
        coco.component_subpath("setup", "table"),
        db.declare_table_target,  # or declare_collection_target for Qdrant
        table_name="my_table",
        table_schema=Schema(...),
    ).result()

    # 3. Declare rows
    target.declare_row(row=MyRecord(...))
```

### Table Schema from Python Class

```python
from dataclasses import dataclass
from typing import Annotated
from numpy.typing import NDArray

@dataclass
class MyRecord:
    id: int
    name: str
    vector: Annotated[NDArray, embedder]

schema = postgres.TableSchema(
    MyRecord,
    primary_key=["id"],
)
```

---

## Type Annotations for Vectors

### With Auto-Dimension Inference

```python
from typing import Annotated
from numpy.typing import NDArray

embedder = SentenceTransformerEmbedder("model-name")

@dataclass
class Record:
    vector: Annotated[NDArray, embedder]  # Dimensions inferred
```

### With Explicit Dimensions

```python
from cocoindex.resources.vector import VectorSchema

schema = VectorSchema(dim=384)

@dataclass
class Record:
    vector: Annotated[NDArray, schema]
```

---

## Async/Sync Flexibility

### Import Strategy

```python
# For async apps
import cocoindex.asyncio as coco_aio

# For sync apps
import cocoindex as coco

# Connectors and utilities work with both
from cocoindex.connectors import postgres, localfs
from cocoindex.ops.text import RecursiveSplitter
```

### Mixing Async and Sync

```python
# Sync main can mount async components
@coco.function
def sync_main():
    for item in items:
        coco.mount(path, async_process, item)  # OK!

# Async main can mount sync components
@coco.function
async def async_main():
    for item in items:
        coco.mount(path, sync_process, item)  # OK!
```

---

## Environment Configuration

### `.env` File

CocoIndex automatically loads `.env` from current directory:

```bash
# CocoIndex database
COCOINDEX_DB=./cocoindex.db

# PostgreSQL
POSTGRES_URL=postgres://user:pass@localhost/db

# API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Manual Environment Builder

```python
@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    # Override DB path
    builder.settings.db_path = Path("./custom.db")
    yield
```

---

## Common Type Hints

```python
from typing import Annotated, AsyncIterator
from numpy.typing import NDArray
import pathlib

from cocoindex.connectors import postgres, localfs
from cocoindex.resources.file import FileLike
from cocoindex.resources.chunk import Chunk
```

---

## See Also

- [Connectors Reference](./connectors.md) - Detailed connector documentation
- [Patterns Reference](./patterns.md) - Common pipeline patterns
- [Programming Guide](../programming_guide/) - Comprehensive guide
