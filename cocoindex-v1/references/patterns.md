# CocoIndex v1 Common Patterns

This document describes common patterns and workflows for building CocoIndex v1 pipelines.

## Core Pattern: TargetState = Transform(SourceState)

All CocoIndex pipelines follow this declarative pattern:

```python
# 1. Read source state
source_data = read_from_source()

# 2. Transform
transformed_data = transform(source_data)

# 3. Declare target state
declare_target_state(transformed_data)
```

CocoIndex handles the incremental sync automatically.

---

## Pattern 1: File Transformation Pipeline

**Use case**: Transform files from one format to another (e.g., Markdown → HTML, PDF → Markdown)

### Structure

```
Walk files → Transform content → Declare output files
```

### Implementation

```python
import pathlib
import cocoindex as coco
from cocoindex.connectors import localfs
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

@coco.function(memo=True)
def process_file(file: FileLike, outdir: pathlib.Path) -> None:
    """Transform a single file."""
    content = file.read_text()

    # Transform (e.g., to uppercase)
    transformed = content.upper()

    # Declare output file
    outname = file.file_path.path.stem + "_processed.txt"
    localfs.declare_file(outdir / outname, transformed, create_parent_dirs=True)


@coco.function
def app_main(sourcedir: pathlib.Path, outdir: pathlib.Path) -> None:
    """Main processing function."""
    # Walk source directory
    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=["*.txt", "*.md"],
            excluded_patterns=[".*/**"],
        ),
    )

    # Process each file in its own component
    for f in files:
        coco.mount(
            coco.component_subpath("file", str(f.file_path.path)),
            process_file,
            f,
            outdir,
        )


app = coco.App(
    coco.AppConfig(name="FileTransform"),
    app_main,
    sourcedir=pathlib.Path("./data"),
    outdir=pathlib.Path("./output"),
)
```

### Key Points

- **`memo=True`**: Skip reprocessing unchanged files
- **Component per file**: Each file gets its own processing component for independent updates
- **Auto-cleanup**: Deleting source file automatically removes output file
- **Stable paths**: Use `str(f.file_path.path)` for stable component keys

---

## Pattern 2: Vector Embedding Pipeline

**Use case**: Chunk and embed documents for vector search

### Structure

```
Walk files → Chunk text → Embed chunks → Store in vector DB
```

### Implementation

```python
import asyncio
import pathlib
from dataclasses import dataclass
from typing import AsyncIterator, Annotated

import cocoindex as coco
import cocoindex.asyncio as coco_aio
from cocoindex.connectors import localfs, postgres
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from cocoindex.resources.chunk import Chunk
from numpy.typing import NDArray


# Context and configuration
DATABASE_URL = "postgres://cocoindex:cocoindex@localhost/cocoindex"
PG_DB = coco.ContextKey[postgres.PgDatabase]("pg_db")

_embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
_splitter = RecursiveSplitter()


@dataclass
class Embedding:
    filename: str
    location: str
    text: str
    embedding: Annotated[NDArray, _embedder]
    start_line: int
    end_line: int


@coco_aio.lifespan
async def coco_lifespan(builder: coco_aio.EnvironmentBuilder) -> AsyncIterator[None]:
    """Set up database connection."""
    async with await postgres.create_pool(DATABASE_URL) as pool:
        builder.provide(PG_DB, postgres.register_db("embedding_db", pool))
        yield


@coco.function(memo=True)
async def process_chunk(
    filename: pathlib.PurePath,
    chunk: Chunk,
    table: postgres.TableTarget,
) -> None:
    """Embed a single chunk."""
    location = f"{chunk.start.char_offset}-{chunk.end.char_offset}"
    table.declare_row(
        row=Embedding(
            filename=str(filename),
            location=location,
            text=chunk.text,
            embedding=await _embedder.embed_async(chunk.text),
            start_line=chunk.start.line,
            end_line=chunk.end.line,
        ),
    )


@coco.function(memo=True)
async def process_file(
    file: FileLike,
    table: postgres.TableTarget,
) -> None:
    """Chunk and embed a single file."""
    text = file.read_text()

    # Detect language for syntax-aware chunking
    language = detect_code_language(filename=str(file.file_path.path.name))

    # Split into chunks
    chunks = _splitter.split(
        text,
        chunk_size=1000,
        min_chunk_size=300,
        chunk_overlap=200,
        language=language,
    )

    # Process chunks in parallel
    await asyncio.gather(
        *(process_chunk(file.file_path.path, chunk, table) for chunk in chunks)
    )


@coco.function
def app_main(sourcedir: pathlib.Path) -> None:
    """Main processing function."""
    # Get database from context
    target_db = coco.use_context(PG_DB)

    # Set up target table
    target_table = coco.mount_run(
        coco.component_subpath("setup", "table"),
        target_db.declare_table_target,
        table_name="embeddings",
        table_schema=postgres.TableSchema(
            Embedding,
            primary_key=["filename", "location"],
        ),
    ).result()

    # Walk source files
    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=["*.py", "*.md", "*.txt"],
            excluded_patterns=[".*/**", "__pycache__/**"],
        ),
    )

    # Process each file
    for file in files:
        coco.mount(
            coco.component_subpath("file", str(file.file_path.path)),
            process_file,
            file,
            target_table,
        )


app = coco_aio.App(
    coco_aio.AppConfig(name="EmbeddingPipeline"),
    app_main,
    sourcedir=pathlib.Path("./data"),
)
```

### Key Points

- **Async processing**: Use `async`/`await` for I/O-bound operations
- **Chunk-level memoization**: `process_chunk` with `memo=True` avoids re-embedding unchanged chunks
- **Language-aware splitting**: Detects language and uses syntax-aware chunking
- **Composite primary key**: `["filename", "location"]` for unique chunk identification
- **Resource sharing**: Use `ContextKey` and `use_context()` for shared resources

---

## Pattern 3: Database Source → Transform → Database Target

**Use case**: Transform data from one database to another

### Structure

```
Read from source DB → Transform → Write to target DB
```

### Implementation

```python
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import cocoindex as coco
import cocoindex.asyncio as coco_aio
from cocoindex.connectors import postgres


# Configuration
SOURCE_DB_URL = "postgres://localhost/source_db"
TARGET_DB_URL = "postgres://localhost/target_db"

SOURCE_DB = coco.ContextKey[postgres.PgDatabase]("source_db")
TARGET_DB = coco.ContextKey[postgres.PgDatabase]("target_db")


@dataclass
class SourceRecord:
    id: int
    name: str
    value: float


@dataclass
class TargetRecord:
    id: int
    name: str
    value: float
    processed: bool


@coco_aio.lifespan
async def coco_lifespan(builder: coco_aio.EnvironmentBuilder) -> AsyncIterator[None]:
    """Set up database connections."""
    async with (
        await postgres.create_pool(SOURCE_DB_URL) as source_pool,
        await postgres.create_pool(TARGET_DB_URL) as target_pool,
    ):
        builder.provide(SOURCE_DB, postgres.register_db("source_db", source_pool))
        builder.provide(TARGET_DB, postgres.register_db("target_db", target_pool))
        yield


@coco.function(memo=True)
def process_record(
    record: SourceRecord,
    target_table: postgres.TableTarget,
) -> None:
    """Transform and declare target record."""
    target_table.declare_row(
        row=TargetRecord(
            id=record.id,
            name=record.name.upper(),
            value=record.value * 2,
            processed=True,
        ),
    )


@coco.function
async def app_main() -> None:
    """Main processing function."""
    # Get databases from context
    source_db = coco.use_context(SOURCE_DB)
    target_db = coco.use_context(TARGET_DB)

    # Set up target table
    target_table = coco.mount_run(
        coco.component_subpath("setup", "target_table"),
        target_db.declare_table_target,
        table_name="target_records",
        table_schema=postgres.TableSchema(
            TargetRecord,
            primary_key=["id"],
        ),
    ).result()

    # Read from source table
    source = postgres.PgTableSource(
        source_db,
        schema=SourceRecord,
        table_name="source_records",
    )

    # Process each record
    async for record in source.iterate_async():
        coco.mount(
            coco.component_subpath("record", record.id),
            process_record,
            record,
            target_table,
        )


app = coco_aio.App(
    coco_aio.AppConfig(name="DatabaseTransform"),
    app_main,
)
```

### Key Points

- **Two databases**: Separate source and target database connections
- **Async iteration**: Stream records from source without loading all into memory
- **Per-record components**: Each record gets its own component for independent updates
- **Type-safe**: Use dataclasses for compile-time type checking

---

## Pattern 4: LLM-Based Extraction Pipeline

**Use case**: Extract structured data using LLMs (e.g., from web pages, documents)

### Structure

```
Fetch data → LLM extraction → Multiple target tables
```

### Implementation

```python
import asyncio
import instructor
from dataclasses import dataclass
from typing import AsyncIterator
from pydantic import BaseModel

import cocoindex as coco
import cocoindex.asyncio as coco_aio
from cocoindex.connectors import postgres
from litellm import acompletion


# Configuration
DATABASE_URL = "postgres://cocoindex:cocoindex@localhost/cocoindex"
PG_DB = coco.ContextKey[postgres.PgDatabase]("pg_db")

_instructor_client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)


# Pydantic models for LLM extraction
class ExtractedTopic(BaseModel):
    name: str
    description: str


class ExtractionResult(BaseModel):
    title: str
    topics: list[ExtractedTopic]


# Database schemas
@dataclass
class Message:
    id: int
    title: str
    content: str


@dataclass
class Topic:
    message_id: int
    name: str
    description: str


@coco_aio.lifespan
async def coco_lifespan(builder: coco_aio.EnvironmentBuilder) -> AsyncIterator[None]:
    """Set up database connection."""
    async with await postgres.create_pool(DATABASE_URL) as pool:
        builder.provide(PG_DB, postgres.register_db("extraction_db", pool))
        yield


@coco.function(memo=True)
async def extract_and_store(
    content: str,
    message_id: int,
    messages_table: postgres.TableTarget,
    topics_table: postgres.TableTarget,
) -> None:
    """Extract topics using LLM and declare rows."""
    # Extract using LLM
    result = await _instructor_client.chat.completions.create(
        model="gpt-4",
        response_model=ExtractionResult,
        messages=[{
            "role": "user",
            "content": f"Extract topics from this text:\n\n{content}"
        }],
    )

    # Declare message row
    messages_table.declare_row(
        row=Message(
            id=message_id,
            title=result.title,
            content=content,
        ),
    )

    # Declare topic rows
    for topic in result.topics:
        topics_table.declare_row(
            row=Topic(
                message_id=message_id,
                name=topic.name,
                description=topic.description,
            ),
        )


@coco.function
def app_main(input_texts: list[str]) -> None:
    """Main processing function."""
    # Get database from context
    target_db = coco.use_context(PG_DB)

    # Set up target tables
    messages_table = coco.mount_run(
        coco.component_subpath("setup", "messages_table"),
        target_db.declare_table_target,
        table_name="messages",
        table_schema=postgres.TableSchema(Message, primary_key=["id"]),
    ).result()

    topics_table = coco.mount_run(
        coco.component_subpath("setup", "topics_table"),
        target_db.declare_table_target,
        table_name="topics",
        table_schema=postgres.TableSchema(
            Topic,
            primary_key=["message_id", "name"],
        ),
    ).result()

    # Process each input
    for idx, text in enumerate(input_texts):
        coco.mount(
            coco.component_subpath("text", idx),
            extract_and_store,
            text,
            idx,
            messages_table,
            topics_table,
        )


app = coco_aio.App(
    coco_aio.AppConfig(name="LLMExtraction"),
    app_main,
    input_texts=["text1...", "text2..."],
)
```

### Key Points

- **LLM memoization**: `memo=True` on extraction function avoids re-calling LLM for unchanged inputs
- **Multiple target tables**: Declare rows in multiple tables from single component
- **Pydantic models**: Use for structured LLM outputs with validation
- **Composite keys**: `["message_id", "name"]` for topic table
- **instructor library**: Simplifies structured LLM extraction

---

## Pattern 5: Context Management for Shared Resources

**Use case**: Share expensive resources (models, connections, configs) across components

### Implementation

```python
import cocoindex as coco
from typing import Any


# Define context keys (typed for safety)
EMBEDDER = coco.ContextKey[Any]("embedder")
CONFIG = coco.ContextKey[dict]("config")


@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    """Set up and provide shared resources."""
    # Load expensive model once
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load configuration
    config = {"chunk_size": 1000, "overlap": 200}

    # Provide to context
    builder.provide(EMBEDDER, embedder)
    builder.provide(CONFIG, config)

    yield

    # Cleanup (if needed)


@coco.function
def process_item(item: str) -> None:
    """Use shared resources from context."""
    embedder = coco.use_context(EMBEDDER)
    config = coco.use_context(CONFIG)

    # Use resources
    embedding = embedder.encode(item)
    chunk_size = config["chunk_size"]
    # ...
```

### Key Points

- **ContextKey**: Type-safe keys for resources
- **Lifespan**: Set up/teardown resources once per app run
- **use_context()**: Retrieve resources in processing components
- **Avoids re-initialization**: Models/connections loaded once and shared

---

## Pattern 6: Stable Component Paths

**Use case**: Ensure component paths remain stable for proper memoization

### Best Practices

```python
# ✅ Good: Stable string paths
coco.component_subpath("file", str(file.file_path.path))
coco.component_subpath("record", record.id)
coco.component_subpath("thread", thread_id, "comment", comment_id)

# ✅ Good: Using file.stable_key
coco.component_subpath(file.stable_key)

# ❌ Bad: Non-stable identifiers
coco.component_subpath("file", file)  # Object reference changes
coco.component_subpath("idx", idx)    # Index may change when items inserted
```

### Hierarchical Paths

```python
# Organize with hierarchy
with coco.component_subpath("setup"):
    # Setup operations
    table = coco.mount_run(...)

with coco.component_subpath("processing"):
    # Processing operations
    for item in items:
        coco.mount(coco.component_subpath(item.id), ...)
```

---

## Pattern 7: Mixing Sync and Async

**Use case**: Use async for I/O-bound operations, sync for CPU-bound

### Implementation

```python
import asyncio
import cocoindex as coco
from cocoindex.connectors import localfs


@coco.function
async def fetch_data(url: str) -> dict:
    """Async I/O operation."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()


@coco.function(memo=True)
def process_data(data: dict) -> None:
    """Sync CPU-bound operation."""
    # Heavy computation
    result = expensive_computation(data)
    # Declare target state
    localfs.declare_file("output.json", json.dumps(result))


@coco.function
def app_main(urls: list[str]) -> None:
    """Sync main can mount async components."""
    for url in urls:
        # Mount async component from sync context
        coco.mount(
            coco.component_subpath(url),
            fetch_data,
            url,
        )
```

### Key Points

- **Mount async from sync**: No restrictions on mixing
- **Each component is consistent**: Keep each component either fully sync or fully async
- **Use async for I/O**: Network requests, file I/O, database queries
- **Use sync for CPU**: Heavy computation, data processing

---

## Common Anti-Patterns to Avoid

### ❌ Reprocessing Everything

```python
# Bad: No memoization
@coco.function  # Missing memo=True
def process_file(file: FileLike) -> None:
    # Expensive operation runs every time
    result = expensive_operation(file.read_text())
```

**Fix**: Add `memo=True` for expensive operations.

### ❌ Unstable Component Paths

```python
# Bad: Using object references
for file in files:
    coco.mount(coco.component_subpath(file), ...)  # File object reference changes

# Bad: Using list indices
for idx, item in enumerate(items):
    coco.mount(coco.component_subpath(idx), ...)  # Index changes when items added
```

**Fix**: Use stable identifiers like IDs or paths.

### ❌ Not Using Context for Shared Resources

```python
# Bad: Loading model in every component
@coco.function
def process(text: str) -> None:
    model = SentenceTransformer("model")  # Loaded every time!
    embedding = model.encode(text)
```

**Fix**: Load once in lifespan and use context.

### ❌ Mixing Target State with Side Effects

```python
# Bad: Side effects in processing function
@coco.function
def process(data: dict) -> None:
    result = transform(data)
    # Side effect - not tracked by CocoIndex!
    requests.post("https://api.example.com", json=result)
```

**Fix**: Only declare target states; CocoIndex handles syncing.

---

## See Also

- [Programming Guide - Core Concepts](../programming_guide/core_concepts.md)
- [Connectors Reference](./connectors.md)
- [API Reference](./api_reference.md)
