# Project Setup Guide

This guide covers setting up CocoIndex v1 projects for different use cases.

## Creating a New Project

### Basic Project

Use the built-in CLI to create a new project:

```bash
cocoindex init my-project
cd my-project
```

This creates:
- `main.py` - Main app definition
- `pyproject.toml` - Dependencies with pre-release config
- `.env` - Environment configuration
- `README.md` - Quick start guide

### Install Dependencies

```bash
pip install -e .
```

## Adding Dependencies for Specific Use Cases

### Vector Embedding Pipeline

For projects that need to generate embeddings:

```toml
# Add to pyproject.toml dependencies
[project]
dependencies = [
    "cocoindex>=1.0.0a1",
    "sentence-transformers",  # For embeddings
    "asyncpg",                # For PostgreSQL
]
```

Install:

```bash
pip install sentence-transformers asyncpg
```

### PostgreSQL Integration

For database-based pipelines:

```toml
# Add to pyproject.toml dependencies
[project]
dependencies = [
    "cocoindex>=1.0.0a1",
    "asyncpg",  # PostgreSQL async driver
]
```

Install:

```bash
pip install asyncpg
```

### SQLite Integration

For local database storage with vector support:

```toml
# Add to pyproject.toml dependencies
[project]
dependencies = [
    "cocoindex>=1.0.0a1",
    "sqlite-vec",  # SQLite vector extension
]
```

Install:

```bash
pip install sqlite-vec
```

### LanceDB Integration

For cloud-native vector storage:

```toml
# Add to pyproject.toml dependencies
[project]
dependencies = [
    "cocoindex>=1.0.0a1",
    "lancedb",
]
```

Install:

```bash
pip install lancedb
```

### Qdrant Integration

For specialized vector database:

```toml
# Add to pyproject.toml dependencies
[project]
dependencies = [
    "cocoindex>=1.0.0a1",
    "qdrant-client",
]
```

Install:

```bash
pip install qdrant-client
```

### LLM-Based Extraction

For structured data extraction using LLMs:

```toml
# Add to pyproject.toml dependencies
[project]
dependencies = [
    "cocoindex>=1.0.0a1",
    "litellm",      # Multi-provider LLM client
    "instructor",   # Structured outputs
    "pydantic>=2.0",  # Data validation
]
```

Install:

```bash
pip install litellm instructor pydantic
```

## Environment Configuration

### Basic Configuration

Create or update `.env` file:

```bash
# CocoIndex database (required)
COCOINDEX_DB=./cocoindex.db
```

### Database Connections

Add database URLs as needed:

```bash
# PostgreSQL
POSTGRES_URL=postgres://user:password@localhost:5432/dbname

# Qdrant
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your-key  # For Qdrant Cloud
```

### API Keys

For LLM-based extraction:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Other providers supported by litellm
# See: https://docs.litellm.ai/docs/providers
```

## Complete Setup Examples

### Example 1: File Transformation Pipeline

**Use case**: Transform markdown files to HTML

```toml
# pyproject.toml
[project]
name = "markdown-to-html"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "cocoindex>=1.0.0a1",
    "markdown-it-py[linkify,plugins]",
]

[tool.uv]
prerelease = "explicit"
```

```bash
# .env
COCOINDEX_DB=./cocoindex.db
```

### Example 2: Vector Embedding Pipeline

**Use case**: Embed code files for semantic search

```toml
# pyproject.toml
[project]
name = "code-embeddings"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "cocoindex>=1.0.0a1",
    "sentence-transformers",
    "asyncpg",
]

[tool.uv]
prerelease = "explicit"
```

```bash
# .env
COCOINDEX_DB=./cocoindex.db
POSTGRES_URL=postgres://cocoindex:cocoindex@localhost:5432/cocoindex
```

### Example 3: LLM Extraction Pipeline

**Use case**: Extract structured data from documents using LLMs

```toml
# pyproject.toml
[project]
name = "llm-extraction"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "cocoindex>=1.0.0a1",
    "litellm",
    "instructor",
    "pydantic>=2.0",
    "asyncpg",
]

[tool.uv]
prerelease = "explicit"
```

```bash
# .env
COCOINDEX_DB=./cocoindex.db
POSTGRES_URL=postgres://cocoindex:cocoindex@localhost:5432/cocoindex
OPENAI_API_KEY=sk-...
```

### Example 4: Multi-Vector Database Pipeline

**Use case**: Store embeddings in multiple vector databases

```toml
# pyproject.toml
[project]
name = "multi-vector-store"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "cocoindex>=1.0.0a1",
    "sentence-transformers",
    "asyncpg",      # PostgreSQL with pgvector
    "lancedb",      # LanceDB
    "qdrant-client",  # Qdrant
]

[tool.uv]
prerelease = "explicit"
```

```bash
# .env
COCOINDEX_DB=./cocoindex.db
POSTGRES_URL=postgres://cocoindex:cocoindex@localhost:5432/cocoindex
QDRANT_URL=http://localhost:6333
```

## Pre-release Configuration

**Important**: CocoIndex v1 is currently in pre-release. Always include this in `pyproject.toml`:

```toml
[tool.uv]
prerelease = "explicit"
```

This allows installing pre-release versions (e.g., `1.0.0a1`).

## Running Your Pipeline

After setup:

```bash
# Install dependencies
pip install -e .

# Run the pipeline
cocoindex update main.py

# Or run specific app
cocoindex update main.py:my_app
```

## Common Setup Issues

### Pre-release Version Not Found

```bash
# Error: Could not find a version that satisfies the requirement cocoindex>=1.0.0a1
```

**Solution**: Add pre-release configuration:

```toml
[tool.uv]
prerelease = "explicit"
```

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'asyncpg'
```

**Solution**: Ensure all dependencies are installed:

```bash
pip install -e .
```

### Database Connection Errors

```bash
# Error: could not connect to server: Connection refused
```

**Solution**: Verify database is running and `.env` has correct URLs. See [Database Setup Guide](./setup_database.md).

## Development Workflow

### Typical Development Cycle

1. **Initialize project**:
   ```bash
   cocoindex init my-project
   cd my-project
   ```

2. **Add dependencies** to `pyproject.toml`

3. **Install**:
   ```bash
   pip install -e .
   ```

4. **Configure** `.env` file

5. **Implement** pipeline in `main.py`

6. **Run**:
   ```bash
   cocoindex update main.py
   ```

7. **Iterate**: Modify code and re-run (only changed data is reprocessed)

### Testing Changes

```bash
# Run pipeline
cocoindex update main.py

# Check what was processed
cocoindex show main.py

# Reset everything and re-run from scratch
cocoindex drop main.py -f
cocoindex update main.py
```

## See Also

- [Database Setup Guide](./setup_database.md) - Database configuration
- [Patterns Reference](./patterns.md) - Example pipeline implementations
- [API Reference](./api_reference.md) - Quick API reference
