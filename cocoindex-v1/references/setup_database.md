# Database Setup Guide

This guide covers setting up databases for use with CocoIndex v1.

## PostgreSQL with pgvector

### Using Docker (Recommended)

The easiest way to get started is using Docker with the official pgvector image:

```bash
# Create docker-compose.yml
cat > docker-compose.yml <<EOF
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: cocoindex-postgres
    environment:
      POSTGRES_USER: cocoindex
      POSTGRES_PASSWORD: cocoindex
      POSTGRES_DB: cocoindex
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cocoindex"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
EOF

# Start the database
docker-compose up -d

# Verify it's running
docker-compose ps
```

Your connection URL will be: `postgres://cocoindex:cocoindex@localhost:5432/cocoindex`

### Using Existing PostgreSQL

If you have an existing PostgreSQL installation, enable the pgvector extension:

```sql
-- Connect to your database
psql -U your_user -d your_database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

Or via command line:

```bash
psql "postgres://user:password@localhost/dbname" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Environment Configuration

Add to your `.env` file:

```bash
POSTGRES_URL=postgres://cocoindex:cocoindex@localhost:5432/cocoindex
```

## SQLite with sqlite-vec

### Installation

SQLite vector support requires the `sqlite-vec` extension:

```bash
# Install via pip (includes sqlite-vec)
pip install sqlite-vec
```

**Note for macOS users**: The default Python may not support extensions. Use Homebrew Python:

```bash
# Install Homebrew Python
brew install python

# Use Homebrew Python for your project
which python3  # Should show /usr/local/bin/python3 or /opt/homebrew/bin/python3
```

### Usage in Code

```python
from cocoindex.connectors import sqlite

# Create connection (auto-loads sqlite-vec if available)
conn = sqlite.connect("./data.db", load_vec="auto")

# Or require sqlite-vec (fails if not available)
conn = sqlite.connect("./data.db", load_vec=True)

# Or disable vector support
conn = sqlite.connect("./data.db", load_vec=False)
```

### Environment Configuration

Add to your `.env` file:

```bash
# SQLite database path
SQLITE_DB=./cocoindex.db
```

## LanceDB

### Installation

```bash
pip install lancedb
```

### Local Storage

No setup needed for local storage. Just specify a directory:

```python
import cocoindex.asyncio as coco_aio
from cocoindex.connectors import lancedb

conn = await lancedb.connect_async("./lancedb_data")
```

### Cloud Storage (S3, GCS, Azure)

For cloud storage, configure credentials according to your provider:

**AWS S3:**

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1

# Or use AWS credentials file
aws configure
```

```python
# Connect to S3 bucket
conn = await lancedb.connect_async("s3://your-bucket/lancedb_data")
```

**Google Cloud Storage:**

```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

```python
# Connect to GCS bucket
conn = await lancedb.connect_async("gs://your-bucket/lancedb_data")
```

## Qdrant

### Using Docker (Recommended)

```bash
# Create docker-compose.yml
cat > docker-compose.yml <<EOF
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: cocoindex-qdrant
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

volumes:
  qdrant_data:
EOF

# Start Qdrant
docker-compose up -d

# Verify it's running
curl http://localhost:6333/
```

### Using Qdrant Cloud

Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) and get your cluster URL and API key:

```python
from cocoindex.connectors import qdrant

client = qdrant.create_client(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
    prefer_grpc=True,
)
```

### Environment Configuration

Add to your `.env` file:

```bash
# Local Qdrant
QDRANT_URL=http://localhost:6333

# Or Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

## Common Issues

### PostgreSQL Connection Refused

```bash
# Check if PostgreSQL is running
docker-compose ps
# or
systemctl status postgresql

# Check if port is accessible
nc -zv localhost 5432
```

### pgvector Extension Not Found

```bash
# Verify extension is installed
psql -U postgres -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

# If not found, install pgvector
# See: https://github.com/pgvector/pgvector#installation
```

### SQLite Extensions Not Loading

```python
# Check if extensions are supported
import sqlite3
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)  # Should not raise an error

# If error, you need Python built with extension support
# On macOS: use Homebrew Python
# On Linux: ensure Python was built with --enable-loadable-sqlite-extensions
```

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
curl http://localhost:6333/

# Check Docker logs
docker logs cocoindex-qdrant

# Try HTTP instead of gRPC
client = qdrant.create_client(url="http://localhost:6333", prefer_grpc=False)
```

## Multi-Database Setup Example

For a complete setup with both PostgreSQL and Qdrant:

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: cocoindex-postgres
    environment:
      POSTGRES_USER: cocoindex
      POSTGRES_PASSWORD: cocoindex
      POSTGRES_DB: cocoindex
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    container_name: cocoindex-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  qdrant_data:
```

```bash
# .env
POSTGRES_URL=postgres://cocoindex:cocoindex@localhost:5432/cocoindex
QDRANT_URL=http://localhost:6333
```

## See Also

- [Connectors Reference](./connectors.md) - Detailed connector documentation
- [Patterns Reference](./patterns.md) - Example usage in pipelines
