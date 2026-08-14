FROM python:3.11-slim

# Set working directory (like WORKDIR in Java Dockerfiles)
WORKDIR /app

# Copy dependency files first (Docker layer caching - same concept as Java)
COPY pyproject.toml requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY documents/ ./documents/

# Create data directory for ChromaDB
RUN mkdir -p data/chroma_db

# Expose MCP server (stdio transport, no port needed)
# For HTTP transport, uncomment:
# EXPOSE 8080

# Run the MCP server
CMD ["python", "-m", "src.mcp_server.server"]
