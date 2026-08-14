FROM python:3.11-slim

WORKDIR /app

# Install deps first for Docker layer caching
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY documents/ ./documents/
COPY scripts/ ./scripts/

RUN mkdir -p data/chroma_db

# Run as non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

# Default: run the MCP server
CMD ["python", "-m", "src.mcp_server.server"]
