FROM python:3.11-slim

WORKDIR /app

# Install deps first for Docker layer caching
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY documents/ ./documents/
COPY scripts/ ./scripts/
# api.py was missing from this list, so the HTTP API could never run in a
# container - only the MCP server could. It was developed against a local
# `uvicorn api:app`, which hid that the image did not contain it.
COPY api.py ./

RUN mkdir -p data/chroma_db

# Run as non-root user (security best practice)
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

# Default: the HTTP API, because that is what a Kubernetes Service can route to
# and what a readiness probe can call. The MCP server is stdio-based and has no
# port to probe, so it cannot be the default for a deployed workload.
#
# To run the MCP server instead, override the command:
#   command: ["python", "-m", "src.mcp_server.server"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
