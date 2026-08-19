"""
Probe what the embedding API actually allows for your key.

    python -m scripts.check_limits

Finds the largest batch that succeeds and measures sustained throughput, so
EMBED_BATCH_SIZE can be set from evidence rather than guesswork. Documented
tier limits are a starting point; this is what your key does in practice.

Deliberately small: about a dozen calls, a few thousand tokens. It will not
meaningfully dent a daily quota.
"""

import logging
import sys
import time

from src.config import get_config
from src.rag.embeddings import EmbeddingService

# ~250 tokens, close to a real chunk at CHUNK_SIZE=1000.
SAMPLE = (
    "A PersistentVolumeClaim is a request for storage by a user. It is similar "
    "to a Pod: Pods consume node resources and PVCs consume PV resources. Pods "
    "can request specific levels of resources such as CPU and memory, whereas "
    "claims can request specific size and access modes. " * 4
)

BATCH_SIZES = [1, 10, 25, 50, 100, 250]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    try:
        config = get_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    service = EmbeddingService(
        api_key=config.gemini.api_key,
        model_name=config.gemini.embedding_model,
        dimensions=config.gemini.embedding_dim,
    )

    print(f"model      : {config.gemini.embedding_model}")
    print(f"dimensions : {config.gemini.embedding_dim}")
    print(f"sample     : ~{len(SAMPLE) // 4} tokens per text\n")

    print(f"{'batch':>6}  {'~tokens':>8}  {'seconds':>8}  result")
    print("-" * 52)

    largest_ok = 0
    for size in BATCH_SIZES:
        texts = [SAMPLE] * size
        approx_tokens = size * (len(SAMPLE) // 4)

        start = time.time()
        try:
            vectors = service.embed_texts(texts)
            elapsed = time.time() - start
            if len(vectors) != size:
                print(f"{size:>6}  {approx_tokens:>8}  {elapsed:>8.1f}  "
                      f"returned {len(vectors)}, expected {size}")
                break
            largest_ok = size
            print(f"{size:>6}  {approx_tokens:>8}  {elapsed:>8.1f}  ok")
        except Exception as exc:  # noqa: BLE001 - the error text is the finding
            elapsed = time.time() - start
            detail = str(exc).replace("\n", " ")[:110]
            print(f"{size:>6}  {approx_tokens:>8}  {elapsed:>8.1f}  FAILED: {detail}")
            break

    print()
    if largest_ok:
        print(f"Largest batch that succeeded: {largest_ok}")
        print(f"Suggested EMBED_BATCH_SIZE  : {max(1, largest_ok // 2)}  "
              f"(half, leaving headroom for longer real chunks)")
    else:
        print("No batch size succeeded - check GOOGLE_API_KEY and the model name.")

    print("\nNote: a 429 above is the useful result, not a failure of this script.")
    print("Its message usually names the quota that was hit.")


if __name__ == "__main__":
    main()
