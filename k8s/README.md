# Running this on Kubernetes

Local cluster only — k3s via Rancher Desktop. Nothing here reaches a cloud, a
registry, or a managed database.

## Before anything: check which cluster you are pointed at

```bash
kubectl config current-context
```

**If that is not `rancher-desktop`, stop.** A `kubectl apply` goes wherever the
current context points, and a context list often contains clusters that are not
yours to deploy hobby workloads to — including production ones, one
`use-context` away.

So every command below passes `--context rancher-desktop` explicitly rather than
switching global state. Switching is a change that persists long after you have
stopped thinking about it; a flag is scoped to the command you are looking at.

```bash
kubectl --context rancher-desktop get nodes
```

## 1. Build the image

k3s under Rancher Desktop shares the Docker image store, so a local build is
visible to the cluster with no registry and no push:

```bash
docker build -t hybrid-rag-service:local .
```

The Deployment sets `imagePullPolicy: Never` to match. Without it Kubernetes
tries to pull `docker.io/library/hybrid-rag-service:local` and the pod sits in
`ImagePullBackOff`, which reads as a network problem and is not one.

## 2. Create the Secret

**There is deliberately no Secret manifest.** One holding a real key cannot be
committed, and one holding a placeholder is worse: `kubectl apply -f k8s/` would
overwrite a working key with `REPLACE_ME`, and the failure surfaces later as a
confusing API error rather than as a missing secret.

Create it from `.env`, without either printing it or writing it to a file:

```bash
KEY=$(grep '^GOOGLE_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'"' \r')
test -n "$KEY" || { echo "GOOGLE_API_KEY not found in .env"; exit 1; }

kubectl --context rancher-desktop create secret generic rag-secrets \
  --from-literal=GOOGLE_API_KEY="$KEY" \
  --dry-run=client -o yaml | kubectl --context rancher-desktop apply -f -
```

Verify the shape without revealing the value:

```bash
kubectl --context rancher-desktop get secret rag-secrets \
  -o go-template='{{range $k,$v := .data}}{{$k}}={{len $v}} base64 chars{{"\n"}}{{end}}'
```

A Secret is base64, not encryption — anyone with `get secrets` can read it. It is
here to demonstrate the mechanism, and the mechanism is *not in the image, not in
the repository's application config, injected at runtime*.

## 3. Apply

```bash
kubectl --context rancher-desktop apply -f k8s/
kubectl --context rancher-desktop get pods -w
```

Expect `rag-postgres-0` ready in ~15s and `rag-service` ready in ~30s. The
service starts before Postgres is ready and simply fails its readiness probe
until the database answers, which is the intended behaviour — no init container
or wait-for script needed.

## 4. Give it a corpus

The cluster database starts **empty**, and deliberately does not point at any
Postgres on the host. A host container may hold a corpus, an evaluation set and
another project's data; a deployment experiment must not be one
`kubectl delete` away from it.

**Option A — copy existing embeddings (free).** If a host database already has an
ingested corpus, the embeddings exist and do not need paying for again:

```bash
docker exec <host-postgres> pg_dump -U postgres -d ragdb \
    --data-only --no-owner --table=chunks --table=embedding_cache \
  | kubectl --context rancher-desktop exec -i rag-postgres-0 -- psql -U raguser -d ragdb
```

`pg_dump` only reads, so the host is untouched. This moved 784 chunks and 337
cached embeddings in a few seconds and saved ~784 embedding API calls.

**Option B — ingest from scratch (costs quota).** The image contains
`documents/`, so the pod can build its own index:

```bash
curl -X POST -H "Host: rag.localhost" http://<traefik-ip>/ingest
```

## 5. Use it

Traefik's external IP:

```bash
kubectl --context rancher-desktop get svc -n kube-system traefik \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

```bash
# readiness / corpus size
curl -H "Host: rag.localhost" http://<traefik-ip>/stats

# ask something
curl -X POST -H "Host: rag.localhost" -H "Content-Type: application/json" \
  -d '{"question":"How do I set memory limits and requests for a container?","top_k":3}' \
  http://<traefik-ip>/query
```

Or skip the ingress entirely:

```bash
kubectl --context rancher-desktop port-forward svc/rag-service 8000:8000
open http://localhost:8000/docs        # FastAPI's generated Swagger UI
```

## The two probes are not interchangeable

```
livenessProbe   GET /healthz   checks NOTHING but the process
readinessProbe  GET /stats     builds the pipeline, queries the corpus
```

Kubernetes answers a **liveness** failure by restarting the container and a
**readiness** failure by removing the pod from the Service. So a liveness probe
that touches the database restarts every healthy pod during a database blip,
turning a short outage into a crash loop that outlives its cause. Readiness is
the probe allowed to check dependencies, because failing it is not fatal.

`/healthz` was added for exactly this reason — the only existing endpoint that
could serve as a health check was `/stats`, and using it for both would have been
the mistake above.

## Tearing it down

```bash
kubectl --context rancher-desktop delete -f k8s/
kubectl --context rancher-desktop delete secret rag-secrets
```

The PersistentVolumeClaim survives on purpose, so a redeploy keeps its corpus. To
remove the data as well:

```bash
kubectl --context rancher-desktop delete pvc data-rag-postgres-0
```

## Two bugs this deployment exposed

Both had been latent for as long as the API existed, and neither was visible
locally:

**`api.py` was never in the image.** The Dockerfile copied `src/`, `documents/`
and `scripts/` — not `api.py`. `fastapi` and `uvicorn` were also absent from
`requirements.txt`; the file's own docstring said `pip install fastapi uvicorn`.
So the HTTP API had never once run in a container. Developing it with
`uvicorn api:app` at a shell hid that completely.

**There was no liveness endpoint.** Only `/stats`, which hits the database. See
above for why that matters.

The general point: **running something locally does not test that it can be
deployed.** Containerising it is what asked whether the API was in the image, and
scheduling it is what asked what "healthy" means.
