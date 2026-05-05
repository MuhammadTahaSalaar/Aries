# Infrastructure & Docker Conventions

## Docker Services (target state)

| Service            | Image / Stack         | Port  |
| ------------------ | --------------------- | ----- |
| `postgres`         | PostgreSQL 16         | 5432  |
| `kafka`            | KRaft (no ZooKeeper)  | 9092  |
| `kafka-ui`         | kafka-ui              | 8090  |
| `elasticsearch`    | ES 8 (single-node)    | 9200  |
| `redis`            | Redis 7               | 6379  |
| `neo4j`            | Neo4j 5               | 7474  |
| `minio`            | MinIO                 | 9000  |
| `mlflow`           | Custom (MLOps/)       | 5000  |
| `dashboard`        | Next.js               | 3000  |
| `go_service`       | Go binary             | 8080  |
| `fastapi_service`  | Python 3.11-slim      | 8000  |
| `wazuh-manager`    | Wazuh 4.x             | 1514  |

## Dockerfile Best Practices

- Multi-stage builds: build stage + minimal runtime stage.
- Non-root user in runtime stage.
- `HEALTHCHECK` instruction in every Dockerfile.
- Pin base image versions (e.g., `python:3.11-slim`, `golang:1.25-alpine`,
  `node:24-alpine`).
- Copy dependency files first, install deps, then copy source (maximise cache).

## Environment Variables

- All config via env vars (12-factor). **Never hardcode secrets.**
- `.env.example` (committed) as template. `.env` is gitignored.
- `mlruns/` directory used for local MLflow tracking.

## MLOps Stack

- MLflow tracking server backed by PostgreSQL + MinIO.
- Artifact store: `s3://mlflow-bucket` (MinIO).
- Access MLflow UI at `http://localhost:5000`.

## NTFS / Bind Mount Workarounds

When running on NTFS (Windows host):

- **PostgreSQL**: use named Docker volume (not bind mount to `/data`).
- **Wazuh scripts**: use `docker cp` + `chmod 750` (not bind mounts).
