# ScamGuard Infrastructure

Container orchestration, Jenkins CI/CD, and shared deployment artifacts.

## Folders

- `docker-compose.yml` - Local dev stack: API, worker, Postgres, Redis, MinIO.
- `docker-compose.jenkins.yml` - CI stack: Jenkins + Docker-in-Docker + local registry + MinIO.
- `jenkins/Dockerfile` - Custom Jenkins image with Docker CLI + Poetry pre-installed.
- `jenkins/plugins.txt` - Plugins installed at image build time.
- `Jenkinsfile` - Declarative pipeline: lint, test, build, push, upload APK.

## Boot Jenkins locally (Windows / Docker Desktop)

```powershell
cd infra
docker compose -f docker-compose.jenkins.yml up -d --build
```

| Service     | Port  | Notes                                       |
|-------------|-------|---------------------------------------------|
| Jenkins     | 8080  | Unlock with `docker logs scamguard-jenkins` |
| Docker-in-Docker | 2376 | TLS-enabled daemon for builds          |
| Local Registry | 5000 | `registry:2` for backend images           |
| MinIO       | 9000  | S3-compatible artifact storage              |
| MinIO UI    | 9001  | `scamguard` / `scamguard-secret`            |
| Postgres    | 5432  | `scamguard` / `scamguard` / `scamguard`     |
| Redis       | 6379  | No auth in CI                               |

## Boot the dev stack

```powershell
cd infra
docker compose up -d --build
```

API: <http://localhost:8000/docs> &nbsp;&nbsp; MinIO: <http://localhost:9001>

## Configure Jenkins

1. Open <http://localhost:8080> and unlock with the initial admin password.
2. Add credentials (Manage Jenkins -> Credentials -> System -> Global):
   - `minio-access-key` (Secret text) = `scamguard`
   - `minio-secret-key` (Secret text) = `scamguard-secret`
   - GitHub credentials if using a webhook.
3. Create a Pipeline job pointing at this repository's `infra/Jenkinsfile`.
4. (Optional) Add a GitHub webhook to `http://<host>:8080/github-webhook/`.

## Triggering a build

Either push to the tracked branch or click **Build Now** in Jenkins.
