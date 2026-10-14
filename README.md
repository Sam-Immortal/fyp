# ScamGuard

**Deepfake Audio Detection & AI Scam Alert System**

A multimodal mobile gateway that intercepts live telecommunication streams (VoIP/SMS) and flags AI-generated voice cloning and conversational phishing in real time.

## Repository layout

```
.
├── backend/          FastAPI service (Python 3.11+, Poetry)
│   ├── app/          Application code (main, core, api)
│   ├── tests/        Pytest + HTTPX async test suite
│   ├── Dockerfile    Multi-stage python:3.11-slim image
│   └── pyproject.toml
│
├── android-app/      Android client (Kotlin, Jetpack Compose)
│   ├── app/          Module source (UI, service, receiver, tests)
│   ├── build.gradle.kts
│   └── settings.gradle.kts
│
└── infra/            CI/CD and orchestration
    ├── docker-compose.yml         Local dev stack
    ├── docker-compose.jenkins.yml CI stack (Jenkins + dind + registry + MinIO)
    ├── jenkins/
    │   ├── Dockerfile             Custom Jenkins image
    │   └── plugins.txt            Pre-installed plugins
    ├── Jenkinsfile                Declarative pipeline (10 stages)
    └── README.md                  Boot/usage instructions
```

## Quick start

### Backend (local Python)

```bash
cd backend
poetry install --with dev
cp .env.example .env
poetry run uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs>.

### Backend + infra (full local stack)

```bash
cd infra
docker compose up -d --build
```

### CI/CD (Jenkins + Docker-in-Docker)

```bash
cd infra
docker compose -f docker-compose.jenkins.yml up -d --build
```

Then open <http://localhost:8080> and follow `infra/README.md`.

## Tech stack

| Tier      | Stack                                                                 |
|-----------|-----------------------------------------------------------------------|
| Android   | Kotlin 2.0+, Jetpack Compose, Retrofit 2, OkHttp 4, Kotlinx Serialization |
| Backend   | Python 3.11, FastAPI 0.110+, Pydantic v2, PyJWT, Taskiq, Redis, asyncpg |
| Database  | PostgreSQL 16 (SQLAlchemy 2.0 async) + Redis 7.2 cache                |
| Storage   | MinIO (S3-compatible, self-hosted)                                    |
| CI/CD     | Jenkins Declarative Pipeline + Docker-in-Docker + Docker Compose      |
| Tooling   | Poetry, Ruff, Pytest, HTTPX, Ktlint, JUnit 5, Mockk                   |

## Pipeline stages

1. Checkout
2. Lint Backend (`ruff`)
3. Test Backend (`pytest`)
4. Build Backend Image (dind)
5. Push to local registry
6. Lint Android (`ktlint`)
7. Test Android (`testDebugUnitTest`)
8. Build Android APK (`assembleDebug`)
9. Upload APK to MinIO
10. Cleanup
