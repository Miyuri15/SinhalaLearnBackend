# FastAPI API Setup Guide

Follow these steps to set up and run the Sinhala ED API locally.

---

## 1. Prerequisites

- Python 3.9+
- pip

Example (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential
```

## 2. Create and activate a virtual environment

```bash
# Create venv next to pyproject.toml
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows cmd)
.venv\Scripts\activate
```

## 3. Upgrade pip and packaging helpers

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install the project and dependencies

Install the project (reads pyproject.toml):

```bash
python -m pip install .
# for editable install during development:
python -m pip install -e ".[dev]"
```

## 5. Set Up PostgreSQL Database with pgvector

### Start Docker Container

```bash
# Default port (5432)
docker run -d \
  --name pgvector-db \
  -e POSTGRES_PASSWORD=your-password \
  -p 5432:5432 \
  ankane/pgvector

# Or custom port (5433) - Windows Command Prompt
docker run -d --name pgvector-db -e POSTGRES_PASSWORD=your-password -p 5433:5432 ankane/pgvector
```

### Configure Database

Connect to the container and run:

```bash
docker exec -it pgvector-db psql -U postgres
```

Then execute:

```sql
CREATE USER test_user WITH PASSWORD 'your-password';
CREATE DATABASE "SinhalaDB" OWNER test_user;
GRANT ALL PRIVILEGES ON DATABASE "SinhalaDB" TO test_user;
\c "SinhalaDB"
CREATE EXTENSION IF NOT EXISTS vector;
```

### Environment Variable

Add to `.env`:

```
DATABASE_URL=postgresql://test_user:your-password@localhost:5432/SinhalaDB
ADMIN_BOOTSTRAP_TOKEN=change-this-one-time-admin-bootstrap-token
```

**Note:** Update port to 5433 if using custom port option.

## 6. Run Migrations

```bash
alembic upgrade head
```

The migration `e1f2a3b4c5d6_promote_admin_user.py` promotes `admin@sinhalalearn.com` to `role = 'admin'` if that user already exists. If the user does not exist yet, the migration still succeeds but updates zero rows.

## 7. Create the First Admin

Use the bootstrap endpoint once, after `ADMIN_BOOTSTRAP_TOKEN` is configured:

```http
POST /api/v1/auth/bootstrap-admin
```

Request body:

```json
{
  "email": "admin@sinhalalearn.com",
  "full_name": "Admin User",
  "password": "your-admin-password",
  "bootstrap_token": "change-this-one-time-admin-bootstrap-token"
}
```

Behavior:

- If no admin exists, this creates or promotes the user as `role = "admin"`.
- If `admin@sinhalalearn.com` already exists, it promotes that user and updates the password from the request.
- If any admin already exists, the endpoint returns `409` and cannot be used again.
- Admins log in normally with `POST /api/v1/auth/signin`.

## 8. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

The API is available at http://localhost:8000 and interactive docs at http://localhost:8000/docs

---

## Quick re-activation (later)

Each time you open a new shell, activate the venv:

```bash
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate
```
