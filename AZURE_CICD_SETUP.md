# Azure CI/CD Setup

This repository has a GitHub Actions workflow at:

```text
.github/workflows/azure-backend-cicd.yml
```

The workflow runs on pushes to:

```text
feat/deployment_fixes
```

It can also be started manually from the GitHub Actions tab.

## What The Workflow Does

1. Checks out the repository.
2. Restores the local Whisper model from a private zip URL if the model is not tracked in git.
3. Logs in to Azure.
4. Logs in to Azure Container Registry.
5. Builds the Docker image.
6. Pushes these image tags:

```text
sinlearnbackendacr.azurecr.io/sinhala-ed-assistant:latest
sinlearnbackendacr.azurecr.io/sinhala-ed-assistant:<short-commit-sha>
```

7. Deployment to Azure Container Apps is currently manual until the Azure tenant allows a service principal or federated credentials.

## Required GitHub Secrets

Add these in GitHub:

```text
Repository -> Settings -> Secrets and variables -> Actions -> New repository secret
```

Required:

```text
ACR_USERNAME
ACR_PASSWORD
WHISPER_MODEL_ZIP_URL
```

Later, when the Container App exists, the app itself will also need runtime secrets such as:

```text
DATABASE_URL
JWT_SECRET_KEY
GEMINI_API_KEY
GOOGLE_API_KEY
GOOGLE_API_KEY_V2
GEMINI_LIGHT_API_KEY
MAIL_USERNAME
MAIL_PASSWORD
FRONTEND_URL
```

## Create ACR_USERNAME And ACR_PASSWORD

Run this locally after `az login`:

```powershell
az acr credential show --name sinlearnbackendacr
```

Add the values to GitHub Actions secrets:

```text
ACR_USERNAME = username
ACR_PASSWORD = passwords[0].value
```

## Create WHISPER_MODEL_ZIP_URL

The Whisper model is currently local and is not tracked by git. GitHub Actions cannot access files that are only on your laptop.

Zip this folder:

```text
app/models/whisper-sinhala-accent-model
```

Upload it to private storage, such as Azure Blob Storage, and create a time-limited SAS download URL. Save that URL as:

```text
WHISPER_MODEL_ZIP_URL
```

The zip must contain:

```text
model.safetensors
config.json
generation_config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
normalizer.json
special_tokens_map.json
added_tokens.json
```

## One-Time Infrastructure Requirement

The workflow updates an existing Azure Container App. Before deployment can succeed, create this Container App once:

```text
sinlearn-backend-api
```

in:

```text
sinlearn-backend-rg
```

After the Container App exists, every push to `feat/deployment_fixes` can build and push a fresh Docker image automatically.

Until Azure login from GitHub Actions is configured, update the Container App manually to use the latest image:

```powershell
az containerapp update `
  --name sinlearn-backend-api `
  --resource-group sinlearn-backend-rg `
  --image sinlearnbackendacr.azurecr.io/sinhala-ed-assistant:latest
```
