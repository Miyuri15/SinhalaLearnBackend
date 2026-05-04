# Azure Deployment Report for Sinhala Learn

## 1. Introduction

Sinhala Learn is an AI-powered educational assistant designed to support Sinhala-medium students and teachers through document processing, resource-based question answering, voice-based interaction, and automatic answer evaluation. The system is implemented as a FastAPI backend with a Flutter mobile client. The backend provides REST APIs for authentication, chat sessions, resource upload, OCR processing, retrieval-augmented generation, voice question answering, rubric management, and answer evaluation.

The Azure deployment was introduced to move the backend from a local development environment into a cloud-hosted, containerized runtime. The main goal of the deployment was to make the API accessible to the mobile application, provide a repeatable release process, separate application configuration from source code, and prepare the system for future scaling as more students, teachers, and classroom resources are added.

The deployment architecture uses Docker for packaging the FastAPI application, Azure Container Registry for storing built images, and Azure Container Apps as the intended runtime platform for the backend API. The backend connects to a PostgreSQL database with pgvector support for structured data, resource metadata, and semantic vector retrieval. External AI services, including Gemini APIs and a fine-tuned Whisper model, are used by the application for language processing, embeddings, speech recognition, and feedback generation.

## 2. Deployment Objectives

The Azure deployment was carried out with the following objectives:

1. Package the backend as a portable container image so that the same application can run consistently in local development and cloud environments.
2. Store built backend images in Azure Container Registry under versioned tags.
3. Support automated build and image push using GitHub Actions.
4. Keep sensitive information such as database credentials, API keys, JWT secrets, mail passwords, and model download URLs outside the repository.
5. Host the backend in Azure Container Apps so the API can be accessed by the mobile application through a stable cloud endpoint.
6. Prepare the backend for horizontal scaling by keeping persistent application data in external services such as PostgreSQL and object storage.
7. Support AI-specific dependencies such as PyTorch CPU wheels, FFmpeg, Tesseract OCR, Poppler, and the Sinhala Whisper model inside the deployed container.

## 3. High-Level Architecture

The deployed Sinhala Learn system follows a cloud-backed client-server architecture.

```text
Flutter Mobile Application
        |
        | HTTPS API requests
        v
Azure Container Apps
Sinhala Learn FastAPI Backend
        |
        | SQL and vector search
        v
PostgreSQL with pgvector
        |
        | AI service calls
        v
Gemini APIs / External AI Providers

Azure Container Registry stores the backend Docker image.
Private model storage provides the Whisper model during CI/CD.
```

The mobile application communicates with the FastAPI backend. The backend handles authentication, file upload, OCR, speech-to-text, RAG, answer generation, grading, and feedback. PostgreSQL stores users, sessions, resources, extracted text, chunks, embeddings, rubrics, evaluation sessions, and message history. The vector search layer uses pgvector to compare question embeddings against document and chunk embeddings. AI model calls are made through configured API keys, while the local Whisper model is loaded inside the backend container for Sinhala speech recognition.

## 4. Backend Runtime Design

The backend is implemented with FastAPI and starts through Uvicorn on port 8000. The Docker container exposes port 8000, and Azure Container Apps routes external traffic to this internal application port.

At startup, the backend performs two important actions:

1. It verifies or creates database tables using SQLAlchemy metadata.
2. It loads the Sinhala Whisper model through the application-level Whisper loader.

The startup model loading step is important because it makes the first voice request faster after the container is ready. However, it also increases cold-start time and memory usage, because each running container replica must load its own copy of the model.

The backend includes API routers for:

- Authentication and users
- Chat sessions and messages
- Resource uploads and resource processing
- OCR, chunking, and embedding generation
- Text-based question answering and summaries
- Voice question answering
- Rubrics and answer evaluation
- WebSocket-based communication paths

Configuration is provided using environment variables. Important runtime values include:

- `DATABASE_URL`
- `JWT_SECRET_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_API_KEY_V2`
- `GEMINI_API_KEY`
- `GEMINI_LIGHT_API_KEY`
- `MAIL_USERNAME`
- `MAIL_PASSWORD`
- `FRONTEND_URL`
- Optional Whisper model path configuration

This configuration approach allows the same image to be used across development, testing, and production while changing only the environment variables in Azure.

## 5. Containerization

The backend is containerized using a multi-stage Dockerfile. The first stage uses `python:3.12-slim` as a builder image and installs dependencies into a virtual environment using `uv`. CPU versions of PyTorch and TorchVision are installed from the PyTorch CPU wheel index. This decision is suitable for Azure CPU-based container hosting because it avoids unnecessary CUDA packages and keeps the image more compatible with standard cloud containers.

The final runtime image installs the system-level packages required by the AI and document-processing pipeline:

- `ffmpeg` for audio conversion and speech input handling
- `libsndfile1` for audio file reading
- `tesseract-ocr` for OCR support
- `poppler-utils` for PDF conversion and processing
- `libgl1`, `libglib2.0-0`, and `libgomp1` for computer vision and ML dependencies
- `libpq-dev` for PostgreSQL connectivity

The image also creates a non-root user and runs the application as that user. This improves the security posture of the container because the API process does not run with root privileges. The application code is copied into `/code`, and directories such as `/code/uploads`, `/code/models`, and `/code/logs` are created for runtime use.

The final command starts the application using:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This makes the FastAPI application reachable from the Azure Container Apps ingress layer.

## 6. Azure Services Used

The deployment uses or is prepared to use the following Azure services:

### Azure Container Registry

Azure Container Registry stores the backend Docker image. The configured registry is:

```text
sinlearnbackendacr.azurecr.io
```

The backend image name is:

```text
sinhala-ed-assistant
```

The CI/CD pipeline pushes two image tags:

```text
sinlearnbackendacr.azurecr.io/sinhala-ed-assistant:latest
sinlearnbackendacr.azurecr.io/sinhala-ed-assistant:<short-commit-sha>
```

The `latest` tag is useful for quick deployment, while the short commit SHA tag gives traceability. If a deployment has a problem, the SHA tag helps identify exactly which source version produced the image.

### Azure Container Apps

Azure Container Apps is the intended backend runtime. The documented backend Container App is:

```text
sinlearn-backend-api
```

It is created in the resource group:

```text
sinlearn-backend-rg
```

The Container App pulls the backend image from Azure Container Registry and runs it as a containerized FastAPI API. Azure Container Apps is suitable for this project because it supports HTTP ingress, environment-based configuration, container revisions, and horizontal scaling without requiring the team to manage Kubernetes directly.

### PostgreSQL with pgvector

The backend uses PostgreSQL as the main persistent data store. The application also uses pgvector-compatible vector columns for document-level and chunk-level embeddings. This allows the same database to store both normal relational records and semantic search vectors.

Stored data includes:

- Users and authentication-related records
- Chat sessions and messages
- Uploaded resource metadata
- Extracted OCR text
- Resource chunks
- Document embeddings and chunk embeddings
- Rubrics and marking schemas
- Evaluation sessions and answer feedback

Vector search is used when a student asks a question. The backend compares the query embedding against stored resource embeddings and resource chunk embeddings to retrieve the most relevant educational content.

### Private Model Storage

The Sinhala Whisper model is not treated as normal source code because it is too large and should not be committed directly to Git. The CI/CD pipeline expects the model to be available either in the repository or through a private ZIP download URL.

The configured secret for this is:

```text
WHISPER_MODEL_ZIP_URL
```

The recommended approach is to upload the model folder to private storage such as Azure Blob Storage and generate a time-limited SAS URL. GitHub Actions downloads the model during the build, extracts it, verifies that `model.safetensors` exists, and then includes the model files in the Docker image.

## 7. CI/CD Workflow

The repository includes a GitHub Actions workflow named:

```text
Azure Backend CI/CD
```

The workflow file is:

```text
.github/workflows/azure-backend-cicd.yml
```

The workflow runs on pushes to:

```text
feat/deployment_fixes
Feat/deployment_fixes
```

It can also be started manually from the GitHub Actions interface using workflow dispatch.

The CI/CD workflow performs the following steps:

1. Checks out the repository.
2. Validates that required GitHub secrets are configured.
3. Restores the Whisper model if it is not present in the repository.
4. Logs in to Azure Container Registry.
5. Creates image tags using `latest` and the short Git commit SHA.
6. Sets up Docker Buildx.
7. Builds the Docker image from the root Dockerfile.
8. Pushes the image to Azure Container Registry.

The required GitHub Actions secrets are:

```text
ACR_USERNAME
ACR_PASSWORD
WHISPER_MODEL_ZIP_URL
```

The application runtime also needs production secrets configured in Azure Container Apps, including database credentials, Gemini API keys, JWT configuration, mail credentials, and frontend URL values.

## 8. Current Deployment Status and What Happened During Deployment

During the deployment work, the backend was prepared for Azure by containerizing the FastAPI application and adding a GitHub Actions workflow to build and push the backend image to Azure Container Registry.

Several deployment-specific issues had to be handled:

1. The backend depends on heavy AI and document-processing libraries. The Dockerfile was updated to install CPU-compatible PyTorch packages and required system packages such as FFmpeg, Tesseract, Poppler, and OpenCV-related libraries.
2. The Whisper speech recognition model is local and not suitable for direct Git tracking. The CI/CD workflow was therefore designed to restore the model from a private ZIP URL before building the Docker image.
3. The application must run in CPU-based cloud containers. The Whisper loader was designed to use FP16 only when CUDA is available and to keep CPU execution in FP32. This avoids precision-related runtime problems in Azure CPU containers.
4. Sensitive configuration values were separated from source code and moved to environment variables and GitHub or Azure secrets.
5. The workflow currently automates the image build and push to Azure Container Registry. The final Azure Container Apps update is documented as a manual step until the Azure tenant permits service principal login or federated credentials for GitHub Actions.

At the current stage, every push to the deployment branch can produce and publish a new backend image. The Container App can then be updated manually to use the newest image:

```text
az containerapp update --name sinlearn-backend-api --resource-group sinlearn-backend-rg --image sinlearnbackendacr.azurecr.io/sinhala-ed-assistant:latest
```

Once Azure login from GitHub Actions is enabled through a service principal or federated identity, this manual step can be added to the workflow so deployment becomes fully automated.

## 9. How the Deployed System Works

When the backend is deployed to Azure Container Apps, the application flow works as follows:

1. The student or teacher uses the Flutter mobile application.
2. The mobile app sends HTTPS requests to the deployed FastAPI backend endpoint.
3. Azure Container Apps receives the request through HTTP ingress and forwards it to the running backend container on port 8000.
4. FastAPI routes the request to the correct module, such as authentication, resource upload, text question answering, voice question answering, or answer evaluation.
5. If the request needs persistent data, the backend opens a PostgreSQL session using `DATABASE_URL`.
6. If the request involves uploaded learning resources, the backend stores resource metadata, extracts text using PDF/OCR pipelines, chunks the text, generates embeddings, and saves the chunks and vectors in PostgreSQL.
7. If the request is a text question, the backend generates an embedding for the question, retrieves relevant document and chunk records through pgvector and BM25-style retrieval, builds a context, and calls the generation model to produce an answer.
8. If the request is a voice question, the backend converts the uploaded audio to a suitable format if required, transcribes it using the local Whisper model, optionally normalizes the Sinhala text, and passes the text into the same retrieval and generation pipeline.
9. If the request is answer evaluation, the backend uses OCR, semantic matching, rubric rules, and Gemini-based feedback generation to produce marks and explanations.
10. The response is sent back to the mobile app as JSON, and the mobile app displays the answer, summary, transcript, grading result, or feedback to the user.

This design keeps the mobile application lightweight. The AI processing, data retrieval, authentication, and evaluation workflows are centralized in the backend, making it easier to update models and improve the system without reinstalling the mobile app.

## 10. Scaling Design

The Azure deployment is designed around horizontal scaling. Instead of running one large server, Azure Container Apps can run multiple replicas of the same backend container. When traffic increases, Azure can start more replicas. When traffic decreases, it can reduce the number of active replicas depending on the configured minimum and maximum replica values.

The backend is suitable for this model because most persistent state is stored outside the application process:

- User records are stored in PostgreSQL.
- Chat and message history are stored in PostgreSQL.
- Resource metadata is stored in PostgreSQL.
- Extracted text and embeddings are stored in PostgreSQL.
- The container image is stored in Azure Container Registry.
- Sensitive runtime configuration is stored as Azure/GitHub secrets and environment variables.

Because the backend container can be recreated from the same image and configuration, Azure can add or replace replicas without rebuilding the application.

### API Scaling

Azure Container Apps can scale the FastAPI backend based on HTTP traffic. For example, when many students ask questions or upload resources at the same time, additional replicas can be started to distribute the request load. Each replica runs the same Uvicorn/FastAPI application and connects to the same database.

The main scaling benefit is that normal API requests, authentication, metadata access, and generation requests can be distributed across replicas. This improves availability and reduces the chance that one container becomes the only bottleneck.

### Database Scaling

The database is scaled independently from the API. PostgreSQL handles persistent data and vector search. As the number of users and uploaded documents grows, database performance becomes important because retrieval, vector comparison, and evaluation history depend on it.

The database can be scaled by:

- Increasing CPU, memory, or storage of the PostgreSQL server.
- Adding appropriate indexes for frequently queried fields.
- Using pgvector indexes for faster embedding search.
- Monitoring slow queries and vector search latency.
- Keeping document chunks at a reasonable size to avoid unnecessary vector-search cost.

The application currently uses SQLAlchemy with `NullPool`. This means database connections are opened and closed without long-lived pooling inside the application process. This can be useful in serverless/container environments because it avoids stale pooled connections, but database connection limits must be monitored when many replicas are active.

### AI Workload Scaling

The AI workload has different scaling behavior from normal API requests.

The Whisper model is loaded inside each backend replica. This means voice-based requests can be distributed across replicas, but each replica needs enough memory to load the model. Scaling from zero or starting a new replica may take longer because the model must load during startup.

Gemini-based embedding and generation calls depend on external API quotas, latency, and rate limits. Adding more backend replicas can increase parallel request capacity, but it can also increase the rate at which external API limits are reached. For that reason, production scaling should include request throttling, retry logic, and usage monitoring.

Document OCR and answer evaluation are heavier than simple chat requests. These tasks may consume more CPU and memory because they process PDFs, images, embeddings, and model calls. For larger deployments, these workloads can be moved to background workers or queues so that user-facing API requests remain responsive.

### Storage Scaling

The current resource upload service saves uploaded files under a local `uploads` directory inside the running application environment. This is acceptable for local development and limited demonstrations, but it is not ideal for multi-replica cloud scaling because each replica has its own local filesystem.

For production-scale deployment, uploaded files should be stored in shared storage such as Azure Blob Storage or mounted Azure Files. The database should store the shared storage URL or object key instead of relying only on a container-local path. This allows any replica to retrieve and process the same uploaded file.

The Whisper model is already handled through private model storage during CI/CD. This approach avoids storing large model files directly in Git and keeps the build process reproducible.

## 11. Reliability and Security Considerations

The deployment improves reliability and security in several ways:

1. The backend runs from a controlled Docker image, reducing differences between local and production environments.
2. The container runs as a non-root user.
3. Secrets are not hard-coded into the application source code.
4. The Docker image can be versioned using commit SHA tags.
5. The backend uses CORS configuration so the deployed frontend origin can be explicitly allowed through `FRONTEND_URL`.
6. JWT authentication protects private user, resource, chat, and evaluation endpoints.
7. The database stores persistent state outside the container, so application containers can be restarted or replaced.

For a stronger production setup, the following improvements are recommended:

- Complete GitHub Actions to Azure Container Apps deployment using federated identity.
- Store uploaded resources in Azure Blob Storage or Azure Files.
- Configure Azure Application Insights or Log Analytics for monitoring.
- Add health, readiness, and startup probes for the Container App.
- Configure minimum replicas to reduce cold starts during demonstrations or classroom sessions.
- Configure maximum replicas to control cost and protect database/API quotas.
- Add explicit rate limiting for expensive AI endpoints.
- Use managed identities where possible instead of static credentials.

## 12. Benefits of the Azure Deployment

The Azure deployment provides several benefits for the Sinhala Learn research project:

1. It makes the backend accessible to the mobile application from outside the local development machine.
2. It provides a repeatable container-based deployment model.
3. It supports AI-specific dependencies that are difficult to install manually on every server.
4. It separates code, configuration, secrets, model files, and runtime infrastructure.
5. It prepares the project for real classroom usage by allowing the API layer to scale horizontally.
6. It provides traceability through Docker image tags based on Git commits.
7. It allows future production improvements such as automated deployments, monitoring, object storage, and controlled autoscaling.

## 13. Conclusion

The Sinhala Learn Azure deployment converts the research prototype into a cloud-deployable system. The backend is packaged as a Docker image, pushed to Azure Container Registry through GitHub Actions, and prepared to run on Azure Container Apps. The FastAPI backend connects to PostgreSQL with pgvector for persistent records and semantic retrieval, while AI features are provided through Gemini services and a fine-tuned Sinhala Whisper model.

The deployment currently automates the build and image publishing stages. The final Container Apps update remains manual until Azure identity configuration is enabled for GitHub Actions. Even at this stage, the architecture provides a strong foundation for research demonstration and future production deployment. Its scaling model is based on container replicas for the API, independent database scaling for persistent and vector data, private storage for large model files, and future shared object storage for uploaded resources.

Overall, the Azure deployment supports the project's goal of delivering a scalable Sinhala educational assistant that can serve students and teachers through mobile access, cloud-hosted AI processing, and centralized educational resource management.
