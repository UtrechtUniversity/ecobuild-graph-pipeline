# Orchestration

This repo orchestrates the other repos. It contains the docker-compose file that can be used to run different tasks like crawling papers, parsing PDFs, or just starting the document DB.

## Getting started

Clone this repo into a root directory called `AUTO-KG-LIT/`. Then from inside this root directory run:

```bash
chmod +x orchestration/utils/clone.sh
./orchestration/utils/clone.sh
```
This will clone all microservices into the correct directory (`AUTO-KG-LIT/microservices`) (if this fails you need to clone them manually)

### .env files
Each microservice should have its own .env file with the required secrets (e.g. DB credentials, API keys). These are mounted into the containers when you run the stack. Each repo contains a .env.template that tells you what the repo requires.


### building and starting the database

Now from `orchestration`, build and start the project using the command below. This will start the database and the Neo4J instance.

```bash
docker compose up -d document-db
```
### running a specific action

To run a specific action like crawling

```bash
docker compose --profile <profile> up --build
```
Current available profiles: `crawl` `extract`

### inspecting the database

While the database container is running, open psql:

```bash
docker exec -it orchestration-document-db-1 psql -U <db-username> -d <db-name>
```

Now you can perform SQL queries:

```SQL
SELECT name, doi FROM table LIMIT 10;
```

### resetting the database

Only do this if you know you really have to. First turn off the running containers and then remove the database:


```bash
docker compose down
docker volume rm orchestration_pgdata orchestration_neo4j_data
```

# GPU / CPU runtime

This repo contains a docker-compose file for both CPU and GPU runtimes for Ollama-based model inference. This is currently only relevant to the `extract` profie. Depending on whether a (nvidia) GPU is available on your machine, you can choose to run the stack with or without GPU acceleration for model inference. The default `docker-compose.yml` uses the CPU runtime, you need to set up the docker NVIDIA runtime if you want to use the GPU. 

### CPU runtime (default)

If you wish to perform extraction without GPU, run: 

```bash
docker compose up --profile extract
```

This is meant for local development/debugging and will obviously be slower. If you have access to a GPU it is not recommended.

### GPU runtime prerequisites

If you have an NVIDIA GPU and want hardware acceleration for model inference, first install NVIDIA container toolkit. These instructions come from [the docker hub page for the Ollama docker image](https://hub.docker.com/r/ollama/ollama)

```bash
# Add key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repo
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

configure docker to use the NVIDIA runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

verify the GPU is accessible to docker:


```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### Running extraction with GPU support: 

```bash
docker compose --profile extract -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
This ensures `ollama` and `knowledge extraction` run with full GPU access.

