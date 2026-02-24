import os
import json
import subprocess
import time
import datetime
import shutil
from neo4j import GraphDatabase
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Docker Compose related
DOCKER_COMPOSE_FILES = ["docker-compose.yml", "docker-compose.gpu.yml"] # Paths relative to orchestrator/
DOCKER_COMPOSE_PROFILES = ["extract"] # Your Docker Compose profiles
PYTHON_SERVICE_NAME = "knowledge-extraction" # Corrected based on your docker-compose.yml
DOCKER_PROJECT_PREFIX = "kg_exp_" # Unique prefix for docker compose projects

# Paths (relative to the orchestration directory)
EXPERIMENTS_DIR = "experiments"
RESULTS_BASE_DIR = "results"
TEMP_CONFIGS_DIR = "temp_configs" # This will hold the dynamically generated config for the KE service
NEO4J_IMPORTS_DIR = "neo4j_imports" # Shared volume for APOC exports

# Neo4j connection details (from your .env)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") # Ensure this is in your .env or handled securely

# PostgreSQL connection details (from your .env)
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost") # Assuming orchestrator talks to DB on localhost
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# --- Helper Functions ---

def setup_orchestrator_directories():
    """Ensures necessary directories for the orchestrator exist."""
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    os.makedirs(TEMP_CONFIGS_DIR, exist_ok=True)
    os.makedirs(NEO4J_IMPORTS_DIR, exist_ok=True) # For APOC exports

def clean_temp_configs():
    """Removes old temporary config files."""
    if os.path.exists(TEMP_CONFIGS_DIR):
        for filename in os.listdir(TEMP_CONFIGS_DIR):
            file_path = os.path.join(TEMP_CONFIGS_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def run_cypher_query(driver, query, database="neo4j"):
    """Executes a Cypher query against Neo4j."""
    with driver.session(database=database) as session:
        try:
            session.run(query)
            print(f"Successfully executed Cypher query: {query.strip().splitlines()[0]}...")
        except Exception as e:
            print(f"Error executing Cypher query: {query}\nError: {e}")
            raise

def connect_to_neo4j():
    """Establishes connection to Neo4j."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}")
        return None

def connect_to_postgres():
    """Establishes connection to PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB
        )
        print("Connected to PostgreSQL successfully.")
        return conn
    except Exception as e:
        print(f"Failed to connect to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}: {e}")
        return None

def create_postgres_table(conn):
    """Creates the experiments table in PostgreSQL if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                experiment_id VARCHAR(255) UNIQUE NOT NULL,
                start_time TIMESTAMP WITH TIME ZONE,
                end_time TIMESTAMP WITH TIME ZONE,
                status VARCHAR(50),
                llm_model_name VARCHAR(255),
                embedding_model_name VARCHAR(255),
                prompt_template_preview TEXT,
                schema_preview JSONB,
                full_config_json JSONB,
                graph_file_path VARCHAR(512),
                log_file_path VARCHAR(512),
                error_message TEXT
            );
        """))
        conn.commit()
        print("PostgreSQL 'experiments' table checked/created.")

def insert_experiment_record(conn, record):
    """Inserts or updates an experiment record in PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            INSERT INTO experiments (
                experiment_id, start_time, end_time, status, llm_model_name,
                embedding_model_name, prompt_template_preview, schema_preview,
                full_config_json, graph_file_path, log_file_path, error_message
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (experiment_id) DO UPDATE SET
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                status = EXCLUDED.status,
                llm_model_name = EXCLUDED.llm_model_name,
                embedding_model_name = EXCLUDED.embedding_model_name,
                prompt_template_preview = EXCLUDED.prompt_template_preview,
                schema_preview = EXCLUDED.schema_preview,
                full_config_json = EXCLUDED.full_config_json,
                graph_file_path = EXCLUDED.graph_file_path,
                log_file_path = EXCLUDED.log_file_path,
                error_message = EXCLUDED.error_message;
        """), (
            record.get("experiment_id"), record.get("start_time"), record.get("end_time"),
            record.get("status"), record.get("llm_model_name"), record.get("embedding_model_name"),
            record.get("prompt_template_preview"), json.dumps(record.get("schema_preview")),
            json.dumps(record.get("full_config_json")), record.get("graph_file_path"),
            record.get("log_file_path"), record.get("error_message")
        ))
        conn.commit()
        print(f"Experiment '{record.get('experiment_id')}' record updated in PostgreSQL.")


# --- Main Orchestration Logic ---
def run_all_experiments():
    setup_orchestrator_directories()
    clean_temp_configs() # Clean up any leftovers from previous runs

    neo4j_driver = connect_to_neo4j()
    if not neo4j_driver:
        print("Exiting: Could not connect to Neo4j. Ensure Neo4j is running and accessible.")
        return

    pg_conn = connect_to_postgres()
    if not pg_conn:
        print("Exiting: Could not connect to PostgreSQL. Ensure PostgreSQL is running and accessible.")
        neo4j_driver.close()
        return

    create_postgres_table(pg_conn)

    experiment_files = [f for f in os.listdir(EXPERIMENTS_DIR) if f.endswith(".json")]
    if not experiment_files:
        print(f"No experiment JSON files found in '{EXPERIMENTS_DIR}'.")
        neo4j_driver.close()
        pg_conn.close()
        return

    print(f"Found {len(experiment_files)} experiments to run.")

    for exp_file in sorted(experiment_files): # Sort for consistent order
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_file)
        try:
            with open(exp_path, 'r') as f:
                experiment_config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed experiment file '{exp_file}': {e}")
            continue

        experiment_id = experiment_config.get("experiment_id", f"untitled_exp_{int(time.time())}")
        llm_model_name = experiment_config.get("ollama_model", os.getenv("OLLAMA_LLM_MODEL", "llama3"))
        embedding_model_name = experiment_config.get("ollama_embedding_model", os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
        
        # This is the actual config object that your Python app (main.py) will consume
        config_for_python_app = experiment_config.get("config_for_python_app", {})
        
        project_name = f"{DOCKER_PROJECT_PREFIX}{experiment_id.replace('-', '_').replace('.', '_').lower()}"
        results_exp_dir = os.path.join(RESULTS_BASE_DIR, experiment_id)
        os.makedirs(results_exp_dir, exist_ok=True)
        log_file_path = os.path.join(results_exp_dir, "run.log")
        graph_file_path = os.path.join(results_exp_dir, "graph.cypher")
        
        # Temp config file path on host for mounting
        temp_config_filename = f"{experiment_id}_config.json"
        temp_config_path_on_host = os.path.join(TEMP_CONFIGS_DIR, temp_config_filename)
        
        # This is the path the KE service expects inside its container
        app_config_path_in_container = f"/app/orchestrator_config/{temp_config_filename}"

        current_exp_record = {
            "experiment_id": experiment_id,
            "start_time": datetime.datetime.now(datetime.timezone.utc),
            "end_time": None,
            "status": "RUNNING",
            "llm_model_name": llm_model_name,
            "embedding_model_name": embedding_model_name,
            "prompt_template_preview": config_for_python_app.get("prompt_template", "")[:500] + "...",
            "schema_preview": config_for_python_app.get("schema"),
            "full_config_json": experiment_config, # Store the full experiment config for reference
            "graph_file_path": None,
            "log_file_path": os.path.abspath(log_file_path),
            "error_message": None
        }
        insert_experiment_record(pg_conn, current_exp_record)

        print(f"\n--- Running Experiment: {experiment_id} ---")
        print(f"Using LLM: {llm_model_name}, Embeddings: {embedding_model_name}")
        print(f"Results will be stored in: {results_exp_dir}")

        try:
            # 1. Write the dynamic config file for the Python app
            with open(temp_config_path_on_host, 'w') as f:
                json.dump(config_for_python_app, f, indent=2)
            print(f"Created temporary config file for KE service: {temp_config_path_on_host}")

            # 2. Clear Neo4j
            print("Clearing Neo4j database...")
            run_cypher_query(neo4j_driver, "MATCH (n) DETACH DELETE n;")
            
            # Also clear any old export files from the shared import directory
            for filename in os.listdir(NEO4J_IMPORTS_DIR):
                file_to_remove = os.path.join(NEO4J_IMPORTS_DIR, filename)
                try:
                    if os.path.isfile(file_to_remove):
                        os.remove(file_to_remove)
                        print(f"Removed old export file: {filename}")
                except Exception as e:
                    print(f"Warning: Could not remove old export file {filename}: {e}")

            # 3. Prepare environment for Docker Compose (to be passed to knowledge-extraction service)
            env_vars = os.environ.copy() # Start with current process env vars
            env_vars["OLLAMA_LLM_MODEL"] = llm_model_name # For pull_model.py
            env_vars["OLLAMA_EMBEDDING_MODEL"] = embedding_model_name # For main.py (via config)
            env_vars["APP_CONFIG_PATH_IN_CONTAINER"] = app_config_path_in_container # For main.py

            # 4. Run Docker Compose
            print(f"Starting Docker Compose project '{project_name}'...")
            docker_compose_cmd = [
                "docker", "compose",
                "-p", project_name, # Use unique project name
                *["-f", f for f in DOCKER_COMPOSE_FILES], # All docker compose files
                "up",
                "--build", # Build images if changed
                "--force-recreate", # Force recreation of containers
                "--exit-code-from", PYTHON_SERVICE_NAME, # Wait for extract service to exit
                *["--profile", p for p in DOCKER_COMPOSE_PROFILES] # Apply profiles
            ]
            print(f"Running command: {' '.join(docker_compose_cmd)}")

            with open(log_file_path, "w") as log_file:
                process = subprocess.run(docker_compose_cmd, env=env_vars, stdout=log_file, stderr=subprocess.STDOUT)
            
            if process.returncode == 0:
                print(f"Docker Compose for '{experiment_id}' completed successfully.")
                current_exp_record["status"] = "SUCCESS"

                # 5. Export graph from Neo4j
                print("Exporting graph from Neo4j...")
                exported_filename_in_container = f"{experiment_id}_graph.cypher"
                cypher_export_query = f"CALL apoc.export.cypher.all('{exported_filename_in_container}', {{format: 'cypher-shell', use={convert_to_cypher_bool(False)}, ifExists:'overwrite'}});"
                # Note: 'use:false' is for older APOC versions if you have issues, otherwise omit.
                # 'ifExists:overwrite' is good practice.
                run_cypher_query(neo4j_driver, cypher_export_query)

                # Move the exported file from the shared volume to the experiment's results directory
                exported_file_on_host = os.path.join(NEO4J_IMPORTS_DIR, exported_filename_in_container)
                # Give some time for file system sync if Docker is slow
                time.sleep(2) 

                if os.path.exists(exported_file_on_host):
                    shutil.move(exported_file_on_host, graph_file_path)
                    print(f"Exported graph saved to: {graph_file_path}")
                    current_exp_record["graph_file_path"] = os.path.abspath(graph_file_path)
                else:
                    print(f"Warning: Exported graph file not found at {exported_file_on_host}. Check APOC configuration and permissions.")
                    current_exp_record["status"] = "EXPORT_FAILED"
                    current_exp_record["error_message"] = "Exported graph file not found."

            else:
                current_exp_record["status"] = "FAILED"
                current_exp_record["error_message"] = f"Docker Compose for {PYTHON_SERVICE_NAME} exited with code {process.returncode}. Check logs: {log_file_path}"
                print(f"Docker Compose for '{experiment_id}' failed with exit code {process.returncode}. Check logs: {log_file_path}")

        except Exception as e:
            current_exp_record["status"] = "ERROR"
            current_exp_record["error_message"] = f"Orchestration script error: {e}"
            print(f"An unexpected error occurred during experiment '{experiment_id}': {e}")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\nORCHESTRATION SCRIPT ERROR: {e}\n")
        finally:
            current_exp_record["end_time"] = datetime.datetime.now(datetime.timezone.utc)
            insert_experiment_record(pg_conn, current_exp_record)
            
            # Clean up temp config file
            if os.path.exists(temp_config_path_on_host):
                os.remove(temp_config_path_on_host)
            
            # Bring down the entire stack for this project
            print(f"Bringing down Docker Compose project '{project_name}'...")
            # Use --remove-orphans in case any service was unexpectedly created
            subprocess.run(["docker", "compose", "-p", project_name, "down", "--remove-orphans"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Docker Compose stack shut down.")

    print("\n--- All experiments finished ---")
    neo4j_driver.close()
    pg_conn.close()

# Helper for APOC query
def convert_to_cypher_bool(python_bool):
    return 'true' if python_bool else 'false'

if __name__ == "__main__":
    # Create a .env file in the 'orchestration' directory like this:
    # NEO4J_USER=neo4j
    # NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
    # POSTGRES_USER=dorus
    # POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD
    # POSTGRES_DB=document_database
    # OLLAMA_HOST=http://localhost:11434 # Or the IP if Ollama is not on localhost for orchestration script

    # Ensure Ollama is directly accessible by the orchestrator script
    # If Ollama is only accessible via 'ollama' hostname from inside docker-compose network,
    # and not directly from your host, you might need to change OLLAMA_HOST here to localhost:11434
    # and ensure your docker-compose exposes ollama's port.

    run_all_experiments()
