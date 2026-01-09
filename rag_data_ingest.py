import sys
import json
import logging
from pathlib import Path
import argparse
import warnings

# Suppress non-critical warnings to keep logs clean
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------
# Ensure project root is available in PYTHONPATH
# This allows running the script from different locations
# ---------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------
# Import project-specific components
# ---------------------------------------------------------
from agents.rag_agent import MedicalRAG
from config import Config

# ---------------------------------------------------------
# Argument Parser Configuration
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Ingest medical documents into the RAG vector database."
)

parser.add_argument(
    "--file",
    type=str,
    required=False,
    help="Path to a single document file to ingest"
)

parser.add_argument(
    "--dir",
    type=str,
    required=False,
    help="Path to a directory containing multiple documents to ingest"
)

# Parse command-line arguments
args = parser.parse_args()

# ---------------------------------------------------------
# Load application configuration
# ---------------------------------------------------------
config = Config()

# Initialize Medical RAG ingestion pipeline
rag = MedicalRAG(config)

# ---------------------------------------------------------
# Document Ingestion Logic
# ---------------------------------------------------------
def data_ingestion() -> bool:
    """
    Ingest documents into the Medical RAG system.

    Supports ingestion of either:
    - A single file (--file)
    - A directory containing multiple files (--dir)

    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    if args.file:
        # Ingest a single document
        file_path = args.file
        logging.info(f"Ingesting file: {file_path}")
        result = rag.ingest_file(file_path)

    elif args.dir:
        # Ingest all documents from a directory
        dir_path = args.dir
        logging.info(f"Ingesting documents from directory: {dir_path}")
        result = rag.ingest_directory(dir_path)

    else:
        logging.error("No input provided. Use --file or --dir.")
        return False

    # Log ingestion result
    print("Ingestion result:")
    print(json.dumps(result, indent=2))

    return result.get("success", False)

# ---------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":

    print("\nStarting document ingestion process...\n")

    ingestion_success = data_ingestion()

    if ingestion_success:
        print("\nDocuments ingested successfully.")
    else:
        print("\nDocument ingestion failed.")
