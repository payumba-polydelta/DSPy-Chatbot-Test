from dsp.modules import GoogleVertexAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google.oauth2 import service_account


load_dotenv()

# Environment variables and constants
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "basic-embeddings"
PINECONE_HOST = "https://basic-embeddings-m8sj7l5.svc.aped-4627-b74a.pinecone.io"
VERTEX_MODEL_ID = "gemini-1.5-pro-001"
VERTEX_PROJECT_ID = "nvcc-dspy-rag"
VERTEX_REGION = "us-central1"
VERTEX_CREDENTIALS = "./vertex_credentials.json"

def load_embedding_model(api_token: str = HF_API_TOKEN) -> SentenceTransformer:
    """
    Load and return the sentence transformer model for embeddings.

    Args:
        api_token (str): HuggingFace API token for authentication.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    return SentenceTransformer("BAAI/bge-base-en-v1.5", use_auth_token=api_token)

def load_gemini_model(
    model_name: str = VERTEX_MODEL_ID,
    project: str = VERTEX_PROJECT_ID,
    location: str = VERTEX_REGION,
    credentials: str = VERTEX_CREDENTIALS
) -> GoogleVertexAI:
    """
    Load and return the Gemini model from Google Vertex AI.

    Args:
        model_name (str): Name of the Gemini model.
        project (str): Google Cloud project ID.
        location (str): Google Cloud region.
        credentials (str): Path to the service account credentials file.

    Returns:
        GoogleVertexAI: The loaded Gemini model.
    """
    credentials_obj = service_account.Credentials.from_service_account_file(credentials)
    return GoogleVertexAI(
        model_name=model_name,
        project=project,
        location=location,
        credentials=credentials_obj
    )