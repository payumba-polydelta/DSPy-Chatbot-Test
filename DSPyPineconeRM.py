import dspy
from dsp.modules import GoogleVertexAI
from dsp.utils import dotdict
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC
from google.oauth2 import service_account
from typing import List, Optional

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

class PineconeRM(dspy.Retrieve):
    """
    Custom retriever module for DSPy that uses Pinecone as the vector database.
    """

    def __init__(
        self,
        pinecone_index_name: str = PINECONE_INDEX_NAME,
        pinecone_api_key: str = PINECONE_API_KEY,
        pinecone_host: str = PINECONE_HOST,
        k: int = 5
    ):
        """
        Initialize the PineconeRM retriever.

        Args:
            pinecone_index_name (str): Name of the Pinecone index.
            pinecone_api_key (str): Pinecone API key.
            pinecone_host (str): Pinecone host URL.
            k (int): Number of top results to retrieve.
        """
        super().__init__(k=k)
        self._embedding_model = load_embedding_model()
        self._pinecone_index = self._connect_pinecone_index(
            pinecone_index_name, pinecone_api_key, pinecone_host
        )

    def _connect_pinecone_index(
        self,
        index_name: str,
        api_key: str,
        host: str
    ) -> PineconeGRPC:
        """
        Connect to an existing Pinecone index.

        Args:
            index_name (str): Name of the Pinecone index.
            api_key (str): Pinecone API key.
            host (str): Pinecone host URL.

        Returns:
            PineconeGRPC: Connected Pinecone index.
        """
        pc = PineconeGRPC(api_key=api_key)
        return pc.Index(index_name, host=host)

    def _get_embedding(self, question: str) -> List[float]:
        """
        Generate embedding for the given query.

        Args:
            query (str): Input query string.

        Returns:
            List[float]: Embedding vector for the query.
        """
        return self._embedding_model.encode(str(question), convert_to_numpy=True).tolist()

    def forward(self, question: str, k: Optional[int] = None) -> dspy.Prediction:
        """
        Retrieve top-k most similar passages for the given query.

        Args:
            query (str): Input query string.
            k (Optional[int]): Number of results to retrieve. If None, uses the default k.

        Returns:
            dspy.Prediction: Object containing the retrieved passages.
        """
        k = k or self.k
        embedding = self._get_embedding(question)

        results_dict = self._pinecone_index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True
        )

        sorted_results = sorted(
            results_dict["matches"],
            key=lambda x: x["score"],
            reverse=True
        )
        
        passages = [result["metadata"]["text"] for result in sorted_results]
        passages = [dotdict({"long_text": passage}) for passage in passages]
        return dspy.Prediction(passages=passages)
    