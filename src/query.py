import os
import time
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

load_dotenv()

project_path = os.getenv("SOURCE_PATH", "/app/source")
storage_path = os.getenv("INDEX_STORAGE", "/app/index")
model_name = os.getenv("MODEL_NAME", "codellama:7b")
index_model_name = os.getenv("INDEX_MODEL_NAME", "all-minilm:l6-v2")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# custom promt to force model to work only with our source
STRICT_CONTEXT_PROMPT = PromptTemplate(
    "You are an expert AI assistant. Answer the question using only the provided context.\n"
    "Check the `{project_path}` and define the code language base.\n"
    "If the answer is not in the context, respond with: 'I couldn't find relevant information in the provided sources.'\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\nAnswer:"
)

# Configure with longer timeouts
embed_model = OllamaEmbedding(
    model_name=index_model_name,
    base_url=ollama_base_url,
    request_timeout=300.0  # 5 minutes timeout
)
llm = Ollama(
    model=model_name,
    base_url=ollama_base_url,
    request_timeout=300.0  # 5 minutes timeout
)

# Set global settings to use Ollama models
Settings.llm = llm
Settings.embed_model = embed_model


class DynamicIndexManager:
    """Manages the vector index and automatically reloads when changes are detected."""

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.index = None
        self.query_engine = None
        self.last_modified = 0
        self._load_index()

    def _get_index_modification_time(self):
        """Get the latest modification time of index files."""
        if not os.path.exists(self.storage_path):
            return 0

        latest_time = 0
        for root, dirs, files in os.walk(self.storage_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    mod_time = os.path.getmtime(file_path)
                    latest_time = max(latest_time, mod_time)
                except OSError:
                    continue
        return latest_time

    def _load_index(self):
        """Load the index from storage."""
        try:
            if os.path.exists(self.storage_path):
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
                self.index = load_index_from_storage(storage_context)
                self.query_engine = self.index.as_query_engine(
                    llm=llm,
                    embed_model=embed_model,
                    similarity_top_k=3,
                    text_qa_template=STRICT_CONTEXT_PROMPT
                )
                self.last_modified = self._get_index_modification_time()
                print(f"📚 Index loaded from: {self.storage_path}")
            else:
                print(f"⚠️  Index not found at: {self.storage_path}")
                self.index = None
                self.query_engine = None
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            self.index = None
            self.query_engine = None

    def _check_and_reload_if_needed(self):
        """Check if index has been updated and reload if necessary."""
        current_mod_time = self._get_index_modification_time()
        if current_mod_time > self.last_modified:
            print(f"🔄 Index has been updated, reloading...")
            self._load_index()

    def query(self, query_text: str):
        """Query the index, reloading if necessary."""
        self._check_and_reload_if_needed()

        if not self.query_engine:
            return "❌ Error: Index not available. Please wait for initial indexing to complete."

        try:
            response = self.query_engine.query(query_text)
            return response.response
        except Exception as e:
            return f"❌ Error querying index: {str(e)}"


# Create global index manager
index_manager = DynamicIndexManager(storage_path)


def ask(query: str):
    """Query the vector index with automatic reloading support."""
    return index_manager.query(query)