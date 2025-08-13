import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

# Configure stdout to be unbuffered for immediate output
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

project_path = os.getenv("SOURCE_PATH", "/app/source")
storage_path = os.getenv("INDEX_STORAGE", "/app/index")
model_name = os.getenv("MODEL_NAME", "codellama:7b")
index_model_name = os.getenv("INDEX_MODEL_NAME", "all-minilm:l6-v2")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Configure LlamaIndex settings
embed_model = OllamaEmbedding(model_name=index_model_name, base_url=ollama_base_url)
llm = Ollama(model=model_name, base_url=ollama_base_url)

Settings.llm = llm
Settings.embed_model = embed_model

class CodeFileHandler(FileSystemEventHandler):
    """Handler for file system events that rebuilds the index when code files change."""

    def __init__(self, source_path, index_path, debounce_time=2):
        self.source_path = source_path
        self.index_path = index_path
        self.debounce_time = debounce_time
        self.last_rebuild_time = 0

    def should_process_file(self, file_path):
        """Check if the file should trigger an index rebuild."""
        if not os.path.isfile(file_path):
            return False

        # Watch all files - no extension filtering
        return True

    def rebuild_index(self):
        """Rebuild the entire index from the source directory."""
        current_time = time.time()

        # Debounce: don't rebuild too frequently
        if current_time - self.last_rebuild_time < self.debounce_time:
            return

        print(f"🔄 Rebuilding index due to file changes...")
        try:
            # Load documents from source directory
            documents = SimpleDirectoryReader(self.source_path).load_data()

            # Create new index
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

            # Persist the index
            index.storage_context.persist(persist_dir=self.index_path)

            self.last_rebuild_time = current_time
            print(f"✅ Index rebuilt successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")

    def on_modified(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            print(f"📝 File modified: {event.src_path}")
            self.rebuild_index()

    def on_created(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            print(f"📄 File created: {event.src_path}")
            self.rebuild_index()

    def on_deleted(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            print(f"🗑️  File deleted: {event.src_path}")
            self.rebuild_index()

    def on_moved(self, event):
        if not event.is_directory:
            if (self.should_process_file(event.src_path) or
                self.should_process_file(event.dest_path)):
                print(f"📦 File moved: {event.src_path} -> {event.dest_path}")
                self.rebuild_index()

def initial_index_build():
    """Build the initial index if it doesn't exist."""
    if not os.path.exists(storage_path) or not os.listdir(storage_path):
        print(f"📁 Building initial index from: {project_path}")
        documents = SimpleDirectoryReader(project_path).load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=storage_path)
        print(f"✅ Initial index created: {storage_path}")
    else:
        print(f"📚 Using existing index: {storage_path}")

def start_file_watcher():
    """Start the file watcher to monitor changes in the source directory."""
    print(f"👀 Starting file watcher for: {project_path}")

    # Build initial index if needed
    initial_index_build()

    # Set up file watcher
    event_handler = CodeFileHandler(project_path, storage_path)
    observer = Observer()
    observer.schedule(event_handler, project_path, recursive=True)

    try:
        observer.start()
        print(f"🚀 File watcher started. Monitoring {project_path} for changes...")
        print("Press Ctrl+C to stop")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping file watcher...")
        observer.stop()

    observer.join()
    print("👋 File watcher stopped")

if __name__ == "__main__":
    start_file_watcher()
