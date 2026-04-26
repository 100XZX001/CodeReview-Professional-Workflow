# tools.py - Real vector retrieval for query_docs, linter, and test runner
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
except ImportError:
    chromadb = None


@dataclass
class ToolBox:
    _embedder = None
    _client = None
    _collection = None

    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            if SentenceTransformer is None:
                return None
            cls._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._embedder

    @classmethod
    def _get_collection(cls):
        if cls._collection is None:
            if chromadb is None:
                return None
            cls._client = chromadb.Client()
            cls._collection = cls._client.create_collection("docs")
            docs = [
                "KeyError occurs when a dictionary key is missing. Use dict.get() or check 'if key in dict'.",
                "pylint error C0304: missing final newline. Add a newline at the end of file.",
                "Deadlock happens when two threads acquire locks in opposite order. Always acquire locks in the same order.",
                "Division by zero: check if list is empty before calculating average, or use try/except.",
                "Threading.Lock: use 'with lock:' to automatically acquire and release.",
                "Off-by-one errors: adjust loop ranges, e.g., range(1, len(arr)-1).",
            ]
            embedder = cls._get_embedder()
            if embedder is None:
                return None
            embeddings = embedder.encode(docs).tolist()
            for i, doc in enumerate(docs):
                cls._collection.add(ids=[str(i)], documents=[doc], embeddings=[embeddings[i]])
        return cls._collection

    @staticmethod
    def run_linter(code: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            f.flush()
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pylint", tmp_path, "--exit-zero", "--output-format=text"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
            )
            output = result.stdout
            if "Your code has been rated" in output:
                output = output.split("Your code has been rated")[0]
            output = output.strip()
            if not output:
                return "No linting issues found."
            return output[:500]
        except FileNotFoundError:
            return "Linter (pylint) not installed."
        except subprocess.TimeoutExpired:
            return "Linter timed out."
        except Exception as e:
            return f"Linter error: {str(e)}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def run_tests(test_script: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            f.flush()
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
            )
            output = result.stdout + result.stderr
            return output.strip() or "Test executed successfully (no output)."
        except subprocess.TimeoutExpired:
            return "Test execution timed out."
        except Exception as e:
            return f"Test runner error: {str(e)}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @classmethod
    def query_docs(cls, topic: str) -> str:
        """Retrieve top 3 relevant docs; fall back cleanly when vector deps are missing."""
        try:
            embedder = cls._get_embedder()
            collection = cls._get_collection()
            if embedder is None or collection is None:
                raise RuntimeError("Vector retrieval dependencies are unavailable")
            query_emb = embedder.encode([topic]).tolist()
            results = collection.query(query_embeddings=query_emb, n_results=3)
            if results["documents"] and results["documents"][0]:
                snippets = []
                for i, doc in enumerate(results["documents"][0]):
                    snippets.append(f"[{i + 1}] {doc}")
                return "Relevant documentation:\n" + "\n".join(snippets)
            return "No relevant documentation found."
        except Exception:
            topic_lower = topic.lower()
            fallback = {
                "null check": "To avoid KeyError, use 'if key in dict:' before accessing.",
                "keyerror": "Catch KeyError with try/except or use dict.get().",
                "deadlock": "Always acquire locks in the same order to avoid deadlock.",
                "race": "Protect shared state with a lock or make the update atomic.",
                "division": "Guard empty inputs before dividing or return a safe default.",
            }
            for key, value in fallback.items():
                if key in topic_lower:
                    return value
            return "No relevant documentation found. Try being more specific."
