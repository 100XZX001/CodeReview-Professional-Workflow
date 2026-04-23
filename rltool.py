# tools.py – Real vector retrieval for query_docs, linter, and test runner
import subprocess
import tempfile
import os
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb

@dataclass
class ToolBox:
    _embedder = None
    _client = None
    _collection = None

    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            cls._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedder

    @classmethod
    def _get_collection(cls):
        if cls._collection is None:
            cls._client = chromadb.Client()
            cls._collection = cls._client.create_collection("docs")
            # Pre‑load real documentation snippets (can be extended)
            docs = [
                "KeyError occurs when a dictionary key is missing. Use dict.get() or check 'if key in dict'.",
                "pylint error C0304: missing final newline. Add a newline at the end of file.",
                "Deadlock happens when two threads acquire locks in opposite order. Always acquire locks in the same order.",
                "Division by zero: check if list is empty before calculating average, or use try/except.",
                "Threading.Lock: use 'with lock:' to automatically acquire and release.",
                "Off‑by‑one errors: adjust loop ranges, e.g., range(1, len(arr)-1).",
            ]
            embedder = cls._get_embedder()
            embeddings = embedder.encode(docs).tolist()
            for i, doc in enumerate(docs):
                cls._collection.add(ids=[str(i)], documents=[doc], embeddings=[embeddings[i]])
        return cls._collection

    @staticmethod
    def run_linter(code: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            f.flush()
            tmp_path = f.name
        try:
            result = subprocess.run(
                ['pylint', tmp_path, '--exit-zero', '--output-format=text'],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8'
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
            except:
                pass

    @staticmethod
    def run_tests(test_script: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_script)
            f.flush()
            tmp_path = f.name
        try:
            result = subprocess.run(
                ['python', tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8'
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
            except:
                pass

    @classmethod
    def query_docs(cls, topic: str) -> str:
        """Retrieve top 3 relevant docs. Forces agent to reason across multiple hints."""
        try:
            embedder = cls._get_embedder()
            collection = cls._get_collection()
            query_emb = embedder.encode([topic]).tolist()
            # Get top 3 results (not just 1)
            results = collection.query(query_embeddings=query_emb, n_results=3)
            if results['documents'] and results['documents'][0]:
                # Return concatenated snippets, labelled for clarity
                snippets = []
                for i, doc in enumerate(results['documents'][0]):
                    snippets.append(f"[{i+1}] {doc}")
                return "Relevant documentation:\n" + "\n".join(snippets)
            return "No relevant documentation found."
        except Exception:
            # Fallback to keyword matching
            topic_lower = topic.lower()
            fallback = {
                "null check": "To avoid KeyError, use 'if key in dict:' before accessing.",
                "keyerror": "Catch KeyError with try/except or use dict.get().",
                "deadlock": "Always acquire locks in the same order to avoid deadlock.",
            }
            for key, value in fallback.items():
                if key in topic_lower:
                    return value
            return "No relevant documentation found. Try being more specific."