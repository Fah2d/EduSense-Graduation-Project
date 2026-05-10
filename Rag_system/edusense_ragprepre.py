"""
EduSense RAG System — FAISS Edition
=====================================
No ChromaDB. Works on Python 3.13, no version conflicts.

Install:
    pip install faiss-cpu sentence-transformers anthropic openai-whisper PyMuPDF
"""

import os, json, time, pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# ─────────────────────────────────────────────
# 1. PDF KNOWLEDGE BASE (FAISS)
# ─────────────────────────────────────────────

class PDFKnowledgeBase:
    """PDF → chunks → embeddings → FAISS index"""

    def __init__(self, persist_dir: str = 'edusense_kb'):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder    = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim         = 384
        self.index_path  = self.persist_dir / 'faiss.index'
        self.meta_path   = self.persist_dir / 'metadata.pkl'

        if self.index_path.exists() and self.meta_path.exists():
            self.index    = faiss.read_index(str(self.index_path))
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Knowledge base loaded — {self.index.ntotal} chunks indexed")
        else:
            self.index    = faiss.IndexFlatIP(self.dim)
            self.metadata = []
            print("✅ Knowledge base ready — 0 chunks indexed")

    def _save(self):
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def add_pdf(self, pdf_path: str, source_name: str = None) -> int:
        import fitz
        if source_name is None:
            source_name = Path(pdf_path).stem

        print(f"📄 Processing: {source_name}")
        doc, chunks = fitz.open(pdf_path), []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) < 50:
                continue
            for para in text.split('\n\n'):
                para = para.strip()
                if len(para) > 50:
                    chunks.append({'text': para, 'source': source_name, 'page': page_num + 1})

        if not chunks:
            print(f"  ⚠️ No text extracted"); return 0

        texts = [c['text'] for c in chunks]
        embs  = self.embedder.encode(texts, batch_size=64,
                                     show_progress_bar=False,
                                     normalize_embeddings=True)
        self.index.add(embs.astype('float32'))
        self.metadata.extend(chunks)
        self._save()
        print(f"  ✅ Added {len(chunks)} chunks ({len(doc)} pages)")
        doc.close()
        return len(chunks)

    def add_multiple_pdfs(self, pdf_paths: List[str]) -> int:
        total = sum(self.add_pdf(p) for p in pdf_paths)
        print(f"\n✅ Total: {self.index.ntotal} chunks")
        return total

    def retrieve(self, query: str, n_results: int = 5,
                 subject_id: str = None) -> List[Dict]:
        """
        Retrieve relevant chunks scoped strictly to this subject's uploaded PDFs.

        Logic:
          1. If subject_id provided → only return chunks from that subject (prefix match).
          2. If no subject-specific chunks exist → return [] with a warning.
             This prevents cross-subject contamination (e.g. binary search chunks
             appearing in a Big Data notebook).
          3. If no subject_id → return top results from the full knowledge base.
        """
        if self.index.ntotal == 0:
            print('⚠️  RAG: knowledge base is empty — no chunks to retrieve')
            return []

        q_emb   = self.embedder.encode([query], normalize_embeddings=True)
        fetch_n = min(max(n_results * 10, 50), self.index.ntotal)
        scores, idxs = self.index.search(q_emb.astype('float32'), fetch_n)
        raw = [
            {**self.metadata[i], 'relevance': float(s)}
            for s, i in zip(scores[0], idxs[0]) if i >= 0
        ]

        if subject_id:
            prefix   = f"{subject_id[:8]}::"
            filtered = [r for r in raw if r['source'].startswith(prefix)]
            if filtered:
                print(f'✅ RAG: retrieved {len(filtered[:n_results])} chunks '
                      f'from subject {prefix} (out of {len(filtered)} matches)')
                return filtered[:n_results]
            else:
                # No PDFs uploaded for this subject — do NOT fall back to other subjects
                # This is the key fix: returning chunks from other subjects causes wrong notebooks
                all_sources = sorted(set(m['source'] for m in self.metadata))
                print(f'⚠️  RAG: no chunks found for subject prefix "{prefix}"')
                print(f'   Available sources: {all_sources}')
                print(f'   → Returning empty context. Upload a PDF for this subject.')
                return []

        # No subject filter — return best global results
        print(f'⚠️  RAG: no subject_id provided — using global knowledge base')
        return raw[:n_results]

    def count(self):
        return self.index.ntotal

    def stats(self):
        print(f"\n📚 Knowledge Base: {self.index.ntotal} chunks")
        if self.metadata:
            for s in sorted(set(m['source'] for m in self.metadata)):
                print(f"   - {s}")


# ─────────────────────────────────────────────
# 2. WHISPER TRANSCRIPTION
# ─────────────────────────────────────────────

class LectureTranscriber:
    def __init__(self, model_size: str = 'base'):
        import whisper
        print(f"Loading Whisper {model_size}...")
        self.model      = whisper.load_model(model_size)
        self.transcript = ""
        self.segments   = []
        print(f"✅ Whisper {model_size} ready")

    def transcribe_file(self, audio_path: str, language: str = 'en') -> str:
        result          = self.model.transcribe(audio_path, language=language,
                           initial_prompt="Computer science lecture.", verbose=False)
        self.transcript = result['text']
        self.segments   = result.get('segments', [])
        print(f"✅ Transcribed {len(self.transcript)} chars")
        return self.transcript

    def get_last_n_seconds(self, n: int = 60) -> str:
        if not self.segments:
            return self.transcript
        cutoff = self.segments[-1]['end'] - n
        recent = [s['text'] for s in self.segments if s['start'] >= cutoff]
        return ' '.join(recent).strip() or self.transcript[-500:]


# ─────────────────────────────────────────────
# 3. NOTEBOOK GENERATOR (Claude API)
# ─────────────────────────────────────────────

class NotebookGenerator:
    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        print("✅ Claude API connected")

    def generate(self, transcript: str, retrieved_chunks: List[Dict],
                 emotion_state: Dict) -> Dict:

        detected     = [e for e, v in emotion_state.items()
                       if isinstance(v, dict) and v.get('positive')]
        emotion_desc = ", ".join(detected) if detected else "low engagement"

        # Only use chunks that actually belong to this session's subject
        if retrieved_chunks:
            context = "\n\n".join([
                f"[From {c['source']}, page {c['page']}]\n{c['text']}"
                for c in retrieved_chunks[:4]
            ])
            context_section = f"""TEXTBOOK CONTENT (from this subject's uploaded materials):
{context}"""
        else:
            # No subject-specific PDFs uploaded — base notebook entirely on transcript
            context_section = (
                "NOTE: No textbook PDFs have been uploaded for this subject yet. "
                "Generate the notebook based solely on the lecture transcript below."
            )
            print('⚠️  Generating notebook from transcript only — no subject PDFs found')

        prompt = f"""You are an expert CS tutor creating a personalized study notebook.

A student showed signs of {emotion_desc} during this lecture:

TRANSCRIPT:
{transcript[:800]}

{context_section}

Return ONLY valid JSON (no markdown, no backticks):
{{
  "title": "concise topic title",
  "detected_topic": "one sentence about what the student struggled with",
  "explanation": "clear 3-4 sentence explanation in simple terms",
  "analogy": "a real-world analogy",
  "key_points": ["point 1", "point 2", "point 3"],
  "code_examples": [
    {{"description": "what this shows", "code": "complete runnable Python code", "expected_output": "output"}},
    {{"description": "second example", "code": "complete runnable Python code", "expected_output": "output"}}
  ],
  "exercises": [
    {{"question": "exercise", "hint": "hint", "solution": "solution code"}}
  ],
  "further_reading": "one sentence about where to learn more"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        return json.loads(raw)

    def to_ipynb(self, data: Dict, output_path: str,
                 emotion_state: Dict = None) -> str:
        detected = [e for e, v in (emotion_state or {}).items()
                   if isinstance(v, dict) and v.get('positive')]
        cells = []

        # Title
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [
            f"# 📚 {data['title']}\n\n",
            "*Auto-generated by **EduSense** · King Khalid University*\n\n---\n",
            f"**Why this notebook?** {data['detected_topic']}\n\n",
            *([ f"> 💡 **Detected:** {', '.join(detected)}\n"] if detected else [])
        ]})

        # Explanation
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [
            "## 💡 Explanation\n\n", data['explanation'],
            "\n\n### 🌍 Real-World Analogy\n\n", data['analogy']
        ]})

        # Key points
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [
            "## 🎯 Key Points\n\n" +
            "\n".join(f"- {p}" for p in data.get('key_points', []))
        ]})

        # Code examples
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": ["## 💻 Code Examples"]})
        for i, ex in enumerate(data.get('code_examples', []), 1):
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": [f"### Example {i}: {ex['description']}"]})
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": [ex['code']]})
            if ex.get('expected_output'):
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": [f"**Expected output:**\n```\n{ex['expected_output']}\n```"]})

        # Exercises
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": ["## 🏋️ Practice Exercises"]})
        for i, ex in enumerate(data.get('exercises', []), 1):
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [
                f"### Exercise {i}\n\n**{ex['question']}**\n\n> 💡 Hint: {ex.get('hint','')}"
            ]})
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": ["# Your solution here\n"]})
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [
                "<details><summary>👁️ Show Solution</summary>\n\n",
                f"```python\n{ex['solution']}\n```\n\n</details>"
            ]})

        if data.get('further_reading'):
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": [f"---\n## 📖 Further Reading\n\n{data['further_reading']}"]})

        ipynb = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.0"},
                "edusense": {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "topic": data.get('title', ''), "detected_emotions": detected}
            },
            "nbformat": 4, "nbformat_minor": 4
        }

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ipynb, f, indent=2)
        print(f"✅ Notebook saved: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
        return output_path


# ─────────────────────────────────────────────
# 4. FULL RAG PIPELINE
# ─────────────────────────────────────────────

class EduSenseRAG:
    def __init__(self, anthropic_api_key: str,
                 kb_dir: str = 'edusense_kb',
                 output_dir: str = 'generated_notebooks',
                 whisper_model: str = 'base'):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        print("Initializing EduSense RAG System...")
        self.kb          = PDFKnowledgeBase(kb_dir)
        self.transcriber = LectureTranscriber(whisper_model)
        self.generator   = NotebookGenerator(anthropic_api_key)
        print("\n✅ EduSense RAG System ready")

    def upload_textbook(self, pdf_path: str, name: str = None) -> int:
        return self.kb.add_pdf(pdf_path, name)

    def upload_textbooks(self, pdf_paths: List[str]) -> int:
        return self.kb.add_multiple_pdfs(pdf_paths)

    def process_with_text(self, transcript: str, emotion_state: Dict,
                          subject_id: str = None) -> str:
        chunks = self.kb.retrieve(transcript, n_results=5, subject_id=subject_id)
        data   = self.generator.generate(transcript, chunks, emotion_state)
        ts     = time.strftime("%Y%m%d_%H%M%S")
        topic  = data.get('title', 'notebook').replace(' ', '_')[:30]
        path   = f"{self.output_dir}/{ts}_{topic}.ipynb"
        self.generator.to_ipynb(data, path, emotion_state)
        return path

    def process_dissatisfaction(self, audio_path: str,
                                 emotion_state: Dict, last_n_secs: int = 60) -> str:
        self.transcriber.transcribe_file(audio_path)
        transcript = self.transcriber.get_last_n_seconds(last_n_secs)
        return self.process_with_text(transcript, emotion_state)

    def knowledge_base_stats(self):
        self.kb.stats()
