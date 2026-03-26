"""
EduSense RAG System
===================
Full pipeline:
1. Upload PDF textbooks/slides → ChromaDB vector store
2. Whisper transcribes lecture audio
3. RAG retrieves relevant content
4. Claude API generates personalized .ipynb notebook
"""

# ─────────────────────────────────────────────────────────
# INSTALL (run once in Colab)
# !pip install chromadb sentence-transformers anthropic openai-whisper PyMuPDF -q
# ─────────────────────────────────────────────────────────

import os
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────
# 1. PDF INGESTION → CHROMADB
# ─────────────────────────────────────────────────────────

class PDFKnowledgeBase:
    """
    Upload PDFs → chunk text → embed → store in ChromaDB
    Supports: textbooks, slides, lecture notes
    """

    def __init__(self, persist_dir: str = '/content/drive/MyDrive/edusense_kb'):
        import chromadb
        from sentence_transformers import SentenceTransformer

        os.makedirs(persist_dir, exist_ok=True)
        self.client     = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="cs_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder   = SentenceTransformer('all-MiniLM-L6-v2')
        self.persist_dir = persist_dir
        print(f"✅ Knowledge base ready — {self.collection.count()} chunks indexed")

    def add_pdf(self, pdf_path: str, source_name: str = None) -> int:
        """
        Extract text from PDF, chunk it, embed and store.
        Returns number of chunks added.
        """
        import fitz  # PyMuPDF

        if source_name is None:
            source_name = Path(pdf_path).stem

        print(f"📄 Processing: {source_name}")
        doc    = fitz.open(pdf_path)
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) < 50:  # skip empty pages
                continue

            # Chunk by paragraph (split on double newline)
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

            for para in paragraphs:
                # Sliding window for better context
                chunks.append({
                    'text':    para,
                    'source':  source_name,
                    'page':    page_num + 1,
                })

        if not chunks:
            print(f"  ⚠️ No text extracted from {source_name}")
            return 0

        # Embed all chunks
        texts      = [c['text'] for c in chunks]
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=False)

        # Store in ChromaDB
        existing_count = self.collection.count()
        ids      = [f"{source_name}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
        metas    = [{'source': c['source'], 'page': c['page']} for c in chunks]

        self.collection.add(
            documents  = texts,
            embeddings = embeddings.tolist(),
            metadatas  = metas,
            ids        = ids
        )

        added = len(chunks)
        print(f"  ✅ Added {added} chunks from {source_name} ({len(doc)} pages)")
        doc.close()
        return added

    def add_multiple_pdfs(self, pdf_paths: List[str]) -> int:
        """Add multiple PDFs at once"""
        total = 0
        for path in pdf_paths:
            total += self.add_pdf(path)
        print(f"\n✅ Total chunks in knowledge base: {self.collection.count()}")
        return total

    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        query_emb = self.embedder.encode([query])
        results   = self.collection.query(
            query_embeddings = query_emb.tolist(),
            n_results        = min(n_results, self.collection.count())
        )

        chunks = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            chunks.append({
                'text':       doc,
                'source':     meta['source'],
                'page':       meta['page'],
                'relevance':  1 - dist  # convert distance to similarity
            })

        return chunks

    def stats(self):
        """Show knowledge base statistics"""
        count = self.collection.count()
        print(f"\n📚 Knowledge Base Stats:")
        print(f"   Total chunks: {count}")
        if count > 0:
            # Get unique sources
            sample = self.collection.get(limit=min(count, 1000))
            sources = set(m['source'] for m in sample['metadatas'])
            print(f"   Sources: {len(sources)}")
            for s in sorted(sources):
                print(f"   - {s}")


# ─────────────────────────────────────────────────────────
# 2. WHISPER TRANSCRIPTION
# ─────────────────────────────────────────────────────────

class LectureTranscriber:
    """Transcribe lecture audio using OpenAI Whisper"""

    def __init__(self, model_size: str = 'base'):
        import whisper
        print(f"Loading Whisper {model_size}...")
        self.model      = whisper.load_model(model_size)
        self.transcript = ""
        self.segments   = []
        print(f"✅ Whisper {model_size} ready")

    def transcribe_file(self, audio_path: str, language: str = 'en') -> str:
        """Transcribe a full audio file"""
        print(f"🎙️ Transcribing: {audio_path}")
        result          = self.model.transcribe(
            audio_path,
            language        = language,
            initial_prompt  = "This is a computer science lecture.",
            verbose         = False
        )
        self.transcript = result['text']
        self.segments   = result.get('segments', [])
        print(f"✅ Transcribed {len(self.transcript)} characters")
        return self.transcript

    def get_last_n_seconds(self, n: int = 60) -> str:
        """Get transcript from last N seconds"""
        if not self.segments:
            return self.transcript

        cutoff = self.segments[-1]['end'] - n if self.segments else 0
        recent = [s['text'] for s in self.segments if s['start'] >= cutoff]
        return ' '.join(recent).strip() or self.transcript[-500:]

    def transcribe_realtime_chunk(self, audio_chunk_path: str) -> str:
        """Transcribe a short audio chunk (for real-time use)"""
        result = self.model.transcribe(
            audio_chunk_path,
            language       = 'en',
            initial_prompt = "Computer science lecture:",
        )
        return result['text'].strip()


# ─────────────────────────────────────────────────────────
# 3. NOTEBOOK GENERATOR (Claude API)
# ─────────────────────────────────────────────────────────

class NotebookGenerator:
    """Generate personalized Jupyter notebooks using Claude API + RAG"""

    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        print("✅ Claude API connected")

    def generate(
        self,
        transcript:      str,
        retrieved_chunks: List[Dict],
        emotion_state:   Dict,
        student_name:    str = "Student"
    ) -> Dict:
        """
        Generate notebook content using Claude.
        Returns dict with title, explanation, code, exercises.
        """
        # Format retrieved context
        context = "\n\n".join([
            f"[From {c['source']}, page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks[:4]
        ])

        # Describe detected emotions
        detected = [e for e, v in emotion_state.items()
                   if isinstance(v, dict) and v.get('positive')]
        emotion_desc = ", ".join(detected) if detected else "low engagement"

        prompt = f"""You are an expert CS tutor creating a personalized study notebook.

A student showed signs of {emotion_desc} during this lecture segment:

LECTURE TRANSCRIPT (last 60 seconds):
{transcript[:800]}

RELEVANT TEXTBOOK CONTENT:
{context}

Create a personalized Jupyter notebook to help this student understand the topic better.
Focus on what they were struggling with based on the transcript.

Return ONLY valid JSON (no markdown, no backticks) in this exact format:
{{
  "title": "concise topic title",
  "detected_topic": "one sentence describing what the student was struggling with",
  "explanation": "clear 3-4 sentence explanation of the core concept in simple terms",
  "analogy": "a real-world analogy that makes this concept intuitive",
  "key_points": ["point 1", "point 2", "point 3"],
  "code_examples": [
    {{
      "description": "what this code demonstrates",
      "code": "complete runnable Python code with comments",
      "expected_output": "what the output will look like"
    }},
    {{
      "description": "second example showing a different aspect",
      "code": "complete runnable Python code with comments",
      "expected_output": "what the output will look like"
    }}
  ],
  "exercises": [
    {{
      "question": "exercise question",
      "hint": "helpful hint",
      "solution": "complete solution code"
    }}
  ],
  "further_reading": "one sentence pointing to where to learn more"
}}"""

        response = self.client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 3000,
            messages   = [{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        # Clean any accidental markdown
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1]
            raw = raw.rsplit('```', 1)[0]

        return json.loads(raw)

    def to_ipynb(self, data: Dict, output_path: str, emotion_state: Dict = None) -> str:
        """Convert generated content to a real .ipynb file"""

        detected = []
        if emotion_state:
            detected = [e for e, v in emotion_state.items()
                       if isinstance(v, dict) and v.get('positive')]

        cells = []

        # Title cell
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# 📚 {data['title']}\n\n",
                f"*Auto-generated by **EduSense** · King Khalid University*\n\n",
                f"---\n",
                f"**Why this notebook?** {data['detected_topic']}\n\n",
            ] + ([f"> 💡 **Detected:** {', '.join(detected)}\n"] if detected else [])
        })

        # Explanation
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 Explanation\n\n",
                data['explanation'],
                "\n\n",
                "### 🌍 Real-World Analogy\n\n",
                data['analogy']
            ]
        })

        # Key points
        key_pts = "\n".join([f"- {p}" for p in data.get('key_points', [])])
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## 🎯 Key Points\n\n{key_pts}"]
        })

        # Code examples
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 💻 Code Examples"]
        })

        for i, ex in enumerate(data.get('code_examples', []), 1):
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### Example {i}: {ex['description']}"]
            })
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [ex['code']]
            })
            if ex.get('expected_output'):
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"**Expected output:**\n```\n{ex['expected_output']}\n```"]
                })

        # Exercises
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🏋️ Practice Exercises"]
        })

        for i, ex in enumerate(data.get('exercises', []), 1):
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"### Exercise {i}\n\n",
                    f"**{ex['question']}**\n\n",
                    f"> 💡 Hint: {ex.get('hint', '')}"
                ]
            })
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Your solution here\n"]
            })
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<details><summary>👁️ Show Solution</summary>\n\n",
                    f"```python\n{ex['solution']}\n```\n\n",
                    "</details>"
                ]
            })

        # Further reading
        if data.get('further_reading'):
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    f"## 📖 Further Reading\n\n{data['further_reading']}"
                ]
            })

        ipynb = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                },
                "edusense": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "topic": data.get('title', ''),
                    "detected_emotions": detected
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ipynb, f, indent=2)

        size = os.path.getsize(output_path) / 1024
        print(f"✅ Notebook saved: {output_path} ({size:.1f} KB)")
        return output_path


# ─────────────────────────────────────────────────────────
# 4. FULL RAG PIPELINE
# ─────────────────────────────────────────────────────────

class EduSenseRAG:
    """
    Complete RAG pipeline:
    PDF → ChromaDB → Whisper → Claude → .ipynb
    """

    def __init__(
        self,
        anthropic_api_key: str,
        kb_dir:            str = '/content/drive/MyDrive/edusense_kb',
        output_dir:        str = '/content/drive/MyDrive/edusense_notebooks',
        whisper_model:     str = 'base'
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir  = output_dir

        print("Initializing EduSense RAG System...")
        self.kb          = PDFKnowledgeBase(kb_dir)
        self.transcriber = LectureTranscriber(whisper_model)
        self.generator   = NotebookGenerator(anthropic_api_key)
        print("\n✅ EduSense RAG System ready")

    def upload_textbook(self, pdf_path: str, name: str = None) -> int:
        """Add a PDF textbook to the knowledge base"""
        return self.kb.add_pdf(pdf_path, name)

    def upload_textbooks(self, pdf_paths: List[str]) -> int:
        """Add multiple PDFs"""
        return self.kb.add_multiple_pdfs(pdf_paths)

    def process_dissatisfaction(
        self,
        audio_path:    str,
        emotion_state: Dict,
        last_n_secs:   int = 60
    ) -> str:
        """
        Full pipeline triggered when student is dissatisfied.
        Returns path to generated .ipynb
        """
        print("\n" + "="*55)
        print("⚠️  DISSATISFACTION DETECTED — Generating notebook")
        print("="*55)

        # Step 1: Transcribe
        print("\n🎙️ Step 1: Transcribing lecture...")
        self.transcriber.transcribe_file(audio_path)
        transcript = self.transcriber.get_last_n_seconds(last_n_secs)
        print(f"   Transcript: {transcript[:100]}...")

        # Step 2: RAG retrieval
        print("\n🔍 Step 2: Retrieving relevant content...")
        chunks = self.kb.retrieve(transcript, n_results=5)
        if chunks:
            print(f"   Found {len(chunks)} relevant chunks:")
            for c in chunks[:3]:
                print(f"   - [{c['source']} p.{c['page']}] relevance={c['relevance']:.2f}")
        else:
            print("   ⚠️ No relevant content found — using general knowledge")

        # Step 3: Generate notebook
        print("\n🤖 Step 3: Generating personalized notebook...")
        notebook_data = self.generator.generate(
            transcript       = transcript,
            retrieved_chunks = chunks,
            emotion_state    = emotion_state
        )
        print(f"   Topic: {notebook_data.get('title', 'Unknown')}")

        # Step 4: Save as .ipynb
        timestamp   = time.strftime("%Y%m%d_%H%M%S")
        topic_clean = notebook_data.get('title', 'notebook').replace(' ', '_')[:30]
        output_path = f"{self.output_dir}/{timestamp}_{topic_clean}.ipynb"

        self.generator.to_ipynb(notebook_data, output_path, emotion_state)

        print(f"\n✅ DONE — Notebook ready: {output_path}")
        print("="*55)
        return output_path

    def process_with_text(
        self,
        transcript:    str,
        emotion_state: Dict
    ) -> str:
        """
        Pipeline when you already have the transcript text
        (no audio file needed — useful for testing)
        """
        print("\n🔍 Retrieving relevant content...")
        chunks = self.kb.retrieve(transcript, n_results=5)

        print("🤖 Generating notebook...")
        data = self.generator.generate(transcript, chunks, emotion_state)

        timestamp   = time.strftime("%Y%m%d_%H%M%S")
        topic_clean = data.get('title', 'notebook').replace(' ', '_')[:30]
        output_path = f"{self.output_dir}/{timestamp}_{topic_clean}.ipynb"

        self.generator.to_ipynb(data, output_path, emotion_state)
        return output_path

    def knowledge_base_stats(self):
        self.kb.stats()


# ─────────────────────────────────────────────────────────
# 5. COLAB DEMO USAGE
# ─────────────────────────────────────────────────────────

DEMO_USAGE = '''
# ══════════════════════════════════════════════════
# EDUSENSE RAG SYSTEM — COLAB SETUP & USAGE
# ══════════════════════════════════════════════════

# STEP 1: Install dependencies
!pip install chromadb sentence-transformers anthropic openai-whisper PyMuPDF -q

# STEP 2: Import
from edusense_rag import EduSenseRAG

# STEP 3: Initialize
rag = EduSenseRAG(
    anthropic_api_key = 'YOUR_KEY_HERE',           # get from console.anthropic.com
    kb_dir            = '/content/drive/MyDrive/edusense_kb',
    output_dir        = '/content/drive/MyDrive/edusense_notebooks',
    whisper_model     = 'base'                     # or 'small' for better quality
)

# STEP 4: Upload your textbooks (run once)
rag.upload_textbooks([
    '/content/drive/MyDrive/textbooks/intro_to_algorithms.pdf',
    '/content/drive/MyDrive/textbooks/data_structures.pdf',
    '/content/drive/MyDrive/textbooks/lecture_slides.pdf',
])
rag.knowledge_base_stats()

# STEP 5: Test with a transcript (no audio needed)
emotion_state = {
    'engagement':  {'positive': False, 'confidence': 0.35},
    'boredom':     {'positive': True,  'confidence': 0.71},
    'confusion':   {'positive': True,  'confidence': 0.68},
    'frustration': {'positive': False, 'confidence': 0.28},
}

notebook_path = rag.process_with_text(
    transcript    = "Today we are covering binary search trees. The left subtree contains smaller values and the right subtree contains larger values. Time complexity for search is O(log n) average case.",
    emotion_state = emotion_state
)
print(f"Generated: {notebook_path}")

# STEP 6: Full pipeline with audio file
notebook_path = rag.process_dissatisfaction(
    audio_path    = '/content/lecture_recording.mp3',
    emotion_state = emotion_state,
    last_n_secs   = 60
)

# Download notebook from Colab
from google.colab import files
files.download(notebook_path)
'''

if __name__ == '__main__':
    print("EduSense RAG System")
    print("===================")
    print(DEMO_USAGE)
