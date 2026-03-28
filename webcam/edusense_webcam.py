"""
EduSense — Local Webcam System
================================
Run on your laptop with:
    python edusense_webcam.py

Requirements:
    pip install opencv-python torch torchvision transformers anthropic
    pip install chromadb sentence-transformers openai-whisper PyMuPDF
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import threading
import queue
from pathlib import Path
from collections import deque
from PIL import Image
from transformers import ViTModel, AutoFeatureExtractor

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
CONFIG = {
    'model_path':        'best_edusense.pth',      # your trained model
    'anthropic_api_key': 'YOUR_KEY_HERE',           # from console.anthropic.com
    'kb_dir':            'edusense_kb',             # knowledge base folder
    'output_dir':        'generated_notebooks',     # where notebooks are saved
    'frames_per_clip':   30,                        # frames to collect per prediction
    'frame_interval':    0.33,                      # seconds between frames (~3fps)
    'trigger_threshold': 3,                         # consecutive disengaged before trigger
    'cooldown_mins':     3,                         # minutes between triggers
    'device':            'cuda' if torch.cuda.is_available() else 'cpu',
}

EMOTION_NAMES  = ['engagement', 'boredom', 'confusion', 'frustration']
THRESHOLDS_INF = {'engagement': 0.5, 'boredom': 0.575,
                  'confusion': 0.525, 'frustration': 0.5}

EMOTION_COLORS = {
    'engagement':  (0,   200, 100),   # green
    'boredom':     (0,   165, 255),   # orange
    'confusion':   (255, 200,   0),   # cyan
    'frustration': (0,    50, 255),   # red
}

# ─────────────────────────────────────────────
# MODEL DEFINITIONS (same as training)
# ─────────────────────────────────────────────

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_basis=8):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.num_basis     = num_basis
        self.spline_coeffs = nn.Parameter(
            torch.randn(in_features, out_features, num_basis) * 0.1
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_norm    = torch.tanh(x).unsqueeze(-1)
        basis_idx = torch.linspace(0, np.pi, self.num_basis, device=x.device)
        basis     = torch.cos(x_norm * basis_idx)
        return torch.einsum('bin,ion->bo', basis, self.spline_coeffs) + self.linear(x)


class EduSenseTransformer(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, num_layers=4,
                 ff_dim=2048, kan_hidden=256, num_emotions=4, dropout=0.3):
        super().__init__()
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, 31, embedding_dim) * 0.02)
        self.embed_drop = nn.Dropout(dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        self.kan      = KANLayer(embedding_dim, kan_hidden)
        self.kan_norm = nn.LayerNorm(kan_hidden)
        self.kan_drop = nn.Dropout(dropout)
        self.coral_heads = nn.ModuleList([
            nn.Linear(kan_hidden, 1) for _ in range(num_emotions)
        ])

    def forward(self, x):
        batch = x.size(0)
        cls   = self.cls_token.expand(batch, -1, -1)
        x     = torch.cat([cls, x], dim=1)
        x     = self.embed_drop(x + self.pos_embed)
        x     = self.transformer(x)
        out   = self.kan_drop(F.relu(self.kan_norm(self.kan(x[:, 0, :]))))
        return [head(out) for head in self.coral_heads]


# ─────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device    = device
        print("Loading AffectNet-ViT...")
        self.processor = AutoFeatureExtractor.from_pretrained(
            'motheecreator/vit-Facial-Expression-Recognition'
        )
        self.model = ViTModel.from_pretrained(
            'motheecreator/vit-Facial-Expression-Recognition'
        ).to(device).eval()
        print("✅ Feature extractor ready")

    def extract_batch(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out  = self.model(**inputs)
            embs = out.last_hidden_state[:, 0, :].cpu().numpy()
        return embs  # (N, 768)


# ─────────────────────────────────────────────
# EMOTION PREDICTOR
# ─────────────────────────────────────────────

class EmotionPredictor:
    def __init__(self, model, extractor, device='cpu',
                 buffer_size=30):
        self.model      = model
        self.extractor  = extractor
        self.device     = device
        self.buffer     = deque(maxlen=buffer_size)
        self.last_result = None

    def add_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((224, 224))
        emb = self.extractor.extract_batch([pil])[0]
        self.buffer.append(emb)

    def predict(self):
        if len(self.buffer) < 30:
            return None

        seq = np.stack(list(self.buffer))
        seq = torch.tensor(seq, dtype=torch.float32)
        seq = F.normalize(seq, p=2, dim=1)
        seq = seq.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq)

        result = {}
        for i, name in enumerate(EMOTION_NAMES):
            prob     = torch.sigmoid(logits[i]).item()
            threshold = THRESHOLDS_INF[name]
            result[name] = {
                'positive':   prob > threshold,
                'confidence': prob
            }

        self.last_result = result
        return result


# ─────────────────────────────────────────────
# STUDENT MONITOR (timing + trigger logic)
# ─────────────────────────────────────────────

class StudentMonitor:
    def __init__(self, trigger_threshold=3, cooldown_mins=3):
        self.history           = []
        self.trigger_count     = 0
        self.trigger_threshold = trigger_threshold
        self.last_trigger      = 0
        self.cooldown          = cooldown_mins * 60
        self.session_start     = time.time()
        self.total_triggers    = 0

    def add_prediction(self, emotion_state):
        ts = time.time()
        self.history.append({
            'time':       ts,
            'elapsed':    ts - self.session_start,
            'state':      emotion_state
        })

        # Is student disengaged?
        disengaged = (
            not emotion_state['engagement']['positive'] and
            any(emotion_state[e]['positive']
                for e in ['boredom', 'confusion', 'frustration'])
        )

        if disengaged:
            self.trigger_count += 1
        else:
            self.trigger_count = max(0, self.trigger_count - 1)

        # Should trigger?
        should_trigger = (
            self.trigger_count >= self.trigger_threshold and
            ts - self.last_trigger > self.cooldown
        )

        if should_trigger:
            self.last_trigger   = ts
            self.trigger_count  = 0
            self.total_triggers += 1
            return True

        return False

    def get_engagement_rate(self):
        if not self.history:
            return 100.0
        engaged = sum(
            1 for h in self.history
            if h['state']['engagement']['positive']
        )
        return engaged / len(self.history) * 100

    def session_duration(self):
        return time.time() - self.session_start


# ─────────────────────────────────────────────
# WEBCAM DISPLAY
# ─────────────────────────────────────────────

class WebcamDisplay:
    """Handles all OpenCV drawing on the webcam frame"""

    @staticmethod
    def draw_emotion_bars(frame, emotion_state, x=20, y=50):
        h, w = frame.shape[:2]

        # Background panel
        cv2.rectangle(frame, (x-10, y-35), (x+230, y+130), (20, 20, 20), -1)
        cv2.rectangle(frame, (x-10, y-35), (x+230, y+130), (60, 60, 60), 1)
        cv2.putText(frame, 'EDUSENSE', (x, y-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        for i, name in enumerate(EMOTION_NAMES):
            bar_y   = y + i * 30
            state   = emotion_state[name]
            conf    = state['confidence']
            active  = state['positive']
            color   = EMOTION_COLORS[name]

            # Emotion label
            label = name[:3].upper()
            cv2.putText(frame, label, (x, bar_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Bar background
            cv2.rectangle(frame, (x+35, bar_y), (x+200, bar_y+14),
                         (50, 50, 50), -1)

            # Bar fill
            bar_w = int(165 * conf)
            bar_color = color if active else (80, 80, 80)
            cv2.rectangle(frame, (x+35, bar_y), (x+35+bar_w, bar_y+14),
                         bar_color, -1)

            # Confidence text
            cv2.putText(frame, f'{conf:.0%}', (x+205, bar_y+11),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                       color if active else (100, 100, 100), 1)

        return frame

    @staticmethod
    def draw_status(frame, monitor, trigger_count, trigger_threshold,
                    frames_collected, frames_needed):
        h, w = frame.shape[:2]
        x, y = 20, h - 120

        # Background
        cv2.rectangle(frame, (x-10, y-10), (x+230, h-10),
                     (20, 20, 20), -1)

        # Session time
        elapsed = int(monitor.session_duration())
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f'SESSION  {mins:02d}:{secs:02d}',
                   (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (150, 150, 150), 1)

        # Engagement rate
        eng_rate = monitor.get_engagement_rate()
        eng_color = (0, 200, 100) if eng_rate > 60 else (0, 100, 255)
        cv2.putText(frame, f'ENGAGED  {eng_rate:.0f}%',
                   (x, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, eng_color, 1)

        # Trigger counter
        cv2.putText(frame, f'ALERT    {monitor.total_triggers}',
                   (x, y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (0, 165, 255), 1)

        # Frame buffer progress
        prog  = frames_collected / frames_needed
        bw    = 165
        cv2.rectangle(frame, (x+35, y+65), (x+35+bw, y+77),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (x+35, y+65),
                     (x+35+int(bw*prog), y+77), (100, 100, 200), -1)
        cv2.putText(frame, 'BUF', (x, y+75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Trigger progress
        if trigger_count > 0:
            cv2.putText(frame, f'TRIGGER {trigger_count}/{trigger_threshold}',
                       (x, y+95), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (0, 50, 255), 1)

        return frame

    @staticmethod
    def draw_alert(frame, message, alpha=1.0):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-40), (w, h//2+40),
                     (0, 0, 180), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame, message, (w//2-200, h//2+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    @staticmethod
    def draw_waiting(frame, frames_collected, frames_needed):
        h, w  = frame.shape[:2]
        pct   = frames_collected / frames_needed
        msg   = f'Collecting frames... {frames_collected}/{frames_needed}'
        cv2.putText(frame, msg, (w//2-150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Progress bar at top
        cv2.rectangle(frame, (0, 0), (int(w*pct), 5), (100, 100, 200), -1)
        return frame


# ─────────────────────────────────────────────
# MAIN WEBCAM LOOP
# ─────────────────────────────────────────────

class EduSenseWebcam:
    def __init__(self, config: dict):
        self.config   = config
        self.device   = config['device']
        print(f"\n🚀 EduSense Webcam System")
        print(f"   Device: {self.device}")

        # Load components
        print("\nLoading models...")
        self.extractor = FeatureExtractor(self.device)

        # Load trained model
        model = EduSenseTransformer().to(self.device)
        if os.path.exists(config['model_path']):
            ckpt = torch.load(config['model_path'],
                            map_location=self.device,
                            weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Model loaded from {config['model_path']}")
        else:
            print(f"⚠️ Model not found at {config['model_path']} — running in demo mode")

        self.predictor = EmotionPredictor(model, self.extractor, self.device)
        self.monitor   = StudentMonitor(
            trigger_threshold = config['trigger_threshold'],
            cooldown_mins     = config['cooldown_mins']
        )
        self.display   = WebcamDisplay()

        # RAG system (optional)
        self.rag = None
        if config.get('anthropic_api_key') != 'YOUR_KEY_HERE':
            try:
                from edusense_rag import EduSenseRAG
                self.rag = EduSenseRAG(
                    anthropic_api_key = config['anthropic_api_key'],
                    kb_dir            = config['kb_dir'],
                    output_dir        = config['output_dir'],
                )
                print("✅ RAG system ready")
            except Exception as e:
                print(f"⚠️ RAG not loaded: {e}")

        # Alert state
        self.alert_msg   = None
        self.alert_until = 0

        os.makedirs(config['output_dir'], exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "="*50)
        print("✅ EduSense monitoring started")
        print("   Press 'q' to quit")
        print("   Press 's' to show session summary")
        print("   Press 't' to manually trigger RAG")
        print("="*50 + "\n")

        frame_count       = 0
        last_frame_time   = 0
        last_predict_time = 0
        emotion_state     = None
        prediction_interval = self.config['frames_per_clip'] * \
                              self.config['frame_interval']

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            # Collect frames at specified interval
            if now - last_frame_time >= self.config['frame_interval']:
                self.predictor.add_frame(frame)
                last_frame_time = now
                frame_count    += 1

            # Run prediction every N frames
            if now - last_predict_time >= prediction_interval:
                result = self.predictor.predict()
                if result is not None:
                    emotion_state     = result
                    last_predict_time = now

                    # Check trigger
                    triggered = self.monitor.add_prediction(emotion_state)
                    if triggered:
                        self._handle_trigger(emotion_state)

            # ── DRAW UI ──────────────────────────────
            display_frame = frame.copy()

            if emotion_state:
                display_frame = self.display.draw_emotion_bars(
                    display_frame, emotion_state
                )
                display_frame = self.display.draw_status(
                    display_frame, self.monitor,
                    self.monitor.trigger_count,
                    self.config['trigger_threshold'],
                    len(self.predictor.buffer),
                    self.config['frames_per_clip']
                )
            else:
                display_frame = self.display.draw_waiting(
                    display_frame,
                    len(self.predictor.buffer),
                    self.config['frames_per_clip']
                )

            # Show alert if active
            if self.alert_msg and time.time() < self.alert_until:
                display_frame = self.display.draw_alert(
                    display_frame, self.alert_msg
                )

            # Title bar
            cv2.putText(display_frame, 'EduSense | Classroom Intelligence',
                       (frame.shape[1]//2 - 160, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

            cv2.imshow('EduSense', display_frame)

            # ── KEY HANDLERS ─────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._print_summary()
            elif key == ord('t') and emotion_state:
                print("\n[Manual trigger]")
                self._handle_trigger(emotion_state)

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    def _handle_trigger(self, emotion_state):
        """Called when dissatisfaction is detected"""
        detected = [e for e in EMOTION_NAMES
                   if emotion_state[e]['positive']]
        print(f"\n⚠️  TRIGGER — Detected: {', '.join(detected)}")

        self.alert_msg   = f"Generating notebook... ({', '.join(detected)})"
        self.alert_until = time.time() + 8

        if self.rag:
            # Run RAG in background thread so webcam doesn't freeze
            def generate():
                try:
                    # Use a dummy transcript — replace with real audio
                    path = self.rag.process_with_text(
                        transcript    = "Student appears confused during lecture",
                        emotion_state = emotion_state
                    )
                    self.alert_msg   = f"✅ Notebook ready: {Path(path).name}"
                    self.alert_until = time.time() + 10
                    print(f"✅ Notebook saved: {path}")
                except Exception as e:
                    print(f"❌ RAG error: {e}")
                    self.alert_msg = None

            threading.Thread(target=generate, daemon=True).start()
        else:
            print("   RAG not configured — skipping notebook generation")

    def _print_summary(self):
        elapsed = self.monitor.session_duration()
        mins, secs = divmod(int(elapsed), 60)
        print(f"\n{'='*50}")
        print(f"SESSION SUMMARY")
        print(f"{'='*50}")
        print(f"Duration:        {mins:02d}:{secs:02d}")
        print(f"Engagement rate: {self.monitor.get_engagement_rate():.1f}%")
        print(f"Total checks:    {len(self.monitor.history)}")
        print(f"Alerts triggered:{self.monitor.total_triggers}")
        print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    system = EduSenseWebcam(CONFIG)
    system.run()
