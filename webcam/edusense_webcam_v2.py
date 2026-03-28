"""
EduSense — Webcam System (CPU Version)
=======================================
Uses DeepFace for real-time emotion detection on CPU laptops.
No GPU required. Works on Windows.

Run:
    python edusense_webcam_v2.py

Install:
    pip install deepface opencv-python anthropic
    pip install chromadb sentence-transformers openai-whisper PyMuPDF
"""

import cv2
import time
import json
import os
import threading
import numpy as np
from pathlib import Path
from collections import deque
from deepface import DeepFace
import os
from dotenv import load_dotenv

load_dotenv()
 
# ─────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────
CONFIG = {
    'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),       # console.anthropic.com
    'kb_dir':            'Rag_system/edusense_kb',
    'output_dir':        'generated_notebooks',
    'trigger_threshold': 3,                      # consecutive alerts before trigger
    'cooldown_mins':     3,                      # minutes between triggers
    'analysis_interval': 3,                      # seconds between DeepFace analysis
    'smoothing_window':  5,                      # smooth predictions over N frames
}

EMOTION_NAMES = ['engagement', 'boredom', 'confusion', 'frustration']

EMOTION_COLORS = {
    'engagement':  (0,   200, 100),
    'boredom':     (0,   165, 255),
    'confusion':   (255, 200,   0),
    'frustration': (0,    50, 255),
}

# ─────────────────────────────────────────────
# DEEPFACE → EDUSENSE EMOTION MAPPER
# ─────────────────────────────────────────────

class DeepFacePredictor:
    """
    Maps DeepFace raw emotions to EduSense 4-emotion system.

    DeepFace emotions: angry, disgust, fear, happy, sad, surprise, neutral
    EduSense emotions: engagement, boredom, confusion, frustration
    """

    def __init__(self, smoothing_window=5):
        self.history   = deque(maxlen=smoothing_window)
        self.last_result = None
        print("✅ DeepFace predictor ready (CPU mode)")

    def analyze_frame(self, frame_bgr):
        """Run DeepFace on a single frame. Returns raw emotion dict or None."""
        try:
            result = DeepFace.analyze(
                frame_bgr,
                actions    = ['emotion'],
                enforce_detection = False,   # don't crash if no face
                silent     = True
            )
            # result is a list — take first face
            if isinstance(result, list):
                result = result[0]
            return result['emotion']          # dict of emotion→score
        except Exception:
            return None

    def map_emotions(self, raw: dict) -> dict:
        """
        Map DeepFace scores (0-100) to EduSense binary predictions.

        Mapping logic (based on psychology of student engagement):
        - Engagement:  high happy OR low neutral (active face)
        - Boredom:     high neutral + low happy (flat face)
        - Confusion:   high fear + surprise (furrowed brow, wide eyes)
        - Frustration: high angry + disgust (tense face)
        """
        happy    = raw.get('happy',    0)
        neutral  = raw.get('neutral',  0)
        fear     = raw.get('fear',     0)
        surprise = raw.get('surprise', 0)
        angry    = raw.get('angry',    0)
        disgust  = raw.get('disgust',  0)
        sad      = raw.get('sad',      0)

        # Compute EduSense scores (0-1)
        eng_score = (happy * 0.7 + (100 - neutral) * 0.3) / 100
        bor_score = (neutral * 0.6 + sad * 0.4) / 100
        con_score = (fear * 0.5 + surprise * 0.5) / 100
        fru_score = (angry * 0.6 + disgust * 0.4) / 100

        return {
            'engagement':  {'score': eng_score, 'raw': happy},
            'boredom':     {'score': bor_score, 'raw': neutral},
            'confusion':   {'score': con_score, 'raw': fear + surprise},
            'frustration': {'score': fru_score, 'raw': angry + disgust},
        }

    def predict(self, frame_bgr):
        """Full prediction pipeline with smoothing."""
        raw = self.analyze_frame(frame_bgr)
        if raw is None:
            return self.last_result  # return last known if no face

        mapped = self.map_emotions(raw)
        self.history.append(mapped)

        # Smooth scores over window
        smoothed = {}
        for e in EMOTION_NAMES:
            avg_score = np.mean([h[e]['score'] for h in self.history])
            smoothed[e] = {
                'score':      avg_score,
                'positive':   avg_score > 0.35,    # threshold
                'confidence': avg_score,
            }

        self.last_result = smoothed
        return smoothed


# ─────────────────────────────────────────────
# STUDENT MONITOR — timing + trigger logic
# ─────────────────────────────────────────────

class StudentMonitor:
    def __init__(self, trigger_threshold=3, cooldown_mins=3):
        self.history            = []
        self.trigger_count      = 0
        self.trigger_threshold  = trigger_threshold
        self.last_trigger       = 0
        self.cooldown           = cooldown_mins * 60
        self.session_start      = time.time()
        self.total_triggers     = 0

    def add_prediction(self, emotion_state) -> bool:
        """Returns True if RAG should be triggered."""
        self.history.append({
            'time':  time.time(),
            'state': emotion_state
        })

        # Dissatisfaction = any of the 3 reliable emotions
        # (we exclude engagement — AUC was too low)
        disengaged = any([
            emotion_state['boredom']['positive'],
            emotion_state['confusion']['positive'],
            emotion_state['frustration']['positive'],
        ])

        if disengaged:
            self.trigger_count += 1
        else:
            self.trigger_count = max(0, self.trigger_count - 1)

        now = time.time()
        should_trigger = (
            self.trigger_count >= self.trigger_threshold and
            now - self.last_trigger > self.cooldown
        )

        if should_trigger:
            self.last_trigger   = now
            self.trigger_count  = 0
            self.total_triggers += 1
            return True

        return False

    def get_engagement_rate(self):
        if not self.history:
            return 100.0
        positive = sum(
            1 for h in self.history
            if not any(h['state'][e]['positive']
                      for e in ['boredom', 'confusion', 'frustration'])
        )
        return positive / len(self.history) * 100

    def session_duration(self):
        return time.time() - self.session_start


# ─────────────────────────────────────────────
# UI DRAWING
# ─────────────────────────────────────────────

class EduSenseUI:

    @staticmethod
    def draw_panel(frame, emotion_state, monitor, trigger_count, trigger_threshold):
        h, w = frame.shape[:2]

        # ── Left panel: emotions ──────────────────
        panel_x, panel_y = 15, 15
        cv2.rectangle(frame,
                     (panel_x-8, panel_y-8),
                     (panel_x+230, panel_y+155),
                     (15, 15, 15), -1)
        cv2.rectangle(frame,
                     (panel_x-8, panel_y-8),
                     (panel_x+230, panel_y+155),
                     (50, 50, 50), 1)

        cv2.putText(frame, 'EduSense', (panel_x, panel_y+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        for i, name in enumerate(EMOTION_NAMES):
            y       = panel_y + 35 + i * 30
            state   = emotion_state[name]
            score   = state['confidence']
            active  = state['positive']
            color   = EMOTION_COLORS[name] if active else (60, 60, 60)

            # Label
            cv2.putText(frame, name[:3].upper(), (panel_x, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                       EMOTION_COLORS[name] if active else (100,100,100), 1)

            # Bar bg
            cv2.rectangle(frame, (panel_x+38, y),
                         (panel_x+200, y+14), (40, 40, 40), -1)

            # Bar fill
            bw = int(162 * score)
            cv2.rectangle(frame, (panel_x+38, y),
                         (panel_x+38+bw, y+14), color, -1)

            # Score
            cv2.putText(frame, f'{score:.0%}',
                       (panel_x+205, y+11),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                       EMOTION_COLORS[name] if active else (80,80,80), 1)

        # ── Bottom panel: session stats ───────────
        bx, by = 15, h - 100
        cv2.rectangle(frame, (bx-8, by-8), (bx+230, h-8),
                     (15, 15, 15), -1)

        elapsed    = int(monitor.session_duration())
        mins, secs = divmod(elapsed, 60)
        eng_rate   = monitor.get_engagement_rate()
        eng_color  = (0, 200, 100) if eng_rate > 60 else (0, 100, 255)

        cv2.putText(frame, f'Time     {mins:02d}:{secs:02d}',
                   (bx, by+15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (140, 140, 140), 1)
        cv2.putText(frame, f'Focused  {eng_rate:.0f}%',
                   (bx, by+33), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, eng_color, 1)
        cv2.putText(frame, f'Alerts   {monitor.total_triggers}',
                   (bx, by+51), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (0, 165, 255), 1)

        # Trigger progress
        if trigger_count > 0:
            for t in range(trigger_threshold):
                color = (0, 50, 255) if t < trigger_count else (50, 50, 50)
                cx = bx + t * 22
                cv2.circle(frame, (cx+10, by+68), 7, color, -1)
            cv2.putText(frame, 'alert level',
                       (bx + trigger_threshold*22 + 5, by+72),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)

        return frame

    @staticmethod
    def draw_alert(frame, message):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-45), (w, h//2+45),
                     (0, 0, 150), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        # Center text
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(frame, message, (tx, h//2+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        return frame

    @staticmethod
    def draw_no_face(frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, 'No face detected',
                   (w//2-80, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        return frame

    @staticmethod
    def draw_footer(frame):
        h, w = frame.shape[:2]
        cv2.putText(frame,
                   'Q: Quit  |  S: Summary  |  T: Trigger  |  EduSense v1.0',
                   (w//2-200, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)
        return frame


# ─────────────────────────────────────────────
# MAIN SYSTEM
# ─────────────────────────────────────────────

class EduSenseWebcam:
    def __init__(self, config):
        self.config = config
        print("\n🚀 EduSense — CPU Webcam System")
        print("   Feature extractor: DeepFace (CPU optimized)")

        self.predictor = DeepFacePredictor(
            smoothing_window=config['smoothing_window']
        )
        self.monitor = StudentMonitor(
            trigger_threshold = config['trigger_threshold'],
            cooldown_mins     = config['cooldown_mins']
        )
        self.ui = EduSenseUI()

        # RAG system
        self.rag = None
        if config.get('anthropic_api_key', 'YOUR_KEY_HERE') != 'YOUR_KEY_HERE':
            try:
                from edusense_rag import EduSenseRAG
                self.rag = EduSenseRAG(
                    anthropic_api_key = config['anthropic_api_key'],
                    kb_dir            = config['kb_dir'],
                    output_dir        = config['output_dir'],
                )
                print("✅ RAG system connected")
            except Exception as e:
                print(f"⚠️  RAG not loaded: {e}")
                print("   Running without notebook generation")
        else:
            print("⚠️  No API key — running without notebook generation")

        os.makedirs(config['output_dir'], exist_ok=True)

        self.alert_msg    = None
        self.alert_until  = 0
        self.emotion_state = None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam. Check your camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n" + "="*50)
        print("✅ Monitoring started")
        print("   Q — quit")
        print("   S — session summary")
        print("   T — manual trigger")
        print("="*50 + "\n")

        last_analysis = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            # Run DeepFace every N seconds
            if now - last_analysis >= self.config['analysis_interval']:
                result = self.predictor.predict(frame)
                last_analysis = now

                if result is not None:
                    self.emotion_state = result

                    # Check trigger
                    triggered = self.monitor.add_prediction(result)
                    if triggered:
                        self._handle_trigger(result)

                    # Print to console
                    detected = [e for e in EMOTION_NAMES
                               if result[e]['positive']]
                    status = ', '.join(detected) if detected else 'focused'
                    elapsed = int(self.monitor.session_duration())
                    m, s   = divmod(elapsed, 60)
                    print(f"[{m:02d}:{s:02d}] {status:<40} "
                          f"trigger={self.monitor.trigger_count}/"
                          f"{self.config['trigger_threshold']}")

            # ── DRAW UI ──────────────────────────
            display = frame.copy()

            if self.emotion_state:
                display = self.ui.draw_panel(
                    display, self.emotion_state,
                    self.monitor,
                    self.monitor.trigger_count,
                    self.config['trigger_threshold']
                )
            else:
                display = self.ui.draw_no_face(display)

            if self.alert_msg and time.time() < self.alert_until:
                display = self.ui.draw_alert(display, self.alert_msg)

            display = self.ui.draw_footer(display)
            cv2.imshow('EduSense', display)

            # ── KEY HANDLERS ─────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._print_summary()
            elif key == ord('t'):
                if self.emotion_state:
                    print("\n[Manual trigger]")
                    self._handle_trigger(self.emotion_state)
                else:
                    print("No emotion state yet — wait a few seconds")

        cap.release()
        cv2.destroyAllWindows()
        self._print_summary()

    def _handle_trigger(self, emotion_state):
        detected = [e for e in EMOTION_NAMES
                   if emotion_state[e]['positive']]
        label    = ', '.join(detected) if detected else 'disengaged'
        print(f"\n⚠️  TRIGGER — {label}")

        self.alert_msg   = f"Generating notebook: {label}..."
        self.alert_until = time.time() + 10

        if self.rag:
            def generate():
                try:
                    path = self.rag.process_with_text(
                        transcript    = f"Student appears {label} during lecture",
                        emotion_state = emotion_state
                    )
                    name = Path(path).name
                    self.alert_msg   = f"✅ Notebook ready: {name}"
                    self.alert_until = time.time() + 12
                    print(f"✅ Notebook: {path}")
                except Exception as e:
                    print(f"❌ RAG error: {e}")
                    self.alert_msg = None

            threading.Thread(target=generate, daemon=True).start()
        else:
            print("   (RAG not configured — skipping notebook generation)")
            self.alert_msg   = f"Detected: {label} — add API key for notebooks"
            self.alert_until = time.time() + 5

    def _print_summary(self):
        elapsed    = self.monitor.session_duration()
        m, s       = divmod(int(elapsed), 60)
        eng_rate   = self.monitor.get_engagement_rate()
        print(f"\n{'='*45}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*45}")
        print(f"  Duration:        {m:02d}:{s:02d}")
        print(f"  Focused:         {eng_rate:.1f}%")
        print(f"  Checks:          {len(self.monitor.history)}")
        print(f"  Notebooks made:  {self.monitor.total_triggers}")
        print(f"{'='*45}\n")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    system = EduSenseWebcam(CONFIG)
    system.run()
