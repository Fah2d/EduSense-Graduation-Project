"""
DeepFace Live Tester
====================
Simple tkinter app to see DeepFace emotion output in real-time.

Run: python deepface_test.py
Install: pip install deepface opencv-python pillow
"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from deepface import DeepFace
import threading
import time
import numpy as np

# ── Config ────────────────────────────────────────────────
ANALYSIS_INTERVAL = 1.5   # seconds between DeepFace runs
CAM_INDEX         = 0     # 0 = default webcam

# Colors
BG      = '#0d0d14'
SURFACE = '#16161f'
BORDER  = '#2a2a3d'
TEXT    = '#e2e8f0'
MUTED   = '#64748b'

EMOTION_COLORS = {
    'angry':    '#f43f5e',
    'disgust':  '#a855f7',
    'fear':     '#f97316',
    'happy':    '#22c55e',
    'sad':      '#3b82f6',
    'surprise': '#eab308',
    'neutral':  '#94a3b8',
}

EDUSENSE_COLORS = {
    'engagement':  '#6ee7b7',
    'boredom':     '#fb923c',
    'confusion':   '#38bdf8',
    'frustration': '#f43f5e',
}


class DeepFaceTester:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DeepFace Live Tester — EduSense")
        self.root.configure(bg=BG)
        self.root.geometry("900x620")
        self.root.resizable(False, False)

        self.cap          = None
        self.running      = False
        self.last_result  = None
        self.frame_count  = 0
        self.last_analysis = 0
        self.current_frame = None  # shared frame

        self._build_ui()
        self._start_camera()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _build_ui(self):
        # ── Header ───────────────────────────────────────
        header = tk.Frame(self.root, bg=BG, pady=10)
        header.pack(fill='x', padx=20)

        tk.Label(header, text="DeepFace", font=('Courier', 18, 'bold'),
                bg=BG, fg='#6ee7b7').pack(side='left')
        tk.Label(header, text=" Live Tester", font=('Courier', 18),
                bg=BG, fg=TEXT).pack(side='left')

        self.status_label = tk.Label(header, text="● Starting...",
                                    font=('Courier', 10),
                                    bg=BG, fg=MUTED)
        self.status_label.pack(side='right')

        # ── Main layout ───────────────────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Left: webcam
        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='y')

        self.cam_label = tk.Label(left, bg='#000', width=480, height=360)
        self.cam_label.pack()

        # FPS label
        self.fps_label = tk.Label(left, text="FPS: --",
                                  font=('Courier', 9), bg=BG, fg=MUTED)
        self.fps_label.pack(pady=4)

        # Right: results
        right = tk.Frame(main, bg=BG, padx=16)
        right.pack(side='left', fill='both', expand=True)

        # ── Raw DeepFace emotions ─────────────────────────
        tk.Label(right, text="RAW DEEPFACE EMOTIONS",
                font=('Courier', 9, 'bold'), bg=BG, fg=MUTED).pack(anchor='w', pady=(0,8))

        self.raw_bars   = {}
        self.raw_labels = {}

        raw_frame = tk.Frame(right, bg=SURFACE,
                            relief='flat', padx=12, pady=12)
        raw_frame.pack(fill='x', pady=(0, 14))

        emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']
        for e in emotions:
            row = tk.Frame(raw_frame, bg=SURFACE)
            row.pack(fill='x', pady=3)

            tk.Label(row, text=e[:7].upper(), font=('Courier', 8),
                    bg=SURFACE, fg=EMOTION_COLORS[e], width=8,
                    anchor='w').pack(side='left')

            bar_bg = tk.Frame(row, bg=BORDER, height=10, width=200)
            bar_bg.pack(side='left', padx=6)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=EMOTION_COLORS[e], height=10, width=0)
            bar_fill.pack(side='left', fill='y')

            pct_lbl = tk.Label(row, text="0%", font=('Courier', 8),
                              bg=SURFACE, fg=EMOTION_COLORS[e], width=5)
            pct_lbl.pack(side='left')

            self.raw_bars[e]   = bar_fill
            self.raw_labels[e] = pct_lbl

        # Dominant emotion
        dom_row = tk.Frame(raw_frame, bg=SURFACE)
        dom_row.pack(fill='x', pady=(8,0))
        tk.Label(dom_row, text="DOMINANT:", font=('Courier', 9),
                bg=SURFACE, fg=MUTED).pack(side='left')
        self.dominant_label = tk.Label(dom_row, text="--",
                                       font=('Courier', 9, 'bold'),
                                       bg=SURFACE, fg=TEXT)
        self.dominant_label.pack(side='left', padx=8)

        # ── EduSense mapped emotions ──────────────────────
        tk.Label(right, text="EDUSENSE MAPPING",
                font=('Courier', 9, 'bold'), bg=BG, fg=MUTED).pack(anchor='w', pady=(0,8))

        self.edu_bars   = {}
        self.edu_labels = {}
        self.edu_dots   = {}

        edu_frame = tk.Frame(right, bg=SURFACE,
                            relief='flat', padx=12, pady=12)
        edu_frame.pack(fill='x', pady=(0, 14))

        edu_emotions = ['engagement', 'boredom', 'confusion', 'frustration']
        for e in edu_emotions:
            row = tk.Frame(edu_frame, bg=SURFACE)
            row.pack(fill='x', pady=4)

            # Active dot
            dot = tk.Label(row, text="●", font=('Courier', 10),
                          bg=SURFACE, fg=BORDER)
            dot.pack(side='left', padx=(0,4))

            tk.Label(row, text=e[:8].upper(), font=('Courier', 8),
                    bg=SURFACE, fg=EDUSENSE_COLORS[e], width=9,
                    anchor='w').pack(side='left')

            bar_bg = tk.Frame(row, bg=BORDER, height=12, width=160)
            bar_bg.pack(side='left', padx=6)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=EDUSENSE_COLORS[e], height=12, width=0)
            bar_fill.pack(side='left', fill='y')

            pct_lbl = tk.Label(row, text="0%", font=('Courier', 8),
                              bg=SURFACE, fg=EDUSENSE_COLORS[e], width=5)
            pct_lbl.pack(side='left')

            self.edu_bars[e]   = bar_fill
            self.edu_labels[e] = pct_lbl
            self.edu_dots[e]   = dot

        # ── Dissatisfaction alert ─────────────────────────
        self.alert_frame = tk.Frame(right, bg='#1a0a0a',
                                   relief='flat', padx=12, pady=10)
        self.alert_frame.pack(fill='x')

        self.alert_label = tk.Label(self.alert_frame,
                                   text="✅  Student appears FOCUSED",
                                   font=('Courier', 9, 'bold'),
                                   bg='#1a0a0a', fg='#22c55e')
        self.alert_label.pack()

        # ── Threshold controls ────────────────────────────
        tk.Label(right, text="THRESHOLD (adjust if needed)",
                font=('Courier', 9, 'bold'), bg=BG, fg=MUTED).pack(anchor='w', pady=(12,4))

        thresh_frame = tk.Frame(right, bg=SURFACE, padx=12, pady=8)
        thresh_frame.pack(fill='x')

        tk.Label(thresh_frame, text="Positive if score >",
                font=('Courier', 8), bg=SURFACE, fg=MUTED).pack(side='left')

        self.threshold_var = tk.DoubleVar(value=0.35)
        thresh_slider = ttk.Scale(thresh_frame, from_=0.1, to=0.8,
                                  variable=self.threshold_var,
                                  orient='horizontal', length=120)
        thresh_slider.pack(side='left', padx=8)

        self.thresh_label = tk.Label(thresh_frame, text="0.35",
                                    font=('Courier', 8, 'bold'),
                                    bg=SURFACE, fg=TEXT)
        self.thresh_label.pack(side='left')

    def _start_camera(self):
        self.cap     = cv2.VideoCapture(CAM_INDEX)
        self.running = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Camera loop in background thread
        threading.Thread(target=self._camera_loop, daemon=True).start()
        # Analysis loop in background thread
        threading.Thread(target=self._analysis_loop, daemon=True).start()
        # UI update loop
        self._update_ui()

    def _camera_loop(self):
        prev_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Mirror
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()

            # Draw face box if result available
            if self.last_result:
                dom = self.last_result.get('dominant', '')
                color = EMOTION_COLORS.get(dom, '#ffffff')
                # Convert hex to BGR
                h = color.lstrip('#')
                rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                bgr = (rgb[2], rgb[1], rgb[0])
                cv2.putText(frame, dom.upper(), (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

            # FPS
            now  = time.time()
            fps  = 1 / max(now - prev_time, 0.001)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, frame.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

            # Convert to tkinter image
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)

            self._current_photo = photo
            self._current_fps   = fps

    def _analysis_loop(self):
        while self.running:
            now = time.time()
            if now - self.last_analysis < ANALYSIS_INTERVAL:
                time.sleep(0.1)
                continue

            self.last_analysis = now
            self.status_label.config(text="● Analyzing...", fg='#f59e0b')

            try:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()

                result = DeepFace.analyze(
                    frame,
                    actions           = ['emotion'],
                    enforce_detection = False,
                    silent            = True
                )
                if isinstance(result, list):
                    result = result[0]

                raw = result['emotion']
                dom = result.get('dominant_emotion', max(raw, key=raw.get))

                # Map to EduSense
                happy    = raw.get('happy',    0)
                neutral  = raw.get('neutral',  0)
                fear     = raw.get('fear',     0)
                surprise = raw.get('surprise', 0)
                angry    = raw.get('angry',    0)
                disgust  = raw.get('disgust',  0)
                sad      = raw.get('sad',      0)

                edu = {
                    'engagement':  (happy * 0.7 + (100 - neutral) * 0.3) / 100,
                    'boredom':     (neutral * 0.6 + sad * 0.4) / 100,
                    'confusion':   (fear * 0.5 + surprise * 0.5) / 100,
                    'frustration': (angry * 0.6 + disgust * 0.4) / 100,
                }

                self.last_result = {
                    'raw': raw,
                    'dominant': dom,
                    'edu': edu,
                }

                self.status_label.config(text="● Live", fg='#6ee7b7')

            except Exception as e:
                self.status_label.config(text=f"● Error: {str(e)[:30]}", fg='#f43f5e')

    def _update_ui(self):
        if not self.running:
            return

        # Update webcam
        if hasattr(self, '_current_photo'):
            self.cam_label.config(image=self._current_photo)
        if hasattr(self, '_current_fps'):
            self.fps_label.config(text=f"FPS: {self._current_fps:.0f}")

        # Update threshold label
        t = self.threshold_var.get()
        self.thresh_label.config(text=f"{t:.2f}")

        # Update bars if result available
        if self.last_result:
            raw = self.last_result['raw']
            edu = self.last_result['edu']
            dom = self.last_result['dominant']

            # Raw bars
            for e, score in raw.items():
                if e in self.raw_bars:
                    w = int(score * 2)  # 0-100 → 0-200px
                    self.raw_bars[e].config(width=max(0, w))
                    self.raw_labels[e].config(text=f"{score:.0f}%")

            self.dominant_label.config(
                text=dom.upper(),
                fg=EMOTION_COLORS.get(dom, TEXT)
            )

            # EduSense bars
            threshold  = self.threshold_var.get()
            dissat     = False

            for e, score in edu.items():
                w      = int(score * 160)
                active = score > threshold
                color  = EDUSENSE_COLORS[e]

                self.edu_bars[e].config(width=max(0, w))
                self.edu_labels[e].config(text=f"{score:.0%}")
                self.edu_dots[e].config(
                    fg=color if active else BORDER
                )

                if e in ['boredom', 'confusion', 'frustration'] and active:
                    dissat = True

            # Alert
            if dissat:
                self.alert_frame.config(bg='#1a0808')
                self.alert_label.config(
                    text="⚠️  DISSATISFIED DETECTED",
                    bg='#1a0808', fg='#f43f5e'
                )
            else:
                self.alert_frame.config(bg='#081a0e')
                self.alert_label.config(
                    text="✅  Student appears FOCUSED",
                    bg='#081a0e', fg='#22c55e'
                )

        self.root.after(100, self._update_ui)

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    print("Starting DeepFace Live Tester...")
    print("Install: pip install deepface opencv-python pillow")
    DeepFaceTester()
