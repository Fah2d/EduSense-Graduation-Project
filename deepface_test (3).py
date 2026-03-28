"""
EduSense Emotion Live Tester — FER Version
Run: python deepface_test.py
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from fer import FER
import threading
import time
import numpy as np

BG    = '#0d0d14'
TEXT  = '#e2e8f0'
MUTED = '#64748b'

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


class App:
    def __init__(self, root):
        self.root        = root
        self.root.title("EduSense Emotion Tester")
        self.root.configure(bg=BG)
        self.root.geometry("860x520")

        self.cap         = cv2.VideoCapture(0)
        self.running     = True
        self.last_result = None
        self.photo       = None
        self.detector    = FER()

        self._build_ui()
        self._update()

        threading.Thread(target=self._analysis_loop, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _build_ui(self):
        tk.Label(self.root, text="EduSense — Emotion Live Tester",
                font=('Courier', 13, 'bold'), bg=BG, fg='#6ee7b7').pack(pady=8)

        self.status = tk.Label(self.root, text="Warming up...",
                              font=('Courier', 9), bg=BG, fg=MUTED)
        self.status.pack()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=16, pady=8)

        # Camera feed
        self.cam = tk.Label(body, bg='#000')
        self.cam.pack(side='left')

        # Right panel
        right = tk.Frame(body, bg=BG, padx=16)
        right.pack(side='left', fill='both', expand=True)

        # Raw FER emotions
        tk.Label(right, text="RAW FER EMOTIONS",
                font=('Courier', 8, 'bold'), bg=BG, fg=MUTED).pack(anchor='w', pady=(0,4))

        self.raw_bars   = {}
        self.raw_labels = {}
        raw_frame = tk.Frame(right, bg='#111118', padx=10, pady=8)
        raw_frame.pack(fill='x', pady=(0,12))

        for e in ['angry','disgust','fear','happy','sad','surprise','neutral']:
            row = tk.Frame(raw_frame, bg='#111118')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=e[:7].upper(), font=('Courier', 7),
                    bg='#111118', fg=EMOTION_COLORS[e],
                    width=8, anchor='w').pack(side='left')
            bg_bar = tk.Frame(row, bg='#1e1e2e', height=8, width=180)
            bg_bar.pack(side='left', padx=4)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=EMOTION_COLORS[e], height=8, width=0)
            fill.pack(side='left', fill='y')
            lbl = tk.Label(row, text="0%", font=('Courier', 7),
                          bg='#111118', fg=EMOTION_COLORS[e], width=5)
            lbl.pack(side='left')
            self.raw_bars[e]   = fill
            self.raw_labels[e] = lbl

        self.dom_label = tk.Label(right, text="DOMINANT: --",
                                 font=('Courier', 9, 'bold'), bg=BG, fg=TEXT)
        self.dom_label.pack(anchor='w', pady=(0,10))

        # EduSense mapping
        tk.Label(right, text="EDUSENSE MAPPING",
                font=('Courier', 8, 'bold'), bg=BG, fg=MUTED).pack(anchor='w', pady=(0,4))

        self.edu_bars   = {}
        self.edu_labels = {}
        self.edu_dots   = {}
        edu_frame = tk.Frame(right, bg='#111118', padx=10, pady=8)
        edu_frame.pack(fill='x', pady=(0,12))

        for e in ['engagement','boredom','confusion','frustration']:
            row = tk.Frame(edu_frame, bg='#111118')
            row.pack(fill='x', pady=3)
            dot = tk.Label(row, text="●", font=('Courier', 9),
                          bg='#111118', fg='#2a2a3d')
            dot.pack(side='left', padx=(0,3))
            tk.Label(row, text=e[:8].upper(), font=('Courier', 7),
                    bg='#111118', fg=EDUSENSE_COLORS[e],
                    width=9, anchor='w').pack(side='left')
            bg_bar = tk.Frame(row, bg='#1e1e2e', height=10, width=140)
            bg_bar.pack(side='left', padx=4)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=EDUSENSE_COLORS[e], height=10, width=0)
            fill.pack(side='left', fill='y')
            lbl = tk.Label(row, text="0%", font=('Courier', 7),
                          bg='#111118', fg=EDUSENSE_COLORS[e], width=5)
            lbl.pack(side='left')
            self.edu_bars[e]   = fill
            self.edu_labels[e] = lbl
            self.edu_dots[e]   = dot

        # Alert
        self.alert = tk.Label(right, text="✅  FOCUSED",
                             font=('Courier', 11, 'bold'),
                             bg=BG, fg='#22c55e')
        self.alert.pack(pady=8)

        # Score info
        tk.Label(right,
                text="engagement = happy\nboredom = neutral+sad\nconfusion = fear+surprise\nfrustration = angry+disgust",
                font=('Courier', 7), bg=BG, fg=MUTED,
                justify='left').pack(anchor='w')

    def _map_to_edusense(self, raw):
        happy    = raw.get('happy',    0)
        neutral  = raw.get('neutral',  0)
        fear     = raw.get('fear',     0)
        surprise = raw.get('surprise', 0)
        angry    = raw.get('angry',    0)
        disgust  = raw.get('disgust',  0)
        sad      = raw.get('sad',      0)

        return {
            'engagement':  (happy * 0.7 + (1 - neutral) * 0.3),
            'boredom':     (neutral * 0.6 + sad * 0.4),
            'confusion':   (fear * 0.5 + surprise * 0.5),
            'frustration': (angry * 0.6 + disgust * 0.4),
        }

    def _analysis_loop(self):
        time.sleep(1)
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            self.root.after(0, lambda: self.status.config(
                text="● Analyzing...", fg='#f59e0b'))

            try:
                result = self.detector.detect_emotions(frame)

                if result:
                    emotions = result[0]['emotions']  # dict of emotion→score (0-1)
                    dom      = max(emotions, key=emotions.get)
                    edu      = self._map_to_edusense(emotions)

                    self.last_result = {
                        'raw':      emotions,
                        'dominant': dom,
                        'edu':      edu,
                    }
                    self.root.after(0, lambda: self.status.config(
                        text="● Live", fg='#6ee7b7'))
                else:
                    self.root.after(0, lambda: self.status.config(
                        text="● No face detected", fg=MUTED))

            except Exception as e:
                self.root.after(0, lambda err=str(e): self.status.config(
                    text=f"● Error: {err[:35]}", fg='#f43f5e'))

            time.sleep(1.5)

    def _update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((400, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.cam.config(image=self.photo)

        if self.last_result:
            raw = self.last_result['raw']
            edu = self.last_result['edu']
            dom = self.last_result['dominant']

            # Raw bars (FER scores are 0-1)
            for e, score in raw.items():
                if e in self.raw_bars:
                    w = int(score * 180)
                    self.raw_bars[e].config(width=max(0, w))
                    self.raw_labels[e].config(text=f"{score:.0%}")

            self.dom_label.config(
                text=f"DOMINANT: {dom.upper()}",
                fg=EMOTION_COLORS.get(dom, TEXT)
            )

            # EduSense bars
            dissat = False
            for e, score in edu.items():
                active = score > 0.30
                w      = int(score * 140)
                self.edu_bars[e].config(width=max(0, w))
                self.edu_labels[e].config(text=f"{score:.0%}")
                self.edu_dots[e].config(
                    fg=EDUSENSE_COLORS[e] if active else '#2a2a3d'
                )
                if e in ['boredom', 'confusion', 'frustration'] and active:
                    dissat = True

            if dissat:
                self.alert.config(text="⚠️  DISSATISFIED", fg='#f43f5e')
            else:
                self.alert.config(text="✅  FOCUSED", fg='#22c55e')

        self.root.after(33, self._update)

    def _quit(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
