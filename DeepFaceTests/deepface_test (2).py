"""
DeepFace Live Tester — Simple Version
Run: python deepface_test.py
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from deepface import DeepFace
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
        self.root       = root
        self.root.title("DeepFace Live Tester")
        self.root.configure(bg=BG)
        self.root.geometry("860x500")

        self.cap         = cv2.VideoCapture(0)
        self.running     = True
        self.last_result = None
        self.analyzing   = False
        self.photo       = None

        self._build_ui()
        self._update()

        # Start analysis thread
        threading.Thread(target=self._analysis_loop, daemon=True).start()

        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _build_ui(self):
        # Header
        tk.Label(self.root, text="DeepFace Live Tester",
                font=('Courier', 14, 'bold'), bg=BG, fg='#6ee7b7').pack(pady=8)

        self.status = tk.Label(self.root, text="Starting...",
                              font=('Courier', 9), bg=BG, fg=MUTED)
        self.status.pack()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=16, pady=8)

        # Camera
        self.cam = tk.Label(body, bg='#000')
        self.cam.pack(side='left')

        # Right panel
        right = tk.Frame(body, bg=BG, padx=16)
        right.pack(side='left', fill='both', expand=True)

        # Raw emotions
        tk.Label(right, text="RAW DEEPFACE",
                font=('Courier', 8, 'bold'), bg=BG, fg=MUTED).pack(anchor='w')

        self.raw_bars   = {}
        self.raw_labels = {}
        for e in ['angry','disgust','fear','happy','sad','surprise','neutral']:
            row = tk.Frame(right, bg=BG)
            row.pack(fill='x', pady=1)
            tk.Label(row, text=e[:7].upper(), font=('Courier', 7),
                    bg=BG, fg=EMOTION_COLORS[e], width=8, anchor='w').pack(side='left')
            bg_bar = tk.Frame(row, bg='#1e1e2e', height=8, width=180)
            bg_bar.pack(side='left', padx=4)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=EMOTION_COLORS[e], height=8, width=0)
            fill.pack(side='left', fill='y')
            lbl = tk.Label(row, text="0%", font=('Courier', 7),
                          bg=BG, fg=EMOTION_COLORS[e], width=5)
            lbl.pack(side='left')
            self.raw_bars[e]   = fill
            self.raw_labels[e] = lbl

        self.dom_label = tk.Label(right, text="DOMINANT: --",
                                 font=('Courier', 9, 'bold'), bg=BG, fg=TEXT)
        self.dom_label.pack(anchor='w', pady=6)

        # EduSense mapping
        tk.Label(right, text="EDUSENSE MAPPING",
                font=('Courier', 8, 'bold'), bg=BG, fg=MUTED).pack(anchor='w')

        self.edu_bars   = {}
        self.edu_labels = {}
        self.edu_dots   = {}
        for e in ['engagement','boredom','confusion','frustration']:
            row = tk.Frame(right, bg=BG)
            row.pack(fill='x', pady=2)
            dot = tk.Label(row, text="●", font=('Courier', 9),
                          bg=BG, fg='#2a2a3d')
            dot.pack(side='left', padx=(0,3))
            tk.Label(row, text=e[:8].upper(), font=('Courier', 7),
                    bg=BG, fg=EDUSENSE_COLORS[e], width=9, anchor='w').pack(side='left')
            bg_bar = tk.Frame(row, bg='#1e1e2e', height=10, width=140)
            bg_bar.pack(side='left', padx=4)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=EDUSENSE_COLORS[e], height=10, width=0)
            fill.pack(side='left', fill='y')
            lbl = tk.Label(row, text="0%", font=('Courier', 7),
                          bg=BG, fg=EDUSENSE_COLORS[e], width=5)
            lbl.pack(side='left')
            self.edu_bars[e]   = fill
            self.edu_labels[e] = lbl
            self.edu_dots[e]   = dot

        # Alert
        self.alert = tk.Label(right, text="✅ FOCUSED",
                             font=('Courier', 10, 'bold'),
                             bg=BG, fg='#22c55e')
        self.alert.pack(pady=10)

    def _analysis_loop(self):
        """Run DeepFace every 2 seconds on a captured frame"""
        time.sleep(2)  # wait for camera to warm up
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            self.root.after(0, lambda: self.status.config(
                text="● Analyzing...", fg='#f59e0b'))

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(result, list):
                    result = result[0]

                raw = result['emotion']
                dom = max(raw, key=raw.get)

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

                self.last_result = {'raw': raw, 'dominant': dom, 'edu': edu}
                self.root.after(0, lambda: self.status.config(
                    text="● Live", fg='#6ee7b7'))

            except Exception as e:
                self.root.after(0, lambda err=e: self.status.config(
                    text=f"● {str(err)[:40]}", fg='#f43f5e'))

            time.sleep(2)

    def _update(self):
        """Update camera feed and UI at 30fps"""
        if not self.running:
            return

        # Camera frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((400, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.cam.config(image=self.photo)

        # Update bars
        if self.last_result:
            raw = self.last_result['raw']
            edu = self.last_result['edu']
            dom = self.last_result['dominant']

            for e, score in raw.items():
                if e in self.raw_bars:
                    self.raw_bars[e].config(width=max(0, int(score * 1.8)))
                    self.raw_labels[e].config(text=f"{score:.0f}%")

            self.dom_label.config(
                text=f"DOMINANT: {dom.upper()}",
                fg=EMOTION_COLORS.get(dom, TEXT)
            )

            dissat = False
            for e, score in edu.items():
                active = score > 0.3
                self.edu_bars[e].config(width=max(0, int(score * 140)))
                self.edu_labels[e].config(text=f"{score:.0%}")
                self.edu_dots[e].config(
                    fg=EDUSENSE_COLORS[e] if active else '#2a2a3d'
                )
                if e in ['boredom','confusion','frustration'] and active:
                    dissat = True

            if dissat:
                self.alert.config(text="⚠️  DISSATISFIED", fg='#f43f5e')
            else:
                self.alert.config(text="✅  FOCUSED", fg='#22c55e')

        self.root.after(33, self._update)  # ~30fps

    def _quit(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
