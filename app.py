"""
EduSense Web App — Flask Backend
=================================
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, json, time, threading, base64
import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv
import sys
print("Python:", sys.version)
print("Working dir:", os.getcwd())

load_dotenv()

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static',
            static_url_path='')

CORS(app)

BASE_DIR   = Path(__file__).parent
KB_DIR     = BASE_DIR / 'Rag_system' / 'edusense_kb'
OUTPUT_DIR = BASE_DIR / 'generated_notebooks'
UPLOAD_DIR = BASE_DIR / 'Rag_system' / 'uploads'
AUDIO_DIR  = BASE_DIR / 'Rag_system' / 'audio'

for d in [KB_DIR, OUTPUT_DIR, UPLOAD_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

session = {
    'active': False, 'youtube_url': None, 'audio_path': None,
    'transcript': None, 'segments': [], 'struggle_moments': [],
    'notebooks': [], 'emotion_history': [], 'trigger_count': 0,
    'last_trigger': 0, 'session_start': 0,
    'status': 'idle', 'status_msg': '', 'rag': None,
}

from collections import deque
emotion_buffer = deque(maxlen=5)

def get_rag():
    if session['rag'] is None:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from Rag_system.edusense_rag import EduSenseRAG
        session['rag'] = EduSenseRAG(
            anthropic_api_key=API_KEY,
            kb_dir=str(KB_DIR),
            output_dir=str(OUTPUT_DIR),
            whisper_model='base'
        )
    return session['rag']

def analyze_frame(frame_bgr):
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(frame_bgr, actions=['emotion'],
                                  enforce_detection=False, silent=True)
        if isinstance(result, list): result = result[0]
        raw = result['emotion']
        happy=raw.get('happy',0); neutral=raw.get('neutral',0)
        fear=raw.get('fear',0); surprise=raw.get('surprise',0)
        angry=raw.get('angry',0); disgust=raw.get('disgust',0); sad=raw.get('sad',0)
        mapped = {
            'engagement':  (happy*0.7+(100-neutral)*0.3)/100,
            'boredom':     (neutral*0.6+sad*0.4)/100,
            'confusion':   (fear*0.5+surprise*0.5)/100,
            'frustration': (angry*0.6+disgust*0.4)/100,
        }
        emotion_buffer.append(mapped)
        smoothed = {}
        for e in mapped:
            avg = np.mean([h[e] for h in emotion_buffer])
            smoothed[e] = {'confidence': float(avg), 'positive': avg > 0.35}
        return smoothed
    except:
        return None

def is_dissatisfied(emotions):
    return any(emotions[e]['positive'] for e in ['boredom','confusion','frustration'])

def get_transcript_at(timestamp, window=60):
    if not session['segments']:
        return "Student appeared disengaged during the lecture."
    relevant = [s['text'] for s in session['segments']
                if timestamp-window <= s['start'] <= timestamp]
    return ' '.join(relevant).strip() or session['transcript'][-500:] or "Lecture content."

@app.route('/')
def index():
    p = BASE_DIR / 'templates' / 'index.html'
    if p.exists():
        return p.read_text(encoding='utf-8')
    return "<h1>EduSense</h1><p>templates/index.html not found</p>"

@app.route('/api/status')
def status():
    elapsed = time.time()-session['session_start'] if session['session_start'] else 0
    m,s = divmod(int(elapsed),60)
    eng = 100.0
    if session['emotion_history']:
        focused = sum(1 for h in session['emotion_history'] if not is_dissatisfied(h['emotions']))
        eng = focused/len(session['emotion_history'])*100
    return jsonify({'status':session['status'],'status_msg':session['status_msg'],
                    'active':session['active'],'elapsed':f'{m:02d}:{s:02d}',
                    'engagement_rate':round(eng,1),'struggle_count':len(session['struggle_moments']),
                    'notebook_count':len(session['notebooks']),'trigger_count':session['trigger_count']})

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    f = request.files['file']
    path = UPLOAD_DIR/f.filename
    f.save(path)
    try:
        rag = get_rag()
        count = rag.upload_textbook(str(path), f.filename.replace('.pdf',''))
        return jsonify({'success':True,'chunks':count,'filename':f.filename})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/kb-stats')
def kb_stats():
    try:
        rag = get_rag()
        return jsonify({'chunks':rag.kb.collection.count()})
    except Exception as e:
        return jsonify({'chunks':0,'error':str(e)})

@app.route('/api/start-session', methods=['POST'])
def start_session():
    data = request.json
    url  = data.get('youtube_url','').strip()
    if not url: return jsonify({'error':'YouTube URL required'}),400
    if session['active']: return jsonify({'error':'Session already active'}),400
    session.update({'youtube_url':url,'active':True,'status':'setup',
                    'status_msg':'Downloading lecture audio...','session_start':time.time(),
                    'struggle_moments':[],'notebooks':[],'emotion_history':[],
                    'trigger_count':0,'last_trigger':0,'transcript':None,'segments':[]})
    def setup():
        try:
            import yt_dlp
            ydl_opts = {'format':'bestaudio/best','outtmpl':str(AUDIO_DIR/'lecture.%(ext)s'),
                        'postprocessors':[{'key':'FFmpegExtractAudio','preferredcodec':'mp3'}],'quiet':True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            session['audio_path'] = str(AUDIO_DIR/'lecture.mp3')
            session['status_msg'] = 'Transcribing with Whisper...'
            import whisper
            model = whisper.load_model('base')
            result = model.transcribe(session['audio_path'],language='en',
                                      initial_prompt='Computer science lecture.',verbose=False)
            session['transcript'] = result['text']
            session['segments']   = result.get('segments',[])
            session['status']     = 'monitoring'
            session['status_msg'] = 'Monitoring student...'
            print(f"✅ Ready. Transcript: {len(session['transcript'])} chars")
        except Exception as e:
            session.update({'status':'idle','status_msg':f'Error: {e}','active':False})
            print(f"❌ {e}")
    threading.Thread(target=setup, daemon=True).start()
    return jsonify({'success':True})

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame_route():
    if not session['active'] or session['status']!='monitoring':
        return jsonify({'status':session['status']})
    try:
        img_data  = request.json.get('image','')
        img_bytes = base64.b64decode(img_data.split(',')[1])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        emotions  = analyze_frame(frame)
        if emotions is None: return jsonify({'emotions':None,'triggered':False})
        now = time.time()
        session['emotion_history'].append({'time':now,'emotions':emotions})
        triggered = False
        if is_dissatisfied(emotions): session['trigger_count']+=1
        else: session['trigger_count']=max(0,session['trigger_count']-1)
        if session['trigger_count']>=3 and now-session['last_trigger']>180:
            session['last_trigger']=now; session['trigger_count']=0; triggered=True
            elapsed = now-session['session_start']
            session['struggle_moments'].append({
                'timestamp':elapsed,'timestamp_fmt':f"{int(elapsed//60):02d}:{int(elapsed%60):02d}",
                'emotions':{e:emotions[e]['confidence'] for e in emotions},
                'detected':[e for e in emotions if emotions[e]['positive']],
                'transcript':get_transcript_at(elapsed),
            })
            print(f"⚠️ Struggle at {elapsed:.0f}s")
        return jsonify({'emotions':{e:{'confidence':emotions[e]['confidence'],'positive':emotions[e]['positive']} for e in emotions},
                        'triggered':triggered,'struggle_count':len(session['struggle_moments'])})
    except Exception as e:
        return jsonify({'error':str(e),'triggered':False})

@app.route('/api/end-session', methods=['POST'])
def end_session():
    if not session['active']: return jsonify({'error':'No active session'}),400
    session['active']=False; session['status']='processing'
    session['status_msg']=f"Generating {len(session['struggle_moments'])} notebooks..."
    def generate():
        try:
            rag = get_rag()
            for i,moment in enumerate(session['struggle_moments']):
                session['status_msg']=f"Notebook {i+1}/{len(session['struggle_moments'])}..."
                emotion_state={e:{'positive':moment['emotions'][e]>0.35,'confidence':moment['emotions'][e]} for e in moment['emotions']}
                path = rag.process_with_text(transcript=moment['transcript'],emotion_state=emotion_state)
                session['notebooks'].append({'path':path,'filename':Path(path).name,
                                             'timestamp':moment['timestamp_fmt'],'detected':moment['detected'],'index':i+1})
            session['status']='done'; session['status_msg']=f"Done! {len(session['notebooks'])} notebooks ready."
        except Exception as e:
            session['status']='done'; session['status_msg']=f'Error: {e}'
    threading.Thread(target=generate, daemon=True).start()
    return jsonify({'success':True,'moments':len(session['struggle_moments'])})

@app.route('/api/notebooks')
def get_notebooks():
    return jsonify({'notebooks':session['notebooks']})

@app.route('/api/download/<filename>')
def download_notebook(filename):
    path = OUTPUT_DIR/filename
    if not path.exists(): return jsonify({'error':'Not found'}),404
    return send_file(str(path), as_attachment=True)

@app.route('/api/struggle-moments')
def get_struggle_moments():
    return jsonify({'moments':session['struggle_moments']})

@app.route('/api/reset', methods=['POST'])
def reset():
    session.update({'active':False,'youtube_url':None,'audio_path':None,'transcript':None,
                    'segments':[],'struggle_moments':[],'notebooks':[],'emotion_history':[],
                    'trigger_count':0,'last_trigger':0,'session_start':0,'status':'idle','status_msg':''})
    emotion_buffer.clear()
    return jsonify({'success':True})

if __name__=='__main__':
    print("\n🚀 EduSense Web App — http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
