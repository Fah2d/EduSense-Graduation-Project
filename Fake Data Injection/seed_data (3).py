"""
EduSense -- Fake Data Injector (crash-safe, handles existing users)
"""

import os, json, random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

sb = create_client(
    os.getenv('SUPABASE_URL', ''),
    os.getenv('SUPABASE_SERVICE_KEY', ''),
)

SEED_FILE = Path('seed_ids.json')

def ts(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def now_minus(days=0, hours=0):
    return datetime.now(timezone.utc) - timedelta(days=days, hours=hours)

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def rng(lo, hi):
    return round(random.uniform(lo, hi), 3)

def load_ids():
    if SEED_FILE.exists():
        return json.loads(SEED_FILE.read_text())
    return {'auth_users': [], 'profiles': [], 'subjects': [],
            'sessions': [], 'struggle_moments': [], 'notebooks': [], 'emotion_history': []}

def save_ids(ids):
    SEED_FILE.write_text(json.dumps(ids, indent=2))

# ── Personas ──────────────────────────────────────────────────────

TEACHER_PERSONAS = [
    {'name': 'Salem Al-Qahtani',  'fid': 'T-00101', 'dept': 'Computer Science', 'email': 'demo_salem_teacher@kku.edu.sa'},
    {'name': 'Mofrej Al-Qahtani', 'fid': 'T-00102', 'dept': 'Computer Science', 'email': 'demo_mofrej_teacher@kku.edu.sa'},
    {'name': 'Hisham Al-Asmari',  'fid': 'T-00103', 'dept': 'Computer Science', 'email': 'demo_hisham_teacher@kku.edu.sa'},
    {'name': 'Saeed Al-Asiri',    'fid': 'T-00104', 'dept': 'Computer Science', 'email': 'demo_saeed_teacher@kku.edu.sa'},
    {'name': 'Ayman Al-Turigi',   'fid': 'T-00105', 'dept': 'Computer Science', 'email': 'demo_ayman_teacher@kku.edu.sa'},
]

STUDENT_PERSONAS = [
    {'name': 'Ahmed Al-Zahrani',   'sid': '441100001', 'dept': 'Computer Science', 'profile': 'strong',     'email': 'demo_ahmed_student@kku.edu.sa'},
    {'name': 'Sara Al-Otaibi',     'sid': '441100002', 'dept': 'Computer Science', 'profile': 'strong',     'email': 'demo_sara_student@kku.edu.sa'},
    {'name': 'Mohammed Al-Ghamdi', 'sid': '441100003', 'dept': 'Computer Science', 'profile': 'average',    'email': 'demo_mohammed_student@kku.edu.sa'},
    {'name': 'Nora Al-Shehri',     'sid': '441100004', 'dept': 'Computer Science', 'profile': 'average',    'email': 'demo_nora_student@kku.edu.sa'},
    {'name': 'Khalid Al-Dossari',  'sid': '441100005', 'dept': 'Computer Science', 'profile': 'struggling', 'email': 'demo_khalid_student@kku.edu.sa'},
    {'name': 'Lama Al-Harthi',     'sid': '441100006', 'dept': 'Computer Science', 'profile': 'struggling', 'email': 'demo_lama_student@kku.edu.sa'},
    {'name': 'Omar Al-Rashidi',    'sid': '441100007', 'dept': 'Computer Science', 'profile': 'average',    'email': 'demo_omar_student@kku.edu.sa'},
    {'name': 'Reem Al-Qahtani',    'sid': '441100008', 'dept': 'Computer Science', 'profile': 'strong',     'email': 'demo_reem_student@kku.edu.sa'},
]

SUBJECTS = [
    {'name': 'Artificial Intelligence & Machine Learning', 'teacher_idx': 0},
    {'name': 'Distributed Systems',                        'teacher_idx': 1},
    {'name': 'Algorithm Design & Analysis',                'teacher_idx': 2},
    {'name': 'Database Systems',                           'teacher_idx': 3},
    {'name': 'Compiler Design',                            'teacher_idx': 4},
]

YOUTUBE_URLS = [
    'https://www.youtube.com/watch?v=aircAruvnKk',
    'https://www.youtube.com/watch?v=IHZwWFHWa-w',
    'https://www.youtube.com/watch?v=Ilg3gGewQ5U',
    'https://www.youtube.com/watch?v=qFJkd2JVpUg',
    'https://www.youtube.com/watch?v=9-Jl0dxWQs8',
    'https://www.youtube.com/watch?v=ysEN5RaKOlA',
]

PERSONA_PARAMS = {
    'strong':     {'eng': (0.65, 0.90), 'bor': (0.05, 0.20), 'con': (0.05, 0.18), 'fru': (0.02, 0.12), 'struggles': (0, 2)},
    'average':    {'eng': (0.40, 0.70), 'bor': (0.15, 0.35), 'con': (0.15, 0.35), 'fru': (0.08, 0.25), 'struggles': (1, 3)},
    'struggling': {'eng': (0.20, 0.50), 'bor': (0.30, 0.55), 'con': (0.25, 0.50), 'fru': (0.20, 0.45), 'struggles': (2, 5)},
}

NOTEBOOK_TOPICS = [
    ('Backpropagation & Gradient Descent',        ['confusion', 'frustration']),
    ('Distributed Consensus Raft Algorithm',      ['confusion']),
    ('Dynamic Programming Optimal Substructure',  ['confusion', 'boredom']),
    ('B-Tree Indexing Query Optimization',        ['confusion', 'frustration']),
    ('LR Parsing Shift-Reduce Conflicts',         ['confusion', 'frustration']),
    ('Attention Mechanisms in Transformers',      ['boredom', 'confusion']),
    ('Lamport Clocks Vector Clocks',              ['confusion']),
    ('NP-Completeness Reduction Proofs',          ['frustration', 'confusion']),
    ('SQL Window Functions',                      ['boredom']),
    ('Regex Finite Automata',                     ['confusion']),
]

# ── Step 1: Users ─────────────────────────────────────────────────

def get_or_create_user(ids, persona, role):
    """Create user, or if already exists fetch their ID."""
    email = persona['email']

    # Try creating
    try:
        res = sb.auth.admin.create_user({
            'email':         email,
            'password':      'Demo@12345',
            'email_confirm': True,
            'user_metadata': {
                'role':       role,
                'full_name':  persona['name'],
                'department': persona['dept'],
                'is_fake':    True,
            }
        })
        user_id = res.user.id
        if user_id not in ids['auth_users']:
            ids['auth_users'].append(user_id)
            save_ids(ids)
        print(f'  OK auth    {role:7s} {persona["name"]}')

    except Exception as e:
        err = str(e)
        if 'already' in err.lower():
            # User exists — find their ID from the user list
            try:
                all_users = sb.auth.admin.list_users()
                user_id = None
                for u in all_users:
                    if u.email == email:
                        user_id = u.id
                        break
                if not user_id:
                    print(f'  ERROR: {email} exists but ID not found')
                    return None
                if user_id not in ids['auth_users']:
                    ids['auth_users'].append(user_id)
                    save_ids(ids)
                print(f'  FOUND  {role:7s} {persona["name"]} (existing)')
            except Exception as e2:
                print(f'  ERROR listing users: {e2}')
                return None
        else:
            print(f'  SKIP   {email}: {e}')
            return None

    # Upsert profile (safe to run even if it already exists)
    profile_data = {
        'id':         user_id,
        'role':       role,
        'full_name':  persona['name'],
        'department': persona['dept'],
        'is_fake':    True,
    }
    if role == 'student':
        profile_data['student_id'] = persona.get('sid', '')
    else:
        profile_data['faculty_id'] = persona.get('fid', '')

    try:
        sb.table('profiles').upsert(profile_data).execute()
        if user_id not in ids['profiles']:
            ids['profiles'].append(user_id)
            save_ids(ids)
        print(f'  OK profile {role:7s} {persona["name"]}')
    except Exception as e:
        print(f'  SKIP profile {email}: {e}')

    return user_id


def create_users(ids):
    print('\n--- Creating users ---')
    teacher_ids = []
    student_ids = []

    all_personas = (
        [(p, 'teacher') for p in TEACHER_PERSONAS] +
        [(p, 'student') for p in STUDENT_PERSONAS]
    )

    for persona, role in all_personas:
        user_id = get_or_create_user(ids, persona, role)
        if not user_id:
            continue
        if role == 'teacher':
            teacher_ids.append(user_id)
        else:
            student_ids.append({'id': user_id, **persona})

    return teacher_ids, student_ids


# ── Step 2: Subjects ──────────────────────────────────────────────

def create_subjects(ids, teacher_ids):
    print('\n--- Creating subjects ---')
    subject_ids = []
    for subj in SUBJECTS:
        t_idx = subj['teacher_idx'] if subj['teacher_idx'] < len(teacher_ids) else 0
        teacher_id = teacher_ids[t_idx]
        try:
            res = sb.table('subjects').insert({
                'name':       subj['name'],
                'teacher_id': teacher_id,
                'is_fake':    True,
            }).execute()
            sid = res.data[0]['id']
            subject_ids.append({'id': sid, 'name': subj['name'], 'teacher_id': teacher_id})
            ids['subjects'].append(sid)
            save_ids(ids)
            print(f'  OK {subj["name"]}')
        except Exception as e:
            print(f'  SKIP subject: {e}')
    return subject_ids


# ── Step 3: Sessions ──────────────────────────────────────────────

def make_emotion_timeline(profile, n_points=20):
    params = PERSONA_PARAMS[profile]
    timeline = []
    eng = rng(*params['eng'])
    for i in range(n_points):
        eng   = clamp(eng + random.gauss(0, 0.04), *params['eng'])
        bor   = clamp(rng(*params['bor']) + random.gauss(0, 0.03), 0, 1)
        con   = clamp(rng(*params['con']) + random.gauss(0, 0.03), 0, 1)
        fru   = clamp(rng(*params['fru']) + random.gauss(0, 0.02), 0, 1)
        total = eng + bor + con + fru
        timeline.append({
            'video_time':   round(i * (3600 / n_points), 1),
            'engagement':   round(eng / total, 3),
            'boredom':      round(bor / total, 3),
            'confusion':    round(con / total, 3),
            'frustration':  round(fru / total, 3),
            'dissatisfied': (bor + con + fru) / total > 0.5,
        })
    return timeline


def create_sessions(ids, student_ids, subject_ids):
    print('\n--- Creating sessions ---')
    for student in student_ids:
        profile    = student['profile']
        params     = PERSONA_PARAMS[profile]
        n_sessions = random.randint(4, 7)

        for _ in range(n_sessions):
            started = now_minus(days=random.randint(0, 28), hours=random.randint(8, 20))
            ended   = started + timedelta(minutes=random.randint(35, 75))
            subj    = random.choice(subject_ids)

            eng_mean = rng(*params['eng'])
            bor_mean = rng(*params['bor'])
            con_mean = rng(*params['con'])
            fru_mean = rng(*params['fru'])
            n_str    = random.randint(*params['struggles'])

            try:
                row = sb.table('sessions').insert({
                    'student_id':      student['id'],
                    'teacher_id':      subj['teacher_id'],
                    'subject_id':      subj['id'],
                    'youtube_url':     random.choice(YOUTUBE_URLS),
                    'status':          'done',
                    'status_msg':      f'Done! {n_str} notebooks ready.',
                    'started_at':      ts(started),
                    'ended_at':        ts(ended),
                    'focus_rate':      round(eng_mean * 100, 1),
                    'avg_engagement':  round(eng_mean, 3),
                    'avg_boredom':     round(bor_mean, 3),
                    'avg_confusion':   round(con_mean, 3),
                    'avg_frustration': round(fru_mean, 3),
                    'is_fake':         True,
                }).execute()
                session_id = row.data[0]['id']
                ids['sessions'].append(session_id)
                save_ids(ids)
            except Exception as e:
                print(f'  SKIP session: {e}')
                continue

            try:
                eh_rows = [{
                    'session_id':   session_id,
                    'video_time':   pt['video_time'],
                    'engagement':   pt['engagement'],
                    'boredom':      pt['boredom'],
                    'confusion':    pt['confusion'],
                    'frustration':  pt['frustration'],
                    'dissatisfied': pt['dissatisfied'],
                    'is_fake':      True,
                } for pt in make_emotion_timeline(profile)]
                res = sb.table('emotion_history').insert(eh_rows).execute()
                ids['emotion_history'].extend([r['id'] for r in res.data])
                save_ids(ids)
            except Exception as e:
                print(f'  SKIP emotion_history: {e}')

            used_times = set()
            for _ in range(n_str):
                vt = random.randint(5, 55) * 60
                while vt in used_times:
                    vt = random.randint(5, 55) * 60
                used_times.add(vt)
                mins, secs      = divmod(vt, 60)
                topic, detected = random.choice(NOTEBOOK_TOPICS)
                det_f = [d for d in detected if d in ['boredom', 'confusion', 'frustration']]

                try:
                    sm = sb.table('struggle_moments').insert({
                        'session_id':     session_id,
                        'student_id':     student['id'],
                        'subject_id':     subj['id'],
                        'video_time':     vt,
                        'video_time_fmt': f'{mins:02d}:{secs:02d}',
                        'engagement':     round(eng_mean * 0.4, 3),
                        'boredom':        round(bor_mean * 1.4, 3),
                        'confusion':      round(con_mean * 1.5, 3),
                        'frustration':    round(fru_mean * 1.3, 3),
                        'detected':       det_f,
                        'transcript':     f'Lecture covering {topic}. Student showed {", ".join(det_f)}.',
                        'is_fake':        True,
                    }).execute()
                    sm_id = sm.data[0]['id']
                    ids['struggle_moments'].append(sm_id)
                    save_ids(ids)
                except Exception as e:
                    print(f'  SKIP struggle: {e}')
                    continue

                fname = f"{session_id[:8]}_{mins:02d}{secs:02d}_{'_'.join(topic.split()[:2])}.ipynb"
                try:
                    nb = sb.table('notebooks').insert({
                        'session_id':     session_id,
                        'student_id':     student['id'],
                        'filename':       fname,
                        'video_time_fmt': f'{mins:02d}:{secs:02d}',
                        'detected':       det_f,
                        'status':         'done',
                        'is_fake':        True,
                    }).execute()
                    ids['notebooks'].append(nb.data[0]['id'])
                    save_ids(ids)
                except Exception as e:
                    print(f'  SKIP notebook: {e}')

        print(f'  OK {student["name"]:28s} [{profile:10s}] {n_sessions} sessions')


# ── Main ──────────────────────────────────────────────────────────

def main():
    print('=' * 52)
    print('  EduSense -- Fake Data Injector')
    print('=' * 52)

    if SEED_FILE.exists():
        print('\nWARNING: seed_ids.json already exists.')
        print('  Run: python delete_seed_data.py --force')
        print('  Then re-run this script.')
        return

    ids = load_ids()

    teacher_ids, student_ids = create_users(ids)
    if not teacher_ids:
        print('\nERROR: No teachers created.')
        return
    if not student_ids:
        print('\nERROR: No students created.')
        return

    subject_ids = create_subjects(ids, teacher_ids)
    if not subject_ids:
        print('\nERROR: No subjects created. Did you run add_is_fake_columns.sql?')
        return

    create_sessions(ids, student_ids, subject_ids)

    print('\n' + '=' * 52)
    print('  DONE!')
    print(f'  Auth users:       {len(ids["auth_users"])}')
    print(f'  Profiles:         {len(ids["profiles"])}')
    print(f'  Subjects:         {len(ids["subjects"])}')
    print(f'  Sessions:         {len(ids["sessions"])}')
    print(f'  Emotion history:  {len(ids["emotion_history"])}')
    print(f'  Struggle moments: {len(ids["struggle_moments"])}')
    print(f'  Notebooks:        {len(ids["notebooks"])}')
    print()
    print('  Credentials (password: Demo@12345)')
    print('  Teacher -> demo_salem_teacher@kku.edu.sa')
    print('  Teacher -> demo_mofrej_teacher@kku.edu.sa')
    print('  Teacher -> demo_hisham_teacher@kku.edu.sa')
    print('  Teacher -> demo_saeed_teacher@kku.edu.sa')
    print('  Teacher -> demo_ayman_teacher@kku.edu.sa')
    print('  Student -> demo_ahmed_student@kku.edu.sa   [strong]')
    print('  Student -> demo_sara_student@kku.edu.sa    [strong]')
    print('  Student -> demo_reem_student@kku.edu.sa    [strong]')
    print('  Student -> demo_mohammed_student@kku.edu.sa [average]')
    print('  Student -> demo_nora_student@kku.edu.sa    [average]')
    print('  Student -> demo_omar_student@kku.edu.sa    [average]')
    print('  Student -> demo_khalid_student@kku.edu.sa  [struggling]')
    print('  Student -> demo_lama_student@kku.edu.sa    [struggling]')
    print('=' * 52)


if __name__ == '__main__':
    random.seed(42)
    main()
