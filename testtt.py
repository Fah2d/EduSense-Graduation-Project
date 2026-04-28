import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')

sb = create_client(url, key)

# Try to read the profiles table (will be empty but should not error)
res = sb.table('profiles').select('*').limit(1).execute()
print('✅ Connected to Supabase!')
print('✅ profiles table reachable')
print(f'   rows returned: {len(res.data)}')