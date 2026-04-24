#!/usr/bin/env python3
"""Poll a RunPod pod, output status JSON; exit 0 running, 1 exited, 2 error."""
import os, sys, json, urllib.request
for line in open('.env.local'):
    line=line.strip()
    if '=' in line and not line.startswith('#'):
        k,_,v=line.partition('='); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
pod_id = sys.argv[1]
req = urllib.request.Request(
    f'https://rest.runpod.io/v1/pods/{pod_id}',
    headers={'Authorization': f'Bearer {os.environ["RUNPOD_API_KEY"]}'})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read())
status = data.get('desiredStatus')
print(json.dumps({'pod_id': pod_id, 'status': status, 'last_change': data.get('lastStatusChange')}))
sys.exit(0 if status == 'RUNNING' else 1)
