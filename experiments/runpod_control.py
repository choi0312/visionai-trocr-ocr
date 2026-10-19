"""Local Runpod REST client; credentials never enter logs or GPU environment."""
import json, os, time, urllib.request, urllib.error, argparse
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

def api(method, path, payload=None):
    key = os.environ.get('RUNPOD_API_KEY')
    if not key:raise RuntimeError('Set RUNPOD_API_KEY in the local environment.')
    req = urllib.request.Request('https://rest.runpod.io/v1/'+path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method, headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json', 'User-Agent':'VisionAI-Practice/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Runpod HTTP {exc.code}: '+exc.read().decode()[:500].replace(key,'[REDACTED]')) from None

def safe_summary(p):
    return {k:p.get(k) for k in ('id','name','desiredStatus','costPerHr','publicIp','portMappings','gpu','machineId')}

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['create','status','delete','watchdog']);args=parser.parse_args()
    state_path=BASE/'.private/experiment_pod.json'
    if args.action=='create':
        if state_path.exists():raise RuntimeError('Existing experiment state; inspect before creating another Pod.')
        public_key=(BASE/'.private/visionai_runpod.pub').read_text().strip()
        startup='mkdir -p /root/.ssh /run/sshd; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; /usr/sbin/sshd; sleep 5400'
        p=api('POST','pods',{'name':'visionai-trocr-week7-20261019','cloudType':'SECURE','computeType':'GPU','gpuCount':1,
            'gpuTypeIds':['NVIDIA A40'],'gpuTypePriority':'availability',
            'imageName':'runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404','containerDiskInGb':30,'volumeInGb':0,
            'ports':['22/tcp'],'supportPublicIp':True,'env':{'PUBLIC_KEY':public_key},'dockerStartCmd':['bash','-lc',startup]})
        state={'pod':p,'created_at':time.time(),'max_dollars':5,'deadline':time.time()+5400}
        state_path.write_text(json.dumps(state));state_path.chmod(0o600)
        print(json.dumps(safe_summary(p)))
        if float(p.get('costPerHr',0))>1.5:
            api('DELETE','pods/'+p['id']);raise RuntimeError('Hourly rate above experiment ceiling; new Pod deleted.')
    else:
        state=json.loads(state_path.read_text());pod_id=state['pod']['id']
        if args.action=='status':
            p=api('GET','pods/'+pod_id);state['pod']=p;state_path.write_text(json.dumps(state));print(json.dumps(safe_summary(p)))
        elif args.action=='delete':
            api('DELETE','pods/'+pod_id);state['deleted_at']=time.time();state_path.write_text(json.dumps(state));print('Experiment Pod deleted.')
        elif args.action=='watchdog':
            while time.time()<state['deadline']:
                time.sleep(min(30,state['deadline']-time.time()))
                if json.loads(state_path.read_text()).get('deleted_at'):raise SystemExit(0)
            api('DELETE','pods/'+pod_id);print('Budget watchdog deleted experiment Pod.')
