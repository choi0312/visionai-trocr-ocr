"""TrOCR decoder adaptation on synthetic receipt-line images; actual CUDA run."""
import os,json,time,random,csv,platform,subprocess,importlib.metadata as md
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
from transformers.modeling_outputs import BaseModelOutput
ROOT=Path(os.environ.get('VISIONAI_ROOT','/workspace/visionai_week7'));DATA=ROOT/'data';OUT=ROOT/'results';OUT.mkdir(exist_ok=True,parents=True)
MODEL='microsoft/trocr-small-printed'
REVISION=json.loads((ROOT/'model_metadata.json').read_text())['sha']
SEED=42;EPOCHS=12;BATCH=16
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_num_threads(8)
assert torch.cuda.is_available(),'CUDA GPU required'
print('GPU:',torch.cuda.get_device_name(),flush=True)
processor=TrOCRProcessor.from_pretrained(MODEL,revision=REVISION)
model=VisionEncoderDecoderModel.from_pretrained(MODEL,revision=REVISION).cuda()
model.config.decoder_start_token_id=model.decoder.config.decoder_start_token_id
model.config.pad_token_id=processor.tokenizer.pad_token_id
model.config.eos_token_id=processor.tokenizer.sep_token_id
for k in ['decoder_start_token_id','pad_token_id','eos_token_id']:setattr(model.generation_config,k,getattr(model.config,k))
model.generation_config.num_beams=1
model.generation_config.use_cache=True
model.generation_config.max_length=48
for p in model.encoder.parameters():p.requires_grad=False
rows=json.loads((DATA/'manifest.json').read_text());groups={}
for split in ['train','val','test']:
 r=[x for x in rows if x['split']==split]
 if split=='train':groups['train']=[dict(x,condition=('clean' if i%2==0 else 'degraded')) for i,x in enumerate(r)]
 elif split=='val':groups['val']=[dict(x,condition='degraded') for x in r]
 else:
  for cond in ['clean','degraded']:groups['test_'+cond]=[dict(x,condition=cond) for x in r]
def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
start=time.perf_counter();cache={};model.eval()
for name,rs in groups.items():
 features=[];labels=[]
 for i in range(0,len(rs),BATCH):
  batch=rs[i:i+BATCH];imgs=[Image.open(DATA/x['paths'][x['condition']]).convert('RGB') for x in batch]
  pv=processor(images=imgs,return_tensors='pt').pixel_values.cuda()
  with torch.no_grad(),amp():enc=model.encoder(pv).last_hidden_state
  features.append(enc.cpu().to(torch.float16))
  lab=processor.tokenizer([x['text'] for x in batch],padding='max_length',max_length=48,truncation=False,return_tensors='pt').input_ids
  assert lab.shape[1]==48;lab=lab[:,1:];lab[lab==processor.tokenizer.pad_token_id]=-100;labels.append(lab)
 cache[name]=(torch.cat(features),torch.cat(labels))
 print('Encoded',name,len(rs),flush=True)
encoding_s=time.perf_counter()-start

def token_loss(emb,y):
 # TrOCR checkpoint starts with EOS (id 2); tokenizer BOS is not a target.
 # Compute aligned seq2seq CE explicitly: Transformers 4.57 generic causal
 # loss would shift these already-shifted sequence targets a second time.
 inp=torch.full_like(y,processor.tokenizer.pad_token_id)
 inp[:,0]=model.config.decoder_start_token_id
 inp[:,1:]=y[:,:-1].masked_fill(y[:,:-1]==-100,processor.tokenizer.pad_token_id)
 logits=model(encoder_outputs=emb,decoder_input_ids=inp,use_cache=False).logits
 return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)

def lev(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
def norm(s):return ' '.join(s.split())
def metrics(records):
 nc=nw=ce=we=em=0
 for x in records:
  a,b=norm(x['reference']),norm(x['prediction']);ce+=lev(a,b);we+=lev(a.split(),b.split());nc+=len(a);nw+=len(a.split());em+=a==b
 return dict(n=len(records),cer=ce/nc,wer=we/nw,exact_match=em/len(records),char_edits=ce,reference_chars=nc,word_edits=we,reference_words=nw)
@torch.no_grad()
def evaluate(name,save=None):
 model.eval();rs=groups[name];feat,lab=cache[name];loss=0;records=[];t=time.perf_counter()
 for i in range(0,len(rs),BATCH):
  emb=BaseModelOutput(last_hidden_state=feat[i:i+BATCH].cuda().to(torch.bfloat16));y=lab[i:i+BATCH].cuda()
  with amp():
   loss+=token_loss(emb,y).item()*len(y)
   ids=model.generate(encoder_outputs=emb,max_new_tokens=47,num_beams=1,do_sample=False,use_cache=True)
  texts=processor.batch_decode(ids,skip_special_tokens=True)
  records.extend(dict(id=x['id'],condition=x['condition'],reference=x['text'],prediction=pred) for x,pred in zip(rs[i:i+BATCH],texts))
 torch.cuda.synchronize();m=metrics(records);m.update(loss=loss/len(rs),decoder_eval_s=time.perf_counter()-t)
 if save:
  (OUT/(save+'_metrics.json')).write_text(json.dumps(m,indent=2));(OUT/(save+'_records.json')).write_text(json.dumps(records,indent=2))
 return m
baseval=evaluate('val','before_val');print('BASELINE val',json.dumps(baseval),flush=True)
for cond in ['clean','degraded']:
 m=evaluate('test_'+cond,'before_'+cond);print('BASELINE',cond,json.dumps(m),flush=True)
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
best=float('inf');bestepoch=None;history=[];t0=time.perf_counter()
feat,labels=cache['train'];generator=torch.Generator().manual_seed(SEED)
for epoch in range(1,EPOCHS+1):
 model.train();model.encoder.eval();order=torch.randperm(len(feat),generator=generator);total=0
 for j in range(0,len(order),BATCH):
  idx=order[j:j+BATCH];emb=BaseModelOutput(last_hidden_state=feat[idx].cuda().to(torch.bfloat16));y=labels[idx].cuda();opt.zero_grad(set_to_none=True)
  with amp():loss=token_loss(emb,y)
  loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);opt.step();total+=loss.item()*len(idx)
 val=evaluate('val');row=dict(epoch=epoch,train_loss=total/len(feat),val_loss=val['loss'],val_cer=val['cer'],val_wer=val['wer'],val_exact_match=val['exact_match']);history.append(row)
 if val['cer']<best:
  best=val['cer'];bestepoch=epoch;torch.save(model.decoder.state_dict(),OUT/'best_decoder.pt')
 with (OUT/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
 print('EPOCH',json.dumps(row),flush=True)
training_s=time.perf_counter()-t0
model.decoder.load_state_dict(torch.load(OUT/'best_decoder.pt',map_location='cuda',weights_only=True))
for cond in ['clean','degraded']:
 m=evaluate('test_'+cond,'after_'+cond);print('FINAL',cond,json.dumps(m),flush=True)
env=dict(model=MODEL,revision=REVISION,seed=SEED,epochs=EPOCHS,batch_size=BATCH,optimizer='AdamW',learning_rate=2e-5,weight_decay=.01,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),encoder_frozen=True,training_wall_s=training_s,encoding_s=encoding_s,best_epoch=bestepoch,best_val_cer=best,gpu=torch.cuda.get_device_name(),python=platform.python_version(),packages={x:md.version(x) for x in ['torch','transformers','Pillow','numpy','sentencepiece']},loss_implementation='explicit aligned sequence CE; tokenizer BOS stripped; decoder input EOS-start shifted once',evaluation='case-sensitive; whitespace collapse; corpus edit-count CER/WER; per-line exact match; greedy decoding max_new_tokens47; original paired clean/degraded test strings; no detection')
(OUT/'environment.json').write_text(json.dumps(env,indent=2));(OUT/'SUCCESS').write_text('completed');print('SUCCESS',json.dumps(env),flush=True)
