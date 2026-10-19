"""Plots are computed only from the saved Runpod metrics, manifest and history."""
from pathlib import Path
import json,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'figures';A.mkdir(exist_ok=True);R=ROOT/'results';D=ROOT/'data'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
C=['#267c9d','#bd6d36','#408655']
rows=json.loads((D/'manifest.json').read_text());train=[x for x in rows if x['split']=='train']
fig,ax=plt.subplots(1,3,figsize=(10.7,2.25),layout='constrained')
n=[sum(x['split']==s for x in rows) for s in ['train','val','test']]
ax[0].bar(['train','val','test'],n,color=C);ax[0].set_title('Unique text lines by split')
for i,v in enumerate(n):ax[0].text(i,v+12,str(v),ha='center')
ax[0].set_ylim(0,1150);ax[1].hist([len(x['text']) for x in train],bins=12,color=C[0],rwidth=.9);ax[1].set_title('Training label length');ax[1].set_xlabel('Characters including spaces');ax[2].hist([x['width']/x['height'] for x in train],bins=12,color=C[1],rwidth=.9);ax[2].set_title('Original image aspect ratio');ax[2].set_xlabel('Width / height')
for a in ax:a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
fig.savefig(A/'eda.png',dpi=240);plt.close(fig)
(A/'eda_stats.json').write_text(json.dumps(dict(min_chars=min(len(x['text']) for x in train),max_chars=max(len(x['text']) for x in train),median_chars=float(np.median([len(x['text']) for x in train])),median_aspect_ratio=float(np.median([x['width']/x['height'] for x in train]))),indent=2))
if not (R/'environment.json').exists():raise SystemExit(0)
h=[{k:float(v) for k,v in x.items()} for x in csv.DictReader((R/'history.csv').open())];env=json.loads((R/'environment.json').read_text());epochs=[x['epoch'] for x in h]
fig,ax=plt.subplots(1,3,figsize=(10.7,2.5),layout='constrained')
for a in ax:a.grid(alpha=.18);a.set_xlabel('Epoch');a.axvline(env['best_epoch'],ls=':',c='#888888')
ax[0].plot(epochs,[x['train_loss'] for x in h],label='Train (mixed)',c=C[0]);ax[0].plot(epochs,[x['val_loss'] for x in h],label='Val (degraded)',c=C[1]);ax[0].set_title('Token cross-entropy');ax[0].legend(fontsize=8)
ax[1].plot(epochs,[x['val_cer']*100 for x in h],c=C[0],label='CER');ax[1].plot(epochs,[x['val_wer']*100 for x in h],c=C[1],label='WER');ax[1].set_title('Validation error (%)');ax[1].legend(fontsize=8)
ax[2].plot(epochs,[x['val_exact_match']*100 for x in h],c=C[2]);ax[2].set_title('Validation exact match (%)');ax[2].set_ylim(0,103)
fig.savefig(A/'curves.png',dpi=240);plt.close(fig)
tags=['before_clean','after_clean','before_degraded','after_degraded'];ms={s:json.loads((R/(s+'_metrics.json')).read_text()) for s in tags}
fig,ax=plt.subplots(1,2,figsize=(10.7,2.35),layout='constrained');x=np.arange(2);w=.34
for a in ax:a.set_xticks(x,['Clean','Degraded']);a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
for a,k,title in [(ax[0],'cer','Test CER (%) - lower is better'),(ax[1],'exact_match','Test exact match (%) - higher is better')]:
 for off,phase,color in [(-w/2,'before',C[0]),(w/2,'after',C[2])]:
  vals=[ms[phase+'_'+cond][k]*100 for cond in ['clean','degraded']];a.bar(x+off,vals,w,label=phase.title(),color=color)
  for xi,v in zip(x+off,vals):a.text(xi,v+(.4 if k=='cer' else 1),f'{v:.2f}',ha='center',fontsize=9)
 a.set_title(title);a.legend(fontsize=8);a.set_ylim(0,max(3,max(ms[t][k]*100 for t in tags)*1.22) if k=='cer' else 115)
fig.savefig(A/'quality_comparison.png',dpi=240);plt.close(fig)

def lev(a,b):
 p=list(range(len(b)+1))
 for i,c in enumerate(a,1):
  q=[i]
  for j,d in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+(c!=d)))
  p=q
 return p[-1]
bef={x['id']:x for x in json.loads((R/'before_degraded_records.json').read_text())};aft=json.loads((R/'after_degraded_records.json').read_text());byid={x['id']:x for x in rows}
gain=lambda x:lev(x['reference'],bef[x['id']]['prediction'])-lev(x['reference'],x['prediction'])
first=max(aft,key=gain);worst=max((x for x in aft if x['id']!=first['id']),key=lambda x:lev(x['reference'],x['prediction']))
selected=[first,worst];(A/'selected_examples.json').write_text(json.dumps(selected,indent=2))
fig,ax=plt.subplots(2,2,figsize=(10.7,3.05),gridspec_kw={'width_ratios':[1,1.15]},layout='constrained')
for i,x in enumerate(selected):
 rec=byid[x['id']];ax[i,0].imshow(Image.open(D/rec['paths']['degraded']));ax[i,0].axis('off');ax[i,0].set_title(x['id']+' | degraded input',fontsize=9)
 a=ax[i,1];a.axis('off');a.text(0,.82,'GT:     '+x['reference'],family='DejaVu Sans Mono',fontsize=10);a.text(0,.50,'Before: '+bef[x['id']]['prediction'],family='DejaVu Sans Mono',fontsize=10,color=C[1]);a.text(0,.18,'After:  '+x['prediction'],family='DejaVu Sans Mono',fontsize=10,color=C[2])
fig.savefig(A/'predictions.png',dpi=240);plt.close(fig)
# Actual source excerpts, not a fabricated screenshot of a terminal or editor.
im=Image.new('RGB',(1800,450),'white');d=ImageDraw.Draw(im);f=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSansMono.ttf'),27);ttl=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),30)
d.rectangle((0,0,1799,449),outline='#d9d9d9',width=2);d.line((1115,60,1115,430),fill='#d9d9d9',width=2)
d.text((20,15),'Core code excerpts | gpu_experiment.py',font=ttl,fill='#222222');d.text((1135,15),'Recorded result excerpts',font=ttl,fill='#222222')
code=['for p in model.encoder.parameters():','    p.requires_grad=False','logits=model(encoder_outputs=emb,','    decoder_input_ids=inp,','    use_cache=False).logits','loss=F.cross_entropy(logits.reshape(-1,','    logits.shape[-1]), y.reshape(-1),','    ignore_index=-100)','loss.backward()','opt.step()']
for i,line in enumerate(code):d.text((20,78+i*35),line,font=f,fill='#244b70')
b=ms['before_degraded'];a=ms['after_degraded'];out=[env['gpu'],f'Epochs: 12 | best: {env["best_epoch"]}',f'Train wall: {env["training_wall_s"]:.1f}s','Paired test: 200 text lines',f'CER: {b["cer"]:.4f} -> {a["cer"]:.4f}',f'WER: {b["wer"]:.4f} -> {a["wer"]:.4f}',f'EM : {b["exact_match"]:.4f} -> {a["exact_match"]:.4f}','Degraded | case-sensitive']
for i,line in enumerate(out):d.text((1135,78+i*35),line,font=f,fill='#222222')
im.save(A/'code_excerpt.png')
