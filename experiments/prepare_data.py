"""Reproducible synthetic receipt-line OCR data. No real customer data."""
import json,random,hashlib,urllib.request,os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageEnhance
import numpy as np
ROOT=Path(os.environ.get('VISIONAI_ROOT',Path(__file__).resolve().parents[1]));DATA=ROOT/'data'
FONT_FILES={
 'sans':'https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf',
 'mono':'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmono/NotoSansMono%5Bwdth,wght%5D.ttf',
 'serif':'https://raw.githubusercontent.com/google/fonts/main/ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf'}
# Font paths are pinned by their recorded SHA-256 after download.
ITEMS=['MILK','COFFEE','BREAD','APPLE','WATER','RICE','TEA','JUICE','SUGAR','CHEESE','EGGS','BUTTER','SOAP','TOWEL','PASTA','COOKIE','YOGURT','SALT','PEPPER','CEREAL','SALAD','SOUP','CAKE','BANANA']
def text_for(rng):
 kind=rng.choice(['item','item','total','date','code'])
 if kind=='item':return f'{rng.choice(ITEMS)} {rng.randint(1,9)} {rng.randint(100,9999)/100:.2f}',kind
 if kind=='total':return f'{rng.choice(["TOTAL","SUBTOTAL","CASH","CHANGE","TAX"])} {rng.randint(1,49999)/100:.2f}',kind
 if kind=='date':return f'DATE 2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}',kind
 return f'RECEIPT {rng.randint(100000,999999)}',kind

def main():
 archive=DATA/'receipt_lines.zip'
 if archive.exists() and os.environ.get('REGENERATE')!='1':
  import zipfile
  with zipfile.ZipFile(archive) as z:z.extractall(DATA)
  ledger=json.loads((DATA/'image_sha256.json').read_text())
  assert all(hashlib.sha256((DATA/name).read_bytes()).hexdigest()==value for name,value in ledger.items())
  print('Extracted and verified',len(ledger),'fixed PNG files');return
 DATA.mkdir(parents=True,exist_ok=True);fonts=DATA/'fonts';fonts.mkdir(exist_ok=True)
 for name,url in FONT_FILES.items():
  dest=fonts/(name+'.ttf')
  if not dest.exists():
   urllib.request.urlretrieve(url,dest)
   (fonts/(name+'.source.txt')).write_text(url)
 rng=random.Random(42);seen=set();records=[]
 for split,count in [('train',1000),('val',150),('test',200)]:
  for i in range(count):
   while True:
    text,kind=text_for(rng)
    if text not in seen:seen.add(text);break
   font_name=rng.choice(['sans','mono','serif']);size=rng.randint(24,34);font=ImageFont.truetype(str(fonts/(font_name+'.ttf')),size)
   box=font.getbbox(text);w=box[2]-box[0]+36;h=box[3]-box[1]+28
   clean=Image.new('RGB',(w,h),'white');ImageDraw.Draw(clean).text((18-box[0],14-box[1]),text,font=font,fill=(20,20,20))
   angle=rng.uniform(-4,4);blur=rng.uniform(.4,1.0);contrast=rng.uniform(.45,.75)
   bad=clean.rotate(angle,Image.Resampling.BICUBIC,expand=True,fillcolor='white')
   bad=ImageEnhance.Contrast(bad).enhance(contrast).filter(ImageFilter.GaussianBlur(blur))
   arr=np.asarray(bad).astype(float);noise=np.random.default_rng(10000+len(records)).normal(0,5,arr.shape);bad=Image.fromarray(np.uint8(np.clip(arr+noise,0,255)))
   stem=f'{split}_{i:04d}';paths={}
   for cond,img in [('clean',clean),('degraded',bad)]:
    p=DATA/'images'/split/cond/(stem+'.png');p.parent.mkdir(parents=True,exist_ok=True);img.save(p);paths[cond]=p.relative_to(DATA).as_posix()
   records.append(dict(id=stem,split=split,text=text,kind=kind,font=font_name,width=w,height=h,angle=angle,blur=blur,contrast=contrast,paths=paths))
 (DATA/'manifest.json').write_text(json.dumps(records,indent=2))
 meta={'seed':42,'counts':{s:sum(r['split']==s for r in records) for s in ['train','val','test']},'unique_texts':len(seen),'font_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fonts.glob('*.ttf')},'manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),'dataset_type':'synthetic English receipt lines; same fonts and grammar across splits; exact text disjoint','degradation':'angle uniform[-4,4] deg, Gaussian blur radius[0.4,1.0], contrast[0.45,0.75], Gaussian noise sd5'}
 (DATA/'summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
if __name__=='__main__':main()
