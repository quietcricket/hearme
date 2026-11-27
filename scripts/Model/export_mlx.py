"""Export only native model assets; shared weights stored once, matrices in FP16."""
import argparse,json,shutil
from pathlib import Path
from scipy.signal import firwin
import numpy as np
from stateful_tf import ROOT,make_model,STFT,iSTFT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'models/MLX')
out=parser.parse_args().output
out.mkdir(parents=True,exist_ok=True)
m=make_model('cpu').eval()
weights={};aliases={};seen={}
for key,tensor in m.state_dict().items():
 identity=(tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()))
 if identity in seen:aliases[key]=seen[identity];continue
 seen[identity]=key
 full=any(s in key for s in ('layernorm','layer_norm','freq_pe','A_log','.dt_proj.bias','.conv1d.')) or key.endswith('.D')
 weights[key]=tensor.detach().numpy().astype(np.float32 if full else np.float16)
import mlx.core as mx
mx.save_safetensors(str(out/'weights.safetensors'),{k:mx.array(v) for k,v in weights.items()})
(out/'weights.npz').unlink(missing_ok=True)
meta={name:{'heads':module.n_heads} for name,module in m.named_modules() if hasattr(module,'n_heads')}
meta['architecture']={'up_min_F':m.up.min_F,'format_version':1};meta['aliases']=aliases
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
shutil.copy2(Path(__file__).with_name('Scan.metal'),out/'Scan.metal')
np.asarray(firwin(127,1/3,window=('kaiser',8.6))[::-1],dtype='<f4').tofile(out/'fir.f32')
for name,transform in [('analysis',STFT(640,320,device='cpu',normalize=True)),
                       ('synthesis',iSTFT(1920,960,device='cpu',normalize=True))]:
 transform.K.detach().cpu().numpy().astype('<f4').tofile(out/(name+'.f32'))
print(json.dumps({'weights':len(weights),'aliases':len(aliases),'bytes':(out/'weights.safetensors').stat().st_size}),flush=True)
