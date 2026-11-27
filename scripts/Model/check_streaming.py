"""Check streamed state against complete-sequence pretrained inference."""
import time
import json
import torch
import numpy as np
import yaml
from mamba_reference import Mamba
from stateful_tf import StatefulMamba,make_model,ROOT,AudioProcessor
from tf_restormer.models.TF_Restormer.modules import network
from tf_restormer.models.TF_Restormer.model import Model

torch.set_num_threads(4)
torch.mps.set_per_process_memory_fraction(.5)
with torch.inference_mode():
    torch.manual_seed(4)
    ref=Mamba(64,d_state=16,d_conv=3,expand=4).eval()
    stream=StatefulMamba(64,d_state=16,d_conv=3,expand=4).eval()
    stream.load_state_dict(ref.state_dict())
    x=torch.randn(2,67,64)
    expected=ref(x)
    stream.to('mps')
    actual=torch.cat([stream(p.to('mps')).cpu() for p in x.split(13,dim=1)],dim=1)
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)
    mamba_error=float((actual-expected).abs().max())
    weights=ROOT/'models/weights'
    cfg=yaml.safe_load((weights/'config.yaml').read_text())['config']
    network.Mamba=Mamba
    ref=Model(**cfg['model']).eval()
    ref.load_state_dict(torch.load(weights/'model.pt',map_location='cpu',weights_only=True)['model_state_dict'])
    ref.to('mps')
    features=torch.randn(1,321,41,2,device='mps')*.05
    expected=ref(features,out_F=961).cpu()[:,:,:-2]
    del ref
    torch.mps.empty_cache()
    stream=make_model()
    actual=torch.cat([stream(part,out_F=961).cpu() for part in features.split(10,dim=2)],dim=2)
    torch.testing.assert_close(actual,expected,atol=5e-5,rtol=5e-4)
    model_error=float((actual-expected).abs().max())
    del stream
    torch.mps.empty_cache()
    processor=AudioProcessor()
    audio=(np.random.default_rng(2).standard_normal(48000*5)*.02).astype(np.float32)
    durations=[];outputs=[]
    for part in np.split(audio,10):
        start=time.perf_counter()
        outputs.append(processor.process(part))
        durations.append(time.perf_counter()-start)
    assert np.isfinite(np.concatenate(outputs)).all()
    report=dict(mamba_max_error=mamba_error,model_max_error=model_error,
                audio_seconds=5,processing_seconds=sum(durations),block_seconds=durations,
                generated_samples=sum(len(x) for x in outputs),sample_rate=48000)
    print(json.dumps(report,indent=2),flush=True)
    report_path=ROOT/'build/tf-live-validation.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
