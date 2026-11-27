"""Official Mamba-1 Python path with an Apple Metal selective-scan backend.

Reference sources and Apache-2.0 license are in the vendored Mamba
directory. No pretrained tensors or architecture are changed.
"""
import ast
import math
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, repeat

REFERENCE=Path(__file__).resolve().parent / 'vendor/mamba'


def extract(filename,name,kind):
    path=REFERENCE/filename
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,kind) and n.name==name)
    return compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec')


namespace=dict(globals())
exec(extract('selective_scan_interface.py','selective_scan_ref',ast.FunctionDef),namespace)
selective_scan_ref=namespace['selective_scan_ref']
_shader=None


def selective_scan_metal(u,delta,A,B,C,D=None,z=None,delta_bias=None,
                         delta_softplus=False,return_last_state=False):
    if u.device.type!='mps':
        return selective_scan_ref(u,delta,A,B,C,D,z,delta_bias,delta_softplus,return_last_state)
    assert u.dtype==torch.float32 and A.shape[1]==16
    assert B.ndim==C.ndim==3 and D is not None and z is not None
    assert delta_bias is not None and delta_softplus and not return_last_state
    global _shader
    if _shader is None:
        _shader=torch.mps.compile_shader('''
        #include <metal_stdlib>
        using namespace metal;
        kernel void scan(device const float* u, device const float* delta,
                         device const float* A, device const float* B,
                         device const float* C, device const float* D,
                         device const float* z, device const float* bias,
                         device float* output, constant uint& channels,
                         constant uint& length, uint idx [[thread_position_in_grid]]) {
            uint batch=idx/channels, channel=idx%channels;
            float state[16];
            for(uint n=0;n<16;n++) state[n]=0.0f;
            for(uint t=0;t<length;t++) {
                uint offset=idx*length+t;
                float raw=delta[offset]+bias[channel];
                float dt=raw>20.0f ? raw : log(1.0f+exp(raw));
                float input=u[offset], sum=0.0f;
                for(uint n=0;n<16;n++) {
                    uint shared=(batch*16+n)*length+t;
                    state[n]=exp(dt*A[channel*16+n])*state[n]+dt*B[shared]*input;
                    sum+=state[n]*C[shared];
                }
                float gate=z[offset];
                output[offset]=(sum+D[channel]*input)*(gate/(1.0f+exp(-gate)));
            }
        }''')
    inputs=[v.contiguous() for v in (u,delta,A,B,C,D,z,delta_bias)]
    out=torch.empty(u.shape,device=u.device,dtype=u.dtype)
    _shader.scan(*inputs,out,u.shape[1],u.shape[2],threads=u.shape[0]*u.shape[1])
    return out


namespace=dict(globals(),selective_scan_fn=selective_scan_metal,
               causal_conv1d_fn=None,causal_conv1d_update=None,
               selective_state_update=None,mamba_inner_fn=None)
exec(extract('mamba_simple.py','Mamba',ast.ClassDef),namespace)
Mamba=namespace['Mamba']


def verify():
    """Check the actual trained scan dimensions and complete Mamba layer."""
    torch.manual_seed(42)
    u=torch.randn(2,32,65)*.1
    delta=torch.randn_like(u)
    A=-torch.exp(torch.randn(32,16))
    B=torch.randn(2,16,65);C=torch.randn_like(B)
    D=torch.randn(32);z=torch.randn_like(u);bias=torch.randn(32)
    with torch.inference_mode():
        expected=selective_scan_ref(u,delta,A,B,C,D,z,bias,True)
        actual=selective_scan_metal(*(v.to('mps') for v in (u,delta,A,B,C,D,z,bias)),delta_softplus=True).cpu()
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        layer=Mamba(d_model=64,d_state=16,d_conv=3,expand=4).eval()
        x=torch.randn(2,65,64)
        reference=layer(x)
        layer.to('mps')
        prediction=layer(x.to('mps')).cpu()
        torch.testing.assert_close(prediction,reference,atol=2e-5,rtol=2e-4)
    return dict(scan_max_error=float((actual-expected).abs().max()),
                mamba_layer_max_error=float((prediction-reference).abs().max()))


if __name__=='__main__':
    print(verify(),flush=True)
