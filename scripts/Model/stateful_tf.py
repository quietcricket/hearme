"""Stateful TF-Restormer: retain Mamba, convolution, STFT and overlap-add state."""
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'models/TF_Restormer'))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from scipy.signal import firwin, lfilter
from mamba_reference import Mamba
from tf_restormer.models.TF_Restormer.modules import network
from tf_restormer.models.TF_Restormer.model import Model
from tf_restormer.utils.util_stft import STFT, iSTFT

SHADER='''
#include <metal_stdlib>
using namespace metal;
kernel void scan(device const float* u, device const float* dt,
 device const float* A, device const float* B, device const float* C,
 device const float* D, device const float* z, device const float* bias,
 device float* state, device float* out, constant uint& channels,
 constant uint& length, uint idx [[thread_position_in_grid]]) {
 uint batch=idx/channels, channel=idx%channels;
 float s[16]; for(uint n=0;n<16;n++) s[n]=state[idx*16+n];
 for(uint t=0;t<length;t++) {
   uint i=idx*length+t; float raw=dt[i]+bias[channel];
   float delta=raw>20.0f ? raw : log(1.0f+exp(raw));
   float v=u[i], sum=0.0f;
   for(uint n=0;n<16;n++) {
     uint j=(batch*16+n)*length+t;
     s[n]=exp(delta*A[channel*16+n])*s[n]+delta*B[j]*v;
     sum+=s[n]*C[j];
   }
   float gate=z[i]; out[i]=(sum+D[channel]*v)*gate/(1.0f+exp(-gate));
 }
 for(uint n=0;n<16;n++) state[idx*16+n]=s[n];
}'''
_shader=None


class StatefulMamba(Mamba):
    def reset(self):
        self.conv_state=None
        self.scan_state=None

    def forward(self,hidden_states,inference_params=None):
        global _shader
        batch,length,_=hidden_states.shape
        if not hasattr(self,'conv_state') or self.conv_state is None:
            self.conv_state=hidden_states.new_zeros(batch,self.d_inner,self.d_conv-1)
            self.scan_state=hidden_states.new_zeros(batch,self.d_inner,self.d_state)
        x,z=self.in_proj(hidden_states).transpose(1,2).chunk(2,dim=1)
        combined=torch.cat([self.conv_state,x],dim=-1)
        self.conv_state=combined[...,-(self.d_conv-1):].clone()
        x=F.silu(F.conv1d(combined,self.conv1d.weight,self.conv1d.bias,groups=self.d_inner))
        dt,B,C=torch.split(self.x_proj(x.transpose(1,2)),[self.dt_rank,self.d_state,self.d_state],dim=-1)
        dt=F.linear(dt,self.dt_proj.weight).transpose(1,2)
        B=B.transpose(1,2);C=C.transpose(1,2)
        A=-torch.exp(self.A_log.float())
        if hidden_states.device.type=='mps':
            if _shader is None: _shader=torch.mps.compile_shader(SHADER)
            args=[v.contiguous() for v in (x,dt,A,B,C,self.D,z,self.dt_proj.bias)]
            output=torch.empty_like(x)
            _shader.scan(*args,self.scan_state,output,self.d_inner,length,threads=batch*self.d_inner)
        else:
            delta=F.softplus(dt+self.dt_proj.bias[None,:,None])
            values=[]
            for t in range(length):
                self.scan_state=torch.exp(delta[:,:,t,None]*A)*self.scan_state+delta[:,:,t,None]*B[:,None,:,t]*x[:,:,t,None]
                y=(self.scan_state*C[:,None,:,t]).sum(-1)+self.D*x[:,:,t]
                values.append(y*F.silu(z[:,:,t]))
            output=torch.stack(values,dim=-1)
        return self.out_proj(output.transpose(1,2))


class BufferedConv(nn.Module):
    """One future frame per symmetric 3x3 projection, without padding boundaries."""
    def __init__(self,conv):
        super().__init__()
        self.conv=conv
        self.history=None

    def reset(self): self.history=None

    def forward(self,x):
        if self.history is None:
            self.history=x.new_zeros(x.shape[0],x.shape[1],x.shape[2],1)
        joined=torch.cat([self.history,x],dim=-1)
        self.history=joined[...,-2:].clone()
        if joined.shape[-1]<3:
            return x.new_empty(x.shape[0],self.conv.out_channels,x.shape[2],0)
        return F.conv2d(joined,self.conv.weight,self.conv.bias,
                        padding=(self.conv.padding[0],0))


def make_model(device='mps'):
    torch.set_num_threads(4)
    weights=ROOT/'models/weights'
    cfg=yaml.safe_load((weights/'config.yaml').read_text())['config']
    assert cfg['model']['online']
    network.Mamba=StatefulMamba
    model=Model(**cfg['model'])
    model.load_state_dict(torch.load(weights/'model.pt',weights_only=True,map_location='cpu')['model_state_dict'],strict=True)
    model.input_embed.embed=BufferedConv(model.input_embed.embed)
    model.estimator.mag.out[0]=BufferedConv(model.estimator.mag.out[0])
    model.estimator.phase.out[0]=BufferedConv(model.estimator.phase.out[0])
    return model.to(device).eval()


def reset_model(model):
    for module in model.modules():
        if isinstance(module,(StatefulMamba,BufferedConv)): module.reset()


class AudioProcessor:
    def __init__(self,device='mps'):
        self.device=device
        self.release_cache=True
        self.model=make_model(device)
        self.stft=STFT(640,320,device='cpu',normalize=True)
        self.istft=iSTFT(1920,960,device='cpu',normalize=True)
        self.filter=firwin(127,1/3,window=('kaiser',8.6)).astype(np.float32)
        self.reset()

    def reset(self):
        reset_model(self.model)
        self.filter_state=np.zeros(len(self.filter)-1,dtype=np.float32)
        self.input=np.empty(0,dtype=np.float32)
        self.tail=torch.zeros(960)
        self.decimation_phase=0

    @torch.inference_mode()
    def process(self,audio):
        filtered,self.filter_state=lfilter(self.filter,[1.],audio,zi=self.filter_state)
        down=filtered[self.decimation_phase::3].astype(np.float32)
        self.decimation_phase=(self.decimation_phase-len(audio))%3
        self.input=np.concatenate([self.input,down])
        frames=max(0,(len(self.input)-640)//320+1)
        if frames<3: return np.empty(0,dtype=np.float32)
        size=(frames-1)*320+640
        spec=self.stft(torch.from_numpy(self.input[:size])[None],cplx=True)
        self.input=self.input[frames*320:].copy()
        features=torch.stack([spec.real,spec.imag],dim=-1).to(self.device)
        output=self.model(features,out_F=961).cpu()
        if output.shape[2]==0: return np.empty(0,dtype=np.float32)
        complex_output=torch.complex(output[...,0],output[...,1])
        wave=self.istft(complex_output,cplx=True).reshape(-1)
        wave[:960]+=self.tail
        self.tail=wave[-960:].clone()
        result=wave[:-960].numpy().copy()
        if self.device=='mps' and self.release_cache: torch.mps.empty_cache()
        return result
