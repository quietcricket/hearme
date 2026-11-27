
uint idx=thread_position_in_grid.x;
if(idx>=F*N)return;
uint b=idx/N,d=idx%N;float h[16];
#pragma unroll
for(uint n=0;n<16;n++)h[n]=state[(b*16+n)*N+d];
for(uint t=0;t<T;t++){
 uint row=b*T+t;float raw=float(dt[row*N+d])+bias[d];
 float delta=raw>20?raw:metal::log(1+metal::exp(raw));float value=u[row*N+d],sum=0;
 #pragma unroll
 for(uint n=0;n<16;n++){
  float B=bc[row*(R+32)+R+n],C=bc[row*(R+32)+R+16+n];
  if(ready[0]>0.5 || t>=2)h[n]=metal::exp(delta*A[d*16+n])*h[n]+delta*B*value;
  sum+=h[n]*C;
 }
 float z=xz[row*N*2+N+d];out[row*N+d]=(sum+D[d]*value)*z/(1+metal::exp(-z));
}
#pragma unroll
for(uint n=0;n<16;n++)newstate[(b*16+n)*N+d]=h[n];
