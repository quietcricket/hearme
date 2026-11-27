#include "AudioRing.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>
struct HMRing {
    float *samples;
    double *times;
    double sample_rate;
    _Atomic double latency;
    uint32_t capacity;
    _Atomic uint64_t read, write, dropped, missing;
    _Atomic float gain;
    _Atomic float peak;
    _Atomic uint64_t ready_at, delay_ticks;
    _Atomic bool started;
    double ticks_per_second;
};
HMRing *hm_ring_create(uint32_t capacity) {
    if (!capacity) return NULL;
    HMRing *r=calloc(1,sizeof(HMRing));
    if (!r) return NULL;
    r->samples=calloc(capacity,sizeof(float)); r->times=calloc(capacity,sizeof(double)); r->capacity=capacity; r->sample_rate=48000;
    if (!r->samples || !r->times) {free(r->samples);free(r->times);free(r); return NULL;}
    mach_timebase_info_data_t timebase;mach_timebase_info(&timebase);
    r->ticks_per_second=1e9*(double)timebase.denom/timebase.numer;
    atomic_init(&r->gain,1.0f); return r;
}
void hm_ring_destroy(HMRing *r) {if(r){free(r->samples);free(r->times);free(r);}}
uint32_t hm_ring_count(HMRing *r) {
    return (uint32_t)(atomic_load_explicit(&r->write,memory_order_acquire)-atomic_load_explicit(&r->read,memory_order_acquire));
}
uint64_t hm_ring_total_written(HMRing *r){return atomic_load(&r->write);}
void hm_ring_discard(HMRing *r) {
    uint64_t w=atomic_load_explicit(&r->write,memory_order_acquire);
    atomic_store_explicit(&r->read,w,memory_order_release);
    atomic_store(&r->dropped,0);
}
double hm_ring_head_time(HMRing *r) {
    uint64_t q=atomic_load_explicit(&r->read,memory_order_relaxed);
    return hm_ring_count(r)>0?r->times[q%r->capacity]:0;
}
void hm_ring_sample_rate(HMRing *r,double rate){r->sample_rate=rate;}
double hm_ring_latency(HMRing *r){return atomic_load(&r->latency);}
uint32_t hm_ring_write(HMRing *r,const float *x,uint32_t n) {
    return hm_ring_write_timed(r,x,n,0);
}
uint32_t hm_ring_write_timed(HMRing *r,const float *x,uint32_t n,double input_time) {
    uint64_t w=atomic_load_explicit(&r->write,memory_order_relaxed);
    uint64_t q=atomic_load_explicit(&r->read,memory_order_acquire);
    uint32_t available=r->capacity-(uint32_t)(w-q), count=n<available?n:available;
    for(uint32_t i=0;i<count;i++) {
        r->samples[(w+i)%r->capacity]=x[i];
        r->times[(w+i)%r->capacity]=input_time>0?input_time+(double)i/r->sample_rate:0;
    }
    if(!atomic_load(&r->ready_at) && count>0)atomic_store(&r->ready_at,mach_absolute_time()+atomic_load(&r->delay_ticks));
    atomic_store_explicit(&r->write,w+count,memory_order_release);
    atomic_fetch_add_explicit(&r->dropped,n-count,memory_order_relaxed); return count;
}
uint32_t hm_ring_read(HMRing *r,float *x,uint32_t n) {
    uint64_t q=atomic_load_explicit(&r->read,memory_order_relaxed);
    uint64_t w=atomic_load_explicit(&r->write,memory_order_acquire);
    uint32_t count=n<(w-q)?n:(uint32_t)(w-q);
    for(uint32_t i=0;i<count;i++)x[i]=r->samples[(q+i)%r->capacity];
    atomic_store_explicit(&r->read,q+count,memory_order_release);return count;
}
static uint32_t capture_at(HMRing *r,const AudioBufferList *audio,uint64_t host_time) {
    if(!audio->mNumberBuffers)return 0;
    uint32_t channels=0, frames=UINT32_MAX;
    for(uint32_t b=0;b<audio->mNumberBuffers;b++) {
        const AudioBuffer *v=&audio->mBuffers[b];
        if(!v->mData||!v->mNumberChannels)continue;
        channels+=v->mNumberChannels;
        uint32_t f=v->mDataByteSize/(sizeof(float)*v->mNumberChannels);
        if(f<frames)frames=f;
    }
    if(!channels)return 0;
    uint64_t w=atomic_load_explicit(&r->write,memory_order_relaxed);
    uint64_t q=atomic_load_explicit(&r->read,memory_order_acquire);
    uint32_t space=r->capacity-(uint32_t)(w-q), count=frames<space?frames:space;
    float peak=0;
    for(uint32_t f=0;f<count;f++) {
        float sum=0;
        for(uint32_t b=0;b<audio->mNumberBuffers;b++) {
            const AudioBuffer *v=&audio->mBuffers[b];
            if(!v->mData)continue;
            const float *p=v->mData;
            for(uint32_t c=0;c<v->mNumberChannels;c++)sum+=p[f*v->mNumberChannels+c];
        }
        r->samples[(w+f)%r->capacity]=sum/channels;
        r->times[(w+f)%r->capacity]=(double)host_time/r->ticks_per_second+(double)f/r->sample_rate;
        peak=fmaxf(peak,fabsf(sum/channels));
    }
    atomic_store_explicit(&r->write,w+count,memory_order_release);
    if(peak>atomic_load(&r->peak))atomic_store(&r->peak,peak);
    atomic_fetch_add_explicit(&r->dropped,frames-count,memory_order_relaxed);return count;
}
uint32_t hm_ring_capture(HMRing *r,const AudioBufferList *audio) {
    return capture_at(r,audio,mach_absolute_time());
}
void hm_ring_render(HMRing *r,AudioBufferList *audio,uint32_t frames) {
    hm_ring_render_timed(r,audio,frames,mach_absolute_time());
}
void hm_ring_render_timed(HMRing *r,AudioBufferList *audio,uint32_t frames,uint64_t host_time) {
    if(!atomic_load(&r->started)) {
        uint64_t ready=atomic_load(&r->ready_at);
        if(!ready || mach_absolute_time()<ready) {
            for(uint32_t b=0;b<audio->mNumberBuffers;b++)
                if(audio->mBuffers[b].mData)memset(audio->mBuffers[b].mData,0,audio->mBuffers[b].mDataByteSize);
            return;
        }
        atomic_store(&r->started,true);
    }
    uint64_t q=atomic_load_explicit(&r->read,memory_order_relaxed);
    uint64_t w=atomic_load_explicit(&r->write,memory_order_acquire);
    uint32_t count=frames<(w-q)?frames:(uint32_t)(w-q);
    if(count>0 && r->times[q%r->capacity]>0) {
        double delay=(double)host_time/r->ticks_per_second-r->times[q%r->capacity];
        if(delay>=0)atomic_store(&r->latency,delay);
    }
    float gain=atomic_load_explicit(&r->gain,memory_order_relaxed);
    float peak=0;
    for(uint32_t f=0;f<frames;f++) {
        float v=f<count?r->samples[(q+f)%r->capacity]*gain:0;
        if(!isfinite(v))v=0;
        v=fmaxf(-0.98f,fminf(0.98f,v));
        peak=fmaxf(peak,fabsf(v));
        for(uint32_t b=0;b<audio->mNumberBuffers;b++) {
            AudioBuffer *out=&audio->mBuffers[b];
            if(!out->mData)continue;
            float *p=out->mData;
            for(uint32_t c=0;c<out->mNumberChannels;c++)p[f*out->mNumberChannels+c]=v;
        }
    }
    atomic_store_explicit(&r->read,q+count,memory_order_release);
    if(peak>atomic_load(&r->peak))atomic_store(&r->peak,peak);
    atomic_fetch_add_explicit(&r->missing,frames-count,memory_order_relaxed);
}
void hm_ring_gain(HMRing *r,float gain){atomic_store(&r->gain,gain);}
void hm_ring_start_delay(HMRing *r,double seconds){atomic_store(&r->delay_ticks,(uint64_t)(fmax(0,seconds)*r->ticks_per_second));}
float hm_ring_take_peak(HMRing *r){return atomic_exchange(&r->peak,0);}
uint64_t hm_ring_dropped(HMRing *r){return atomic_load(&r->dropped);}
uint64_t hm_ring_missing(HMRing *r){return atomic_load(&r->missing);}
OSStatus hm_audio_io_proc(AudioObjectID device,const AudioTimeStamp *now,
 const AudioBufferList *input,const AudioTimeStamp *inputTime,
 AudioBufferList *output,const AudioTimeStamp *outputTime,void *context) {
    capture_at((HMRing *)context,input,(inputTime->mFlags & kAudioTimeStampHostTimeValid)?inputTime->mHostTime:mach_absolute_time());
    for(uint32_t i=0;i<output->mNumberBuffers;i++)
        if(output->mBuffers[i].mData)memset(output->mBuffers[i].mData,0,output->mBuffers[i].mDataByteSize);
    return noErr;
}
