#pragma once
#include <stdint.h>
#include <CoreAudio/CoreAudio.h>
typedef struct HMRing HMRing;
HMRing * _Nullable hm_ring_create(uint32_t capacity);
void hm_ring_discard(HMRing * _Nonnull ring);
double hm_ring_head_time(HMRing * _Nonnull ring);
void hm_ring_sample_rate(HMRing * _Nonnull ring, double rate);
uint32_t hm_ring_write_timed(HMRing * _Nonnull ring, const float * _Null_unspecified samples, uint32_t count, double input_time);
void hm_ring_render_timed(HMRing * _Nonnull ring, AudioBufferList * _Nonnull audio, uint32_t frames, uint64_t host_time);
double hm_ring_latency(HMRing * _Nonnull ring);
void hm_ring_destroy(HMRing * _Nullable ring);
uint32_t hm_ring_count(HMRing * _Nonnull ring);
uint64_t hm_ring_total_written(HMRing * _Nonnull ring);
uint32_t hm_ring_write(HMRing * _Nonnull ring, const float * _Null_unspecified samples, uint32_t count);
uint32_t hm_ring_read(HMRing * _Nonnull ring, float * _Null_unspecified samples, uint32_t count);
uint32_t hm_ring_capture(HMRing * _Nonnull ring, const AudioBufferList * _Nonnull audio);
void hm_ring_render(HMRing * _Nonnull ring, AudioBufferList * _Nonnull audio, uint32_t frames);
void hm_ring_gain(HMRing * _Nonnull ring, float gain);
void hm_ring_start_delay(HMRing * _Nonnull ring, double seconds);
float hm_ring_take_peak(HMRing * _Nonnull ring);
uint64_t hm_ring_dropped(HMRing * _Nonnull ring);
uint64_t hm_ring_missing(HMRing * _Nonnull ring);
OSStatus hm_audio_io_proc(AudioObjectID device, const AudioTimeStamp * _Nonnull now,
 const AudioBufferList * _Nonnull input, const AudioTimeStamp * _Nonnull inputTime,
 AudioBufferList * _Nonnull output, const AudioTimeStamp * _Nonnull outputTime, void * _Nullable context);
