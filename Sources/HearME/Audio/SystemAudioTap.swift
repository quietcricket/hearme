import Foundation
import CoreAudio
import AudioRing

final class SystemAudioTap {
    private var tap: AudioObjectID=0
    private var aggregate: AudioObjectID=0
    private var proc: AudioDeviceIOProcID?
    private(set) var sampleRate: Double=48000

    func start(ring: OpaquePointer, output: OutputDevice) throws {
        do {
            var a=address(kAudioHardwarePropertyTranslatePIDToProcessObject)
            var pid=getpid(), own=AudioObjectID(0), size=UInt32(MemoryLayout<AudioObjectID>.size)
            try checked(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),&a,UInt32(MemoryLayout<pid_t>.size),&pid,&size,&own),"Exclude HearME playback")
            guard own != kAudioObjectUnknown else {throw AudioFailure(message:"HearME's output process could not be excluded from capture.")}
            let description=CATapDescription(stereoGlobalTapButExcludeProcesses:[own])
            description.name="HearME system audio"
            description.isPrivate=true
            description.muteBehavior = .mutedWhenTapped
            try checked(AudioHardwareCreateProcessTap(description,&tap),"Create system audio tap")
            let tapUID=try readString(tap,kAudioTapPropertyUID)
            var format=AudioStreamBasicDescription()
            var formatAddress=address(kAudioTapPropertyFormat)
            size=UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            try checked(AudioObjectGetPropertyData(tap,&formatAddress,0,nil,&size,&format),"Read captured audio format")
            guard format.mFormatID==kAudioFormatLinearPCM,format.mBitsPerChannel==32,
                  format.mFormatFlags & kAudioFormatFlagIsFloat != 0 else {
                throw AudioFailure(message:"The system audio tap did not provide 32-bit floating-point audio.")
            }
            sampleRate=format.mSampleRate
            // Use an output-only hardware clock. Never include microphone streams
            // from a duplex headset in the aggregate input.
            let devices=try OutputDevice.all()
            let clock=([output]+devices).first { device in
                var inputAddress=address(kAudioDevicePropertyStreams,scope:kAudioDevicePropertyScopeInput)
                var inputSize: UInt32=0
                return AudioObjectGetPropertyDataSize(device.id,&inputAddress,0,nil,&inputSize)==noErr && inputSize==0
            }
            guard let clock else {throw AudioFailure(message:"No output-only device is available for the capture clock. Connect headphones or enable the built-in output.")}
            let spec: [String:Any]=[
                kAudioAggregateDeviceNameKey:"HearME private capture",
                kAudioAggregateDeviceUIDKey:"HearME-\(UUID().uuidString)",
                kAudioAggregateDeviceIsPrivateKey:true,
                kAudioAggregateDeviceIsStackedKey:false,
                kAudioAggregateDeviceSubDeviceListKey:[[kAudioSubDeviceUIDKey:clock.uid]],
                kAudioAggregateDeviceMainSubDeviceKey:clock.uid,
                kAudioAggregateDeviceTapAutoStartKey:true,
                kAudioAggregateDeviceTapListKey:[[kAudioSubTapUIDKey:tapUID,kAudioSubTapDriftCompensationKey:true]]
            ]
            try checked(AudioHardwareCreateAggregateDevice(spec as CFDictionary,&aggregate),"Create private capture device")
            // Aggregate creation is asynchronous. Do not register I/O until HAL
            // reports the device alive; this runs only on the routing queue.
            var aliveAddress=address(kAudioDevicePropertyDeviceIsAlive)
            var alive: UInt32=0
            for _ in 0..<30 {
                var aliveSize=UInt32(MemoryLayout<UInt32>.size)
                let status=AudioObjectGetPropertyData(aggregate,&aliveAddress,0,nil,&aliveSize,&alive)
                if status==noErr && alive != 0 {break}
                Thread.sleep(forTimeInterval:0.1)
            }
            guard alive != 0 else {throw AudioFailure(message:"The capture device did not become ready. Try starting again.")}
            // The only input is the stereo process tap; the hardware subdevice
            // is output-only and provides a reliable clock on live audio.
            var aggregateFormat=AudioStreamBasicDescription()
            var aggregateAddress=address(kAudioDevicePropertyStreamFormat,scope:kAudioDevicePropertyScopeInput)
            size=UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            if AudioObjectGetPropertyData(aggregate,&aggregateAddress,0,nil,&size,&aggregateFormat)==noErr {
                sampleRate=aggregateFormat.mSampleRate
            }
            hm_ring_sample_rate(ring,sampleRate)
            try checked(AudioDeviceCreateIOProcID(aggregate,hm_audio_io_proc,UnsafeMutableRawPointer(ring),&proc),"Register system audio capture")
            try checked(AudioDeviceStart(aggregate,proc),"Start capture. Allow HearME in System Settings → Privacy & Security → Screen & System Audio Recording")
        } catch { stop(); throw error }
    }

    func stop() {
        if aggregate != 0 {
            if let proc {AudioDeviceStop(aggregate,proc); AudioDeviceDestroyIOProcID(aggregate,proc)}
            AudioHardwareDestroyAggregateDevice(aggregate)
        }
        if tap != 0 {AudioHardwareDestroyProcessTap(tap)}
        aggregate=0;tap=0;proc=nil
    }
    deinit {stop()}
}
