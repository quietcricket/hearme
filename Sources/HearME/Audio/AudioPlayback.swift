import Foundation
import AVFoundation
import AudioRing
import Darwin

final class AudioPlayback {
    private var engine: AVAudioEngine?
    private var configurationObserver: NSObjectProtocol?
    var onConfigurationChange: (() -> Void)?
    func start(device: OutputDevice, ring: OpaquePointer) throws {
        let engine=AVAudioEngine()
        guard let unit=engine.outputNode.audioUnit else {throw AudioFailure(message:"Audio output is unavailable.")}
        var deviceID=device.id
        try checked(AudioUnitSetProperty(unit,kAudioOutputUnitProperty_CurrentDevice,kAudioUnitScope_Global,0,&deviceID,UInt32(MemoryLayout.size(ofValue:deviceID))),"Choose output device")
        let format=AVAudioFormat(standardFormatWithSampleRate:48000,channels:1)!
        let source=AVAudioSourceNode(format:format) { _,timestamp,frames,buffers in
            let time=timestamp.pointee
            let host=(time.mFlags.contains(.hostTimeValid)) ? time.mHostTime:mach_absolute_time()
            hm_ring_render_timed(ring,buffers,frames,host)
            return noErr
        }
        engine.attach(source)
        engine.connect(source,to:engine.mainMixerNode,format:format)
        engine.prepare()
        try engine.start()
        self.engine=engine
        configurationObserver=NotificationCenter.default.addObserver(forName:.AVAudioEngineConfigurationChange,object:engine,queue:nil) { [weak self] _ in
            self?.onConfigurationChange?()
        }
    }
    func stop() {
        if let observer=configurationObserver {NotificationCenter.default.removeObserver(observer)}
        configurationObserver=nil
        engine?.stop();engine=nil
    }
    deinit {stop()}
}
