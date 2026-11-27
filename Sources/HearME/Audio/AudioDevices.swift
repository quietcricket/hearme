import Foundation
import CoreAudio

struct AudioFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

func checked(_ status: OSStatus, _ operation: String) throws {
    guard status == noErr else { throw AudioFailure(message: "\(operation) failed (Core Audio \(status)).") }
}

func address(_ selector: AudioObjectPropertySelector, scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func readID(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> AudioObjectID {
    var a=address(selector), result=AudioObjectID(0), size=UInt32(MemoryLayout<AudioObjectID>.size)
    try checked(AudioObjectGetPropertyData(object,&a,0,nil,&size,&result),"Read audio device")
    return result
}

func readString(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> String {
    var a=address(selector), value: Unmanaged<CFString>?
    var size=UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    try checked(AudioObjectGetPropertyData(object,&a,0,nil,&size,&value),"Read device name")
    guard let value else {throw AudioFailure(message:"The audio device did not provide a name.")}
    return value.takeRetainedValue() as String
}

struct OutputDevice: Identifiable, Hashable {
    let id: AudioObjectID
    let name: String
    let uid: String

    static func systemDefault() throws -> OutputDevice {
        let id=try readID(AudioObjectID(kAudioObjectSystemObject),kAudioHardwarePropertyDefaultOutputDevice)
        guard id != kAudioObjectUnknown,let device=try all().first(where:{$0.id==id}) else {
            throw AudioFailure(message:"Waiting for macOS to provide an audio output.")
        }
        return device
    }

    static func all() throws -> [OutputDevice] {
        var a=address(kAudioHardwarePropertyDevices), size: UInt32=0
        try checked(AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),&a,0,nil,&size),"List audio devices")
        var ids=[AudioObjectID](repeating:0,count:Int(size)/MemoryLayout<AudioObjectID>.size)
        try checked(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),&a,0,nil,&size,&ids),"List audio devices")
        return ids.compactMap { id in
            var streamAddress=address(kAudioDevicePropertyStreams,scope:kAudioDevicePropertyScopeOutput)
            var bytes: UInt32=0
            guard AudioObjectGetPropertyDataSize(id,&streamAddress,0,nil,&bytes)==noErr,bytes>0,
                  let name=try? readString(id,kAudioObjectPropertyName),
                  let uid=try? readString(id,kAudioDevicePropertyDeviceUID), !uid.hasPrefix("HearME-") else {return nil}
            return OutputDevice(id:id,name:name,uid:uid)
        }
    }
}

/// Watches system routing without changing the user's macOS selection.
final class SystemOutputMonitor {
    private var listener: AudioObjectPropertyListenerBlock?
    private let selectors: [AudioObjectPropertySelector]=[
        kAudioHardwarePropertyDefaultOutputDevice,kAudioHardwarePropertyDevices
    ]
    func start(onChange: @escaping () -> Void) throws {
        let block: AudioObjectPropertyListenerBlock = { _,_ in onChange() }
        listener=block
        for selector in selectors {
            var property=address(selector)
            try checked(AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),&property,.main,block),"Watch macOS audio output")
        }
    }
    deinit {
        if let listener {
            for selector in selectors {
                var property=address(selector)
                AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),&property,.main,listener)
            }
        }
    }
}
