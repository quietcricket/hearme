import XCTest
import AudioRing
import AVFoundation
import Darwin
final class AudioRingTests: XCTestCase {
    func testWraparoundAndBoundedOverflow() {
        let ring=hm_ring_create(4)!
        defer {hm_ring_destroy(ring)}
        let input:[Float]=[1,2,3,4,5]
        XCTAssertEqual(input.withUnsafeBufferPointer {hm_ring_write(ring,$0.baseAddress,5)},4)
        XCTAssertEqual(hm_ring_dropped(ring),1)
        var output=[Float](repeating:0,count:3)
        XCTAssertEqual(output.withUnsafeMutableBufferPointer {hm_ring_read(ring,$0.baseAddress,3)},3)
        XCTAssertEqual(output,[1,2,3])
        _=input.withUnsafeBufferPointer {hm_ring_write(ring,$0.baseAddress,3)}
        output=[Float](repeating:0,count:4)
        XCTAssertEqual(output.withUnsafeMutableBufferPointer {hm_ring_read(ring,$0.baseAddress,4)},4)
        XCTAssertEqual(output,[4,1,2,3])
        XCTAssertEqual(hm_ring_count(ring),0)
    }

    func testStereoCaptureAndFinitePlayback() {
        let ring=hm_ring_create(16)!
        defer {hm_ring_destroy(ring)}
        let stereo=AVAudioPCMBuffer(pcmFormat:AVAudioFormat(standardFormatWithSampleRate:48000,channels:2)!,frameCapacity:2)!
        stereo.frameLength=2
        stereo.floatChannelData![0][0]=0.2;stereo.floatChannelData![0][1]=0.4
        stereo.floatChannelData![1][0]=0.6;stereo.floatChannelData![1][1]=0.8
        XCTAssertEqual(hm_ring_capture(ring,stereo.audioBufferList),2)
        var mixed=[Float](repeating:0,count:2)
        _=mixed.withUnsafeMutableBufferPointer {hm_ring_read(ring,$0.baseAddress,2)}
        XCTAssertEqual(mixed[0],0.4,accuracy:0.00001)
        XCTAssertEqual(mixed[1],0.6,accuracy:0.00001)
        let loud:[Float]=[2,-2,.nan]
        _=loud.withUnsafeBufferPointer {hm_ring_write(ring,$0.baseAddress,3)}
        let mono=AVAudioPCMBuffer(pcmFormat:AVAudioFormat(standardFormatWithSampleRate:48000,channels:1)!,frameCapacity:4)!
        mono.frameLength=4
        hm_ring_render(ring,mono.mutableAudioBufferList,4)
        XCTAssertEqual(mono.floatChannelData![0][0],0.98,accuracy:0.00001)
        XCTAssertEqual(mono.floatChannelData![0][1],-0.98,accuracy:0.00001)
        XCTAssertEqual(mono.floatChannelData![0][2],0)
        XCTAssertEqual(mono.floatChannelData![0][3],0)
        XCTAssertEqual(hm_ring_missing(ring),1)
    }

    func testConcurrentSampleOrdering() {
        let ring=hm_ring_create(1024)!
        defer {hm_ring_destroy(ring)}
        let finished=DispatchSemaphore(value:0)
        let total=100_000
        DispatchQueue.global().async {
            var index=0
            while index<total {
                let chunk=(index..<min(index+37,total)).map(Float.init)
                let count=chunk.withUnsafeBufferPointer {hm_ring_write(ring,$0.baseAddress,UInt32(chunk.count))}
                index+=Int(count)
            }
            finished.signal()
        }
        var received=0
        var output=[Float](repeating:0,count:53)
        while received<total {
            let count=output.withUnsafeMutableBufferPointer {hm_ring_read(ring,$0.baseAddress,53)}
            for i in 0..<Int(count) {XCTAssertEqual(output[i],Float(received+i))}
            received+=Int(count)
        }
        XCTAssertEqual(finished.wait(timeout:.now()+5),.success)
        XCTAssertEqual(hm_ring_count(ring),0)
    }

    func testDiscardRemovesStartupBacklog() {
        let ring=hm_ring_create(4)!
        defer {hm_ring_destroy(ring)}
        let old:[Float]=[0.1,0.2,0.3,0.4,0.5]
        _=old.withUnsafeBufferPointer {hm_ring_write_timed(ring,$0.baseAddress,5,100)}
        XCTAssertEqual(hm_ring_dropped(ring),1)
        hm_ring_discard(ring)
        XCTAssertEqual(hm_ring_count(ring),0)
        XCTAssertEqual(hm_ring_dropped(ring),0)
        let fresh:[Float]=[0.7,0.8]
        _=fresh.withUnsafeBufferPointer {hm_ring_write_timed(ring,$0.baseAddress,2,101)}
        var output=[Float](repeating:0,count:2)
        XCTAssertEqual(hm_ring_head_time(ring),101)
        _=output.withUnsafeMutableBufferPointer {hm_ring_read(ring,$0.baseAddress,2)}
        XCTAssertEqual(output,fresh)
    }

    func testTimestampPropagationAndLatency() {
        let ring=hm_ring_create(16)!
        defer {hm_ring_destroy(ring)}
        var timebase=mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let ticksPerSecond=1e9*Double(timebase.denom)/Double(timebase.numer)
        let now=mach_absolute_time()
        let captureTime=Double(now)/ticksPerSecond-0.25
        let input:[Float]=[0.1,0.2,0.3,0.4]
        _=input.withUnsafeBufferPointer {hm_ring_write_timed(ring,$0.baseAddress,4,captureTime)}
        XCTAssertEqual(hm_ring_head_time(ring),captureTime,accuracy:0.000001)
        let buffer=AVAudioPCMBuffer(pcmFormat:AVAudioFormat(standardFormatWithSampleRate:48000,channels:1)!,frameCapacity:2)!
        buffer.frameLength=2
        hm_ring_render_timed(ring,buffer.mutableAudioBufferList,2,now)
        XCTAssertEqual(hm_ring_latency(ring),0.25,accuracy:0.000001)
        XCTAssertEqual(hm_ring_head_time(ring),captureTime+2.0/48000,accuracy:0.000001)
        XCTAssertEqual(buffer.floatChannelData![0][1],0.2,accuracy:0.00001)
    }

    func testStartupPrerollPreservesSamples() {
        let ring=hm_ring_create(16)!
        defer {hm_ring_destroy(ring)}
        hm_ring_start_delay(ring,0.02)
        let samples:[Float]=[0.2,0.4]
        _=samples.withUnsafeBufferPointer {hm_ring_write(ring,$0.baseAddress,2)}
        let buffer=AVAudioPCMBuffer(pcmFormat:AVAudioFormat(standardFormatWithSampleRate:48000,channels:1)!,frameCapacity:2)!
        buffer.frameLength=2
        hm_ring_render(ring,buffer.mutableAudioBufferList,2)
        XCTAssertEqual(buffer.floatChannelData![0][0],0)
        XCTAssertEqual(hm_ring_count(ring),2)
        XCTAssertEqual(hm_ring_missing(ring),0)
        Thread.sleep(forTimeInterval:0.03)
        hm_ring_render(ring,buffer.mutableAudioBufferList,2)
        XCTAssertEqual(buffer.floatChannelData![0][0],0.2,accuracy:0.00001)
        XCTAssertEqual(buffer.floatChannelData![0][1],0.4,accuracy:0.00001)
        XCTAssertEqual(hm_ring_count(ring),0)
    }
}
