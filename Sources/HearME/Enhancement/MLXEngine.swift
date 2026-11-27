#if canImport(MLX)
  import Accelerate
  import Foundation
  import MLX

  /// GPU-resident neural graph; one evaluation per audio block. No Python runtime.
  final class MLXEngine {
    private let useAutoreleasePool: Bool
    let frames: Int
    var sampleCount: Int { frames * 960 }
    private let weights: [String: MLXArray]
    private var cache = [String: MLXArray]()
    private let scan: MLXFast.MLXFastKernel
    private let fir, analysis, synthesis: [Float]
    private var firHistory = [Float](repeating: 0, count: 126)
    private var analysisHistory = [Float](repeating: 0, count: 320)
    private var synthesisHistory = [Float](repeating: 0, count: 960)
    private var histories = [MLXArray]()
    private var ready = false
    private let upMinimum: Int
    private var neural: (([MLXArray]) -> [MLXArray])!

    init(assets: URL, frames: Int, useAutoreleasePool: Bool = true) throws {
      self.useAutoreleasePool = useAutoreleasePool
      guard [5, 10, 25].contains(frames) else { throw Self.failure("Unsupported MLX batch size") }
      self.frames = frames
      var loadedWeights = try loadArrays(url: assets.appendingPathComponent("weights.safetensors"))
      let metadata =
        try JSONSerialization.jsonObject(
          with: Data(contentsOf: assets.appendingPathComponent("metadata.json"))) as? [String: Any]
      guard let architecture = metadata?["architecture"] as? [String: Int],
        let minimum = architecture["up_min_F"]
      else { throw Self.failure("Invalid MLX model metadata") }
      for (name, original) in metadata?["aliases"] as? [String: String] ?? [:] {
        guard let value = loadedWeights[original] else {
          throw Self.failure("Invalid shared model weight")
        }
        loadedWeights[name] = value
      }
      weights = loadedWeights
      upMinimum = minimum
      func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: assets.appendingPathComponent(name + ".f32"))
        guard data.count % 4 == 0 else { throw Self.failure("Invalid DSP weights") }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
      }
      fir = try floats("fir")
      analysis = try floats("analysis")
      synthesis = try floats("synthesis")
      guard fir.count == 127, analysis.count == 642 * 640, synthesis.count == 1922 * 1920 else {
        throw Self.failure("Invalid DSP weight dimensions")
      }
      scan = MLXFast.metalKernel(
        name: "hearme_scan",
        inputNames: ["u", "dt", "bc", "xz", "A", "D", "bias", "state", "ready"],
        outputNames: ["out", "newstate"],
        source: try String(contentsOf: assets.appendingPathComponent("Scan.metal"), encoding: .utf8)
      )
      reset()
      neural = compile(inputs: [], outputs: [], shapeless: false) { [unowned self] arrays in
        self.forward(arrays)
      }
    }
    private static func failure(_ text: String) -> NSError {
      NSError(domain: "HearME.MLX", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
    }
    private func w(_ key: String, half: Bool = true) -> MLXArray {
      let name = key + (half ? ".half" : ".float")
      if let value = cache[name] { return value }
      let value = weights[key]!.asType(half ? .float16 : .float32)
      cache[name] = value
      return value
    }
    private func linear(_ x: MLXArray, _ p: String) -> MLXArray {
      var y = matmul(x.asType(.float16), w(p + ".weight").transposed())
      if weights[p + ".bias"] != nil { y = y + w(p + ".bias") }
      return y
    }
    private func norm(_ x: MLXArray, _ p: String) -> MLXArray {
      MLXFast.layerNorm(
        x.asType(.float32), weight: w(p + ".weight", half: false),
        bias: w(p + ".bias", half: false), eps: 1e-5)
    }
    private func conv1(_ x: MLXArray, _ p: String) -> MLXArray {
      let k = w(p + ".weight")
      return conv1d(x.asType(.float16), k.transposed(0, 2, 1), padding: k.dim(2) / 2)
        + w(p + ".bias")
    }
    private func ffn(_ x: MLXArray, _ p: String) -> MLXArray {
      let h = conv1(norm(x, p + ".layernorm"), p + ".net1")
      let parts = split(h, parts: 2, axis: -1)
      return x + 0.5 * conv1(parts[0] * sigmoid(parts[0]) * parts[1], p + ".net2").asType(.float32)
    }
    private func attention(_ x: MLXArray, _ p: String, encoded: MLXArray? = nil) -> MLXArray {
      let heads = w(p + ".Ekv.Ekv1").dim(0)
      let batch = x.dim(0)
      let length = x.dim(1)
      let q: MLXArray
      let k: MLXArray
      let v: MLXArray
      if let encoded {
        q = linear(x, p + ".q").reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        let kv = linear(encoded, p + ".kv").reshaped(batch, encoded.dim(1), 2, heads, -1)
          .transposed(2, 0, 3, 1, 4)
        k = kv[0]
        v = kv[1]
      } else {
        let qkv = linear(x, p + ".qkv").reshaped(batch, length, 3, heads, -1).transposed(
          2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
      }
      let e = w(p + ".Ekv.Ekv1")[0..., 0..., ..<k.dim(-2)][.newAxis]
      let projectedK = matmul(e, k)
      let projectedV = matmul(e, v)
      let y = MLXFast.scaledDotProductAttention(
        queries: q, keys: projectedK, values: projectedV, scale: 1 / sqrt(Float(q.dim(-1))),
        mask: nil)
      return linear(y.transposed(0, 2, 1, 3).reshaped(batch, length, -1), p + ".aggregate_heads.0")
        .asType(.float32)
    }
    private func frequency(_ input: MLXArray, _ p: String, encoded: MLXArray? = nil) -> MLXArray {
      let b = input.dim(0)
      let f = input.dim(1)
      let t = input.dim(2)
      let c = input.dim(3)
      var x = input.transposed(0, 2, 1, 3).reshaped(b * t, f, c)
      if let encoded {
        let e = encoded.transposed(0, 2, 1, 3).reshaped(b * t, 321, 128)
        let orig = x[0..., ..<321]
        let pad = x[0..., 321...]
        let updated =
          pad + attention(norm(pad, p + ".ca.layer_norm_q"), p + ".ca.block", encoded: e)
        x = concatenated([orig, updated], axis: 1)
        x = x + attention(norm(x, p + ".sa.layer_norm"), p + ".sa.block")
        x = ffn(x, p + ".ffn")
      } else {
        x = ffn(x, p + ".ffn_1")
        x = x + attention(norm(x, p + ".sa.layer_norm"), p + ".sa.block")
        x = ffn(x, p + ".ffn_2")
      }
      return x.reshaped(b, t, f, c).transposed(0, 2, 1, 3)
    }
    private func mamba(_ x: MLXArray, _ p: String, conv: MLXArray, state: MLXArray, ready: MLXArray)
      -> (MLXArray, MLXArray, MLXArray)
    {
      let b = x.dim(0)
      let f = x.dim(1)
      let t = x.dim(2)
      let c = x.dim(3)
      let h = x.reshaped(f, t, c)
      let z = linear(norm(h, p + ".layernorm"), p + ".mamba_fw.in_proj")
      let n = z.dim(-1) / 2
      let prefix = p + ".mamba_fw"
      let valid = which(MLXArray(0..<t) .< 2, ready, Float(1)).reshaped(1, t, 1)
      let joined = concatenated([conv, z[0..., 0..., ..<n].asType(.float32) * valid], axis: 1)
      let weight = w(prefix + ".conv1d.weight", half: false)[0..., 0, 0...]
      let u0 = joined[0..., ..<t] * weight[0..., 0] + joined[0..., 1..<(t + 1)] * weight[0..., 1]
      let u1 =
        u0 + joined[0..., 2..<(t + 2)] * weight[0..., 2] + w(prefix + ".conv1d.bias", half: false)
      let u = (u1 * sigmoid(u1)).asType(.float16)
      let bc = linear(u, prefix + ".x_proj")
      let rank = bc.dim(-1) - 32
      let dt = matmul(bc[0..., 0..., ..<rank], w(prefix + ".dt_proj.weight").transposed())
      let a = -exp(w(prefix + ".A_log", half: false))
      let scanned = scan(
        [
          u, dt, bc, z, a, w(prefix + ".D", half: false), w(prefix + ".dt_proj.bias", half: false),
          state, ready,
        ], template: [("F", f), ("N", n), ("T", t), ("R", rank)], grid: (f * n, 1, 1),
        threadGroup: (128, 1, 1), outputShapes: [[f, t, n], [f, 16, n]],
        outputDTypes: [.float16, .float32])
      let y = linear(scanned[0], prefix + ".out_proj").asType(.float32) * valid
      return ((h + y).reshaped(b, f, t, c), joined[0..., (joined.dim(1) - 2)...], scanned[1])
    }
    private func forward(_ arrays: [MLXArray]) -> [MLXArray] {
      let features = arrays[0]
      let ready = arrays[1]
      let old = Array(arrays.dropFirst(2))
      let t = frames
      let valid = which(MLXArray(0..<t) .< 2, ready, Float(1)).reshaped(1, 1, t, 1)
      let joined = concatenated([old[0], features], axis: 2)
      var x =
        conv2d(
          joined.asType(.float16), w("input_embed.embed.conv.weight").transposed(0, 2, 3, 1),
          padding: [1, 0]) + w("input_embed.embed.conv.bias")
      let inverse = exp((-log(Float(10000)) / 63) * MLXArray(0..<64).asType(.float32))
      let pos = MLXArray(0..<321).asType(.float32)[0..., .newAxis] * inverse[.newAxis]
      x =
        norm(x, "input_embed.layernorm") + w("input_embed.freq_pe.p", half: false)[0..., ..<321]
        + concatenated([sin(pos), cos(pos)], axis: -1)[.newAxis, 0..., .newAxis, 0...]
      var updated = [joined[0..., 0..., (joined.dim(2) - 2)...]]
      var index = 1
      for i in 0..<6 {
        x = frequency(x, "encoder.tf_block.\(i).frame_wise_block.block")
        for j in 0..<2 {
          let result = mamba(
            x, "encoder.tf_block.\(i).freq_wise_block.block.\(j)", conv: old[index],
            state: old[index + 1], ready: ready)
          x = result.0
          updated += [result.1, result.2]
          index += 2
        }
      }
      let encoded = norm(x, "encoder.layer_norm")
      let low = linear(norm(x, "encoder.layer_norm"), "up.proj")
      let pad = broadcast(
        w("up.mask_token")[0..., (321 - upMinimum)..<(961 - upMinimum)], to: [1, 640, t, 64])
      x = norm(concatenated([low, pad], axis: 1), "up.layernorm")
      for i in 0..<3 {
        x = frequency(x, "decoder.tf_block.\(i).frame_wise_block.block", encoded: encoded)
        for j in 0..<2 {
          let result = mamba(
            x, "decoder.tf_block.\(i).freq_wise_block.block.\(j)", conv: old[index],
            state: old[index + 1], ready: ready)
          x = result.0
          updated += [result.1, result.2]
          index += 2
        }
      }
      let hidden = norm(x, "decoder.layer_norm") * valid
      let outputJoined = concatenated([old.last!, hidden], axis: 2)
      func conv(_ p: String) -> MLXArray {
        conv2d(
          outputJoined.asType(.float16), w(p + ".weight").transposed(0, 2, 3, 1), padding: [1, 0])
          + w(p + ".bias")
      }
      let mag = logAddExp(conv("estimator.mag.out.0.conv").asType(.float32), MLXArray(Float(0)))
      let phase = tanh(conv("estimator.phase.out.0.conv").asType(.float32))
      let pred = mag * phase * which(MLXArray(0..<t) .< 3, ready, Float(1)).reshaped(1, 1, t, 1)
      updated.append(outputJoined[0..., 0..., (outputJoined.dim(2) - 2)...])
      return [pred.transposed(0, 2, 3, 1).reshaped(t, 1922)] + updated
    }
    func reset() {
      histories = [MLXArray.zeros([1, 321, 2, 2])]
      for (stage, layers, frequency) in [("encoder", 6, 321), ("decoder", 3, 961)] {
        for i in 0..<layers {
          for j in 0..<2 {
            let key = "\(stage).tf_block.\(i).freq_wise_block.block.\(j).mamba_fw.in_proj.weight"
            let inner = weights[key]!.dim(0) / 2
            histories += [.zeros([frequency, 2, inner]), .zeros([frequency, 16, inner])]
          }
        }
      }
      histories.append(MLXArray.zeros([1, 961, 2, 64]))
      firHistory = Array(repeating: 0, count: 126)
      analysisHistory = Array(repeating: 0, count: 320)
      synthesisHistory = Array(repeating: 0, count: 960)
      ready = false
    }
    func process(_ samples: [Float]) throws -> [Float] {
      if useAutoreleasePool { return try autoreleasepool { try processBlock(samples) } }
      return try processBlock(samples)
    }
    private func processBlock(_ samples: [Float]) throws -> [Float] {
      guard samples.count == sampleCount else { throw Self.failure("Invalid MLX audio block size") }
      let joined = firHistory + samples
      firHistory = Array(joined.suffix(126))
      var down = [Float](repeating: 0, count: frames * 320)
      joined.withUnsafeBufferPointer { x in
        fir.withUnsafeBufferPointer { h in
          down.withUnsafeMutableBufferPointer { y in
            vDSP_desamp(
              x.baseAddress!, 3, h.baseAddress!, y.baseAddress!, vDSP_Length(frames * 320), 127)
          }
        }
      }
      let waveform = analysisHistory + down
      analysisHistory = Array(waveform.suffix(320))
      var windows = [Float](repeating: 0, count: frames * 640)
      for t in 0..<frames { for n in 0..<640 { windows[t * 640 + n] = waveform[t * 320 + n] } }
      var spectral = [Float](repeating: 0, count: frames * 642)
      windows.withUnsafeBufferPointer { x in
        analysis.withUnsafeBufferPointer { k in
          spectral.withUnsafeMutableBufferPointer { y in
            cblas_sgemm(
              CblasRowMajor, CblasNoTrans, CblasTrans, Int32(frames), 642, 640, 1, x.baseAddress!,
              640, k.baseAddress!, 640, 0, y.baseAddress!, 642)
          }
        }
      }
      var features = [Float](repeating: 0, count: frames * 642)
      for f in 0..<321 {
        for t in 0..<frames {
          for c in 0..<2 {
            features[(f * frames + t) * 2 + c] =
              !ready && t == 0 ? 0 : spectral[t * 642 + c * 321 + f]
          }
        }
      }
      let result = neural(
        [MLXArray(features, [1, 321, frames, 2]), MLXArray([ready ? Float(1) : Float(0)])]
          + histories)
      eval(result)
      histories = Array(result.dropFirst())
      let spec = result[0].asArray(Float.self)
      var synthesized = [Float](repeating: 0, count: frames * 1920)
      spec.withUnsafeBufferPointer { x in
        synthesis.withUnsafeBufferPointer { k in
          synthesized.withUnsafeMutableBufferPointer { y in
            cblas_sgemm(
              CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(frames), 1920, 1922, 1,
              x.baseAddress!, 1922, k.baseAddress!, 1920, 0, y.baseAddress!, 1920)
          }
        }
      }
      var wave = [Float](repeating: 0, count: sampleCount + 960)
      for t in 0..<frames { for n in 0..<1920 { wave[t * 960 + n] += synthesized[t * 1920 + n] } }
      for n in 0..<960 { wave[n] += synthesisHistory[n] }
      synthesisHistory = Array(wave.suffix(960))
      ready = true
      return Array(wave.prefix(sampleCount))
    }
  }
#endif
