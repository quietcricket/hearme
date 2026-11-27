import Foundation

struct ModelRuntime: Hashable {
  let assets: URL
  let frames: Int

  static func resolve(blockSeconds: Double = 0.5) throws -> ModelRuntime {
    guard let root = Bundle.main.resourceURL else {
      throw AudioFailure(message: "Voice enhancement assets are missing.")
    }
    let assets = root.appendingPathComponent("MLX")
    guard FileManager.default.fileExists(
      atPath: assets.appendingPathComponent("weights.safetensors").path)
    else {
      throw AudioFailure(
        message: "The TF-Restormer model is missing. Export the model and rebuild HearME.")
    }
    let frames = blockSeconds < 0.2 ? 5 : blockSeconds < 0.5 ? 10 : 25
    return ModelRuntime(assets: assets, frames: frames)
  }
}
