import Capacitor
import CoreML
import Foundation

@objc(PeytonTTSPlugin)
public class PeytonTTSPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "PeytonTTSPlugin"
    public let jsName = "PeytonTTS"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "status", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "forward", returnType: CAPPluginReturnPromise)
    ]

    private let seqLen = 64
    private let melDim = 100
    private var model: MLModel?

    @objc func status(_ call: CAPPluginCall) {
        call.resolve([
            "available": modelURL() != nil,
            "runtime": "coreml",
            "model": "F5TTS_Peyton_DiT_seq64",
            "quantization": "int4-palettized",
            "seqLen": seqLen,
            "melDim": melDim
        ])
    }

    @objc func forward(_ call: CAPPluginCall) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let xValues = try self.floatArray(call, "x", expected: self.seqLen * self.melDim)
                let condValues = try self.floatArray(call, "cond", expected: self.seqLen * self.melDim)
                let textValues = try self.intArray(call, "text", expected: self.seqLen)
                let timeValue = call.getDouble("time") ?? 0.0

                let model = try self.loadModel()
                let x = try self.multiArrayFloat32([1, NSNumber(value: self.seqLen), NSNumber(value: self.melDim)], xValues)
                let cond = try self.multiArrayFloat32([1, NSNumber(value: self.seqLen), NSNumber(value: self.melDim)], condValues)
                let text = try self.multiArrayInt32([1, NSNumber(value: self.seqLen)], textValues)
                let time = try self.multiArrayFloat32([1], [Float(timeValue)])
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "x": MLFeatureValue(multiArray: x),
                    "cond": MLFeatureValue(multiArray: cond),
                    "text": MLFeatureValue(multiArray: text),
                    "time": MLFeatureValue(multiArray: time)
                ])
                let result = try model.prediction(from: input)
                guard let pred = result.featureValue(for: "pred")?.multiArrayValue else {
                    throw PeytonTTSError.invalidOutput
                }
                call.resolve([
                    "pred": self.floatArray(pred),
                    "seqLen": self.seqLen,
                    "melDim": self.melDim
                ])
            } catch {
                call.reject("Peyton native TTS forward failed: \(error.localizedDescription)")
            }
        }
    }

    private func modelURL() -> URL? {
        if let url = Bundle.main.url(
            forResource: "F5TTS_Peyton_DiT_seq64",
            withExtension: "mlmodelc",
            subdirectory: "NativeModels"
        ) {
            return url
        }
        return Bundle.main.url(
            forResource: "F5TTS_Peyton_DiT_seq64",
            withExtension: "mlpackage",
            subdirectory: "NativeModels"
        )
    }

    private func loadModel() throws -> MLModel {
        if let model = model { return model }
        guard let url = modelURL() else { throw PeytonTTSError.modelMissing }
        let loadURL: URL
        if url.pathExtension == "mlpackage" {
            loadURL = try MLModel.compileModel(at: url)
        } else {
            loadURL = url
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let loaded = try MLModel(contentsOf: loadURL, configuration: config)
        model = loaded
        return loaded
    }

    private func floatArray(_ call: CAPPluginCall, _ key: String, expected: Int) throws -> [Float] {
        guard let raw = call.getArray(key) else { throw PeytonTTSError.missingInput(key) }
        let values = raw.compactMap { value -> Float? in
            if let number = value as? NSNumber { return number.floatValue }
            if let double = value as? Double { return Float(double) }
            if let int = value as? Int { return Float(int) }
            return nil
        }
        guard values.count == expected else { throw PeytonTTSError.invalidInputCount(key, values.count, expected) }
        return values
    }

    private func intArray(_ call: CAPPluginCall, _ key: String, expected: Int) throws -> [Int32] {
        guard let raw = call.getArray(key) else { throw PeytonTTSError.missingInput(key) }
        let values = raw.compactMap { value -> Int32? in
            if let number = value as? NSNumber { return number.int32Value }
            if let int = value as? Int { return Int32(int) }
            return nil
        }
        guard values.count == expected else { throw PeytonTTSError.invalidInputCount(key, values.count, expected) }
        return values
    }

    private func multiArrayFloat32(_ shape: [NSNumber], _ values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        for index in 0..<values.count {
            array[index] = NSNumber(value: values[index])
        }
        return array
    }

    private func multiArrayInt32(_ shape: [NSNumber], _ values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        for index in 0..<values.count {
            array[index] = NSNumber(value: values[index])
        }
        return array
    }

    private func floatArray(_ array: MLMultiArray) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(array.count)
        for index in 0..<array.count {
            values.append(array[index].floatValue)
        }
        return values
    }
}

enum PeytonTTSError: LocalizedError {
    case modelMissing
    case invalidOutput
    case missingInput(String)
    case invalidInputCount(String, Int, Int)

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "F5TTS_Peyton_DiT_seq64 Core ML model is not bundled"
        case .invalidOutput:
            return "Core ML output pred was missing"
        case .missingInput(let key):
            return "missing input \(key)"
        case .invalidInputCount(let key, let actual, let expected):
            return "input \(key) had \(actual) values, expected \(expected)"
        }
    }
}
