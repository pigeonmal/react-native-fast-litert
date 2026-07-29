# react-native-fast-litert

A high-performance [LiteRT](https://developers.google.com/edge/litert) library for React Native, built with [Nitro Modules](https://github.com/mrousavy/nitro).

- ⚡ Powered by Nitro Modules
- 💨 Direct ArrayBuffer input and output
- 🔧 Uses the LiteRT 2.1.6 C++ CompiledModel runtime
- 🔄 Supports swapping out TensorFlow Models at runtime
- 🖥️ Prefers GPU acceleration through Metal on iOS and OpenCL/OpenGL on Android
- 📸 Easy [VisionCamera](https://github.com/mrousavy/react-native-vision-camera) integration

## Migrating from v2

If you are upgrading from v2, see the [Migration Guide](./MIGRATION_V2_TO_V3.md) for breaking changes and upgrade steps.

## Installation

1. Add the npm packages:
   ```sh
   yarn add react-native-fast-litert react-native-nitro-modules
   ```
2. In `metro.config.js`, add `tflite` as a supported asset extension:
   ```js
   module.exports = {
     // ...
     resolver: {
       assetExts: ['tflite', // ...
       // ...
   ```
   This allows you to drop `.tflite` files into your app and swap them out at runtime without rebuilding. 🔥
3. GPU acceleration is bundled and preferred by default. See [GPU acceleration](#gpu-acceleration) for Android vendor-library configuration.
4. Run your app (`yarn android` / `npx pod-install && yarn ios`)

LiteRT 2.1.6 requires iOS 15 or newer. Android supports `armeabi-v7a`,
`arm64-v8a`, and `x86_64`; LiteRT 2.1.6 does not publish an Android `x86`
runtime.

## Usage

1. Find a TensorFlow Lite (`.tflite`) model. There are thousands of public models on [tfhub.dev](https://tfhub.dev).
2. Drag your model into your app's asset folder (e.g. `src/assets/my-model.tflite`)
3. Load the Model:

   ```ts
   // Option A: Standalone Function
   const model = await loadTensorflowModel(require('assets/my-model.tflite'))

   // Option B: Hook in a Function Component
   const plugin = useTensorflowModel(require('assets/my-model.tflite'))
   ```

4. Call the Model:
   ```ts
   const inputData: ArrayBuffer = ...
   const outputData = await model.run([inputData])
   console.log(outputData)
   ```

### Loading Models

Models can be loaded from the React Native bundle via `require(..)`, or any URI/URL (`http://..` or `file://..`):

```ts
// Asset from React Native Bundle
loadTensorflowModel(require('assets/my-model.tflite'))
// File on the local filesystem
loadTensorflowModel({ url: 'file:///var/mobile/.../my-model.tflite' })
// Remote URL
loadTensorflowModel(
  { url: 'https://tfhub.dev/google/lite-model/object_detection_v1.tflite' }
)
```

Loading a Model is asynchronous since buffers need to be allocated. Make sure to handle errors when loading.

### Input and Output data

TensorFlow uses _tensors_ as input and output. Since TensorFlow Lite is optimized for fixed-size byte buffers, you are responsible for interpreting the raw data yourself.

Input and output values are passed as `ArrayBuffer`. To inspect tensor shapes, open your model in [Netron](https://netron.app).

For example, the `object_detection_mobile_object_localizer_v1_1_default_1.tflite` model on [tfhub.dev](https://tfhub.dev) has **1 input tensor** and **4 output tensors**:

![Screenshot of netron.app inspecting the model](./img/netron-inspect-model.png)

In the description on [tfhub.dev](https://tfhub.dev) we can find the description of all tensors:

![Screenshot of tfhub.dev inspecting the model](./img/tfhub-description.png)

From that we know we need a 192 x 192 input image with 3 bytes per pixel (RGB).

#### Usage (VisionCamera)

If you're using this model with a [VisionCamera](https://github.com/mrousavy/react-native-vision-camera) Frame Processor, you need to convert the Frame to the model's expected input size.
Use [vision-camera-resizer](https://visioncamera.margelo.com/api/react-native-vision-camera-resizer) to do the conversion:

```tsx
import { Camera, useFrameOutput } from 'react-native-vision-camera'
import { useResizer } from 'react-native-vision-camera-resizer'
import { useTensorflowModel } from 'react-native-fast-litert'

const objectDetection = useTensorflowModel(require('object_detection.tflite'))

// 1. Create a resizer that converts Frames to 192x192x3 (RGB, uint8)
const { resizer } = useResizer({
  width: 192,
  height: 192,
  channelOrder: 'rgb',
  dataType: 'uint8',
})

const frameOutput = useFrameOutput({
  pixelFormat: 'yuv',
  onFrame(frame) {
    'worklet'
    if (objectDetection.state !== 'loaded' || resizer == null) {
      frame.dispose()
      return
    }

    // 2. Resize the Frame to the model's input size
    const resized = resizer.resize(frame)
    frame.dispose()
    const data = new Uint8Array(resized.getPixelBuffer())
    resized.dispose()

    // 3. Extract the exact slice of the underlying ArrayBuffer
    const inputBuffer = data.buffer.slice(
      data.byteOffset,
      data.byteOffset + data.byteLength
    )

    // 4. Run model with given input buffer synchronously
    const outputs = objectDetection.model.runSync([inputBuffer])

    // 5. Interpret outputs accordingly
    const detection_boxes = new Float32Array(outputs[0]!)
    const detection_classes = new Float32Array(outputs[1]!)
    const detection_scores = new Float32Array(outputs[2]!)
    const num_detections = new Float32Array(outputs[3]!)
    console.log(`Detected ${num_detections[0]} objects!`)
  },
})

return <Camera device="back" isActive={true} outputs={[frameOutput]} {...otherProps} />
```

> [!NOTE]
> Unlike v4, VisionCamera v5 no longer requires boxing the model with `NitroModules.box()`. Since v5 is built on Nitro Modules and uses [react-native-worklets](https://docs.swmansion.com/react-native-worklets/), worklets can access HybridObjects like the TFLite model directly.

### GPU acceleration

`loadTensorflowModel(...)` and `useTensorflowModel(...)` default to `['gpu']`.
LiteRT compiles supported operations for Metal on iOS or OpenCL/OpenGL on
Android, while unsupported operations remain on CPU. If GPU compilation is not
safe or fails, model creation automatically retries on CPU.

```ts
// GPU preferred, CPU fallback
const model = await loadTensorflowModel(require('assets/my-model.tflite'))

// Explicitly require CPU
const cpuModel = await loadTensorflowModel(
  require('assets/my-model.tflite'),
  []
)
```

The platform-specific `metal` and `android-gpu` names remain aliases for
`gpu`. The old `core-ml` and `nnapi` delegates are not part of LiteRT 2.x's
CompiledModel API.

On Android 12 and newer, GPU access can additionally require vendor native
libraries to be declared in the app manifest.

#### Expo

##### Expo

Use the config plugin in your expo config (`app.json`, `app.config.json` or `app.config.js`) with `enableAndroidGpuLibraries`:

```json
{
  "name": "my app",
  "plugins": [
    [
      "react-native-fast-litert",
      {
        "enableAndroidGpuLibraries": true
      }
    ]
  ]
}
```

By default, when enabled, `libOpenCL.so` will be included in your `AndroidManifest.xml`. You can also include more libraries by passing an array:

```json
{
  "name": "my app",
  "plugins": [
    [
      "react-native-fast-litert",
      {
        "enableAndroidGpuLibraries": ["libOpenCL-pixel.so", "libGLES_mali.so"]
      }
    ]
  ]
}
```

> [!NOTE]
> For Expo, remember to run prebuild if the library is not yet included in your `AndroidManifest.xml`.

##### Bare React Native

Add any needed entries to your `AndroidManifest.xml`:

```xml
<uses-native-library android:name="libOpenCL.so" android:required="false" />
<uses-native-library android:name="libOpenCL-pixel.so" android:required="false" />
<uses-native-library android:name="libGLES_mali.so" android:required="false" />
<uses-native-library android:name="libPVROCL.so" android:required="false" />
```

> [!NOTE]
> Android does not officially support OpenCL, but most GPU vendors do.

The implementation follows the official
[LiteRT C++ CompiledModel API](https://developers.google.com/edge/litert/next/cpp)
and bundles Google's
[prebuilt GPU accelerators](https://developers.google.com/edge/litert/next/cpp_sdk#prebuilt-gpu-accelerators).

## Community Discord

[Join the Margelo Community Discord](https://discord.gg/6CSHz2qAvA) to chat about React Native performance libraries.

## Adopting at scale

<a href="https://github.com/sponsors/mrousavy">
  <img align="right" width="160" alt="This library helped you? Consider sponsoring!" src=".github/funding-octocat.svg">
</a>

This library is provided _as is_, I work on it in my free time.

This project is based on the original
[`react-native-fast-tflite`](https://github.com/mrousavy/react-native-fast-tflite)
work by Marc Rousavy and contributors.

## Contributing

1. Clone the repo
2. Make sure you have installed Xcode CLI tools such as `gcc`, `cmake` and `python`/`python3`. See the TensorFlow documentation on what you need exactly.
3. Run `yarn bootstrap` and select `y` on all iOS and Android related questions.
4. Open the example app and start developing
   - iOS: `example/ios/TfliteExample.xcworkspace`
   - Android: `example/android`

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT
