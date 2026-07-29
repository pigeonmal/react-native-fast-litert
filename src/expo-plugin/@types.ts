export type ConfigProps = {
  /**
   * @deprecated LiteRT 2.x uses its bundled Metal accelerator automatically.
   * This option is retained as a no-op so existing Expo configuration remains valid.
   */
  enableCoreMLDelegate?: boolean
  /**
   * Whether to enable the GPU acceleration delegate for GPU, by including related native libraries.
   * You can leave it as boolean for an array of native libraries.
   *
   * If enabled, "libOpenCL.so" will always be included.
   *
   * When ran prebuild, it will yield following result.
   *
   * ```xml
   * <uses-native-library android:name="libOpenCL.so" android:required="false"/>
   * ```
   *
   * @example
   * You can include more native libraries if needed.
   * ```json
   * [
   *   "react-native-fast-litert",
   *   {
   *     "enableAndroidGpuLibraries": ["libOpenCL-pixel.so", "libGLES_mali.so"]
   *   }
   * ]
   * ```
   *
   *
   * @default false
   */
  enableAndroidGpuLibraries?: boolean | string[]
} | void
