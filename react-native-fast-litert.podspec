require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-fast-litert"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "15.0" }
  s.module_name  = 'NitroTflite'
  s.source       = { :git => "https://github.com/pigeonmal/react-native-fast-litert.git", :tag => "#{s.version}" }

  s.source_files = [
    "ios/**/*.{swift}",
    "ios/**/*.{h,m,mm}",
    "cpp/**/*.{hpp,cpp,c,h}",
  ]

  s.vendored_frameworks = [
    "ios/LiteRT.xcframework",
    "ios/LiteRTMetalAccelerator.xcframework",
  ]
  s.preserve_paths = "third_party/litert/**/*"
  s.frameworks = "Foundation", "Metal"
  s.libraries = "c++"

  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' => "$(inherited) \"$(PODS_TARGET_SRCROOT)/third_party/litert/include\"",
    'CLANG_CXX_LANGUAGE_STANDARD' => "c++20",
  }

  load 'nitrogen/generated/ios/NitroTflite+autolinking.rb'
  add_nitrogen_files(s)

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  install_modules_dependencies(s)
end
