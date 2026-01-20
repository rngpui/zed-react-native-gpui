use std::env;
use std::path::PathBuf;

fn main() {
    // Build the CXX bridge for Rust ↔ C++ FFI
    // We use cxx_build directly instead of craby_build::setup() to include
    // additional source files and custom include paths.

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let craby_modules_dir = manifest_dir.join("..").join("..");
    let rngpui_dir = manifest_dir
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .join("react-native-gpui");

    // Find React Native headers in node_modules
    let rn_headers = rngpui_dir
        .join("node_modules")
        .join("react-native")
        .join("ReactCommon");

    // RNGPUI native headers
    let rngpui_native = rngpui_dir.join("cpp").join("src");

    // Third-party dependencies (Folly, Boost, etc.) used by React Native bridging
    // These are bundled in React Native's Android JNI directory
    let third_party = rngpui_dir
        .join("node_modules")
        .join("react-native")
        .join("ReactAndroid")
        .join("src")
        .join("main")
        .join("jni")
        .join("third-party");

    // CXX-generated headers (cxx.h is in rust/ subdirectory)
    let cxx_include_dir = out_dir.join("cxxbridge").join("include");
    let cxx_rust_dir = cxx_include_dir.join("rust");
    let cxx_crate_dir = cxx_include_dir.join("zedmodules").join("src");

    // Step 1: Build CXX bridge (generates cxx.h and ffi.rs.h)
    cxx_build::bridge("src/ffi.rs")
        .std("c++20")
        .include("include")
        .compile("cxxbridge");

    // Generated C++ module implementations (Craby output)
    let craby_cpp_dir = craby_modules_dir.join("cpp");

    // Step 2: Build the zed_modules C++ code AFTER cxx headers are generated
    // This includes:
    // - zed_modules_shim.cpp: Registers Zed's TurboModules with RNGPUI
    // - CxxZed*Module.cpp: Generated module implementations from Craby
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        // Folly configuration - these flags tell folly to not use folly-config.h
        // and instead use these compile-time definitions (from RN's folly CMakeLists.txt)
        .define("FOLLY_NO_CONFIG", "1")
        .define("FOLLY_HAVE_CLOCK_GETTIME", "1")
        .define("FOLLY_USE_LIBCPP", "1")
        .define("FOLLY_CFG_NO_COROUTINES", "1")
        .define("FOLLY_MOBILE", "1")
        .define("FOLLY_HAVE_PTHREAD", "1")
        // Zed-specific shim for module registration
        .file(craby_modules_dir.join("..").join("native").join("cpp").join("zed_modules_shim.cpp"))
        // Generated Craby module implementations
        .file(craby_cpp_dir.join("CxxZedThemeModule.cpp"))
        .file(craby_cpp_dir.join("CxxZedLlmModule.cpp"))
        .file(craby_cpp_dir.join("CxxZedMarkdownModule.cpp"))
        .file(craby_cpp_dir.join("CxxZedWorkspaceModule.cpp"))
        .file(craby_cpp_dir.join("CxxZedIconsModule.cpp"))
        .include("include")
        .include(&cxx_rust_dir)                       // For cxx.h (direct include)
        .include(&cxx_include_dir)                    // For rust/cxx.h (relative include)
        .include(&cxx_crate_dir)                      // For ffi.rs.h
        .include(craby_modules_dir.join("desktop"))   // CrabyzedmodulesRegistration.hpp
        .include(&craby_cpp_dir)                      // CxxZedThemeModule.hpp, etc.
        .include(&rn_headers)                          // ReactCommon/TurboModule.h
        .include(rn_headers.join("react").join("nativemodule").join("core"))
        .include(rn_headers.join("jsi"))               // jsi/jsi.h
        .include(rn_headers.join("callinvoker"))       // ReactCommon/CallInvoker.h
        .include(&rngpui_native)
        // Third-party dependencies for React Native bridging
        .include(third_party.join("folly"))
        .include(third_party.join("boost"))
        .include(third_party.join("double-conversion"))
        .include(third_party.join("fmt").join("include"))
        .include(third_party.join("glog").join("exported"))
        .compile("zed_modules_shim");
}
