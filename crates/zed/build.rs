#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=10.15.7");

        // Weakly link ReplayKit to ensure Zed can be used on macOS 10.15+.
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,ReplayKit");

        // Seems to be required to enable Swift concurrency
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

        // Register exported Objective-C selectors, protocols, etc
        println!("cargo:rustc-link-arg=-Wl,-ObjC");

        // weak link to support Catalina
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,ScreenCaptureKit");
    }

    link_rngpui_for_rn_chat_demo();

    // Populate git sha environment variable if git is available
    println!("cargo:rerun-if-changed=../../.git/logs/HEAD");
    println!(
        "cargo:rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );
    if let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output()
        && output.status.success()
    {
        let git_sha = String::from_utf8_lossy(&output.stdout);
        let git_sha = git_sha.trim();

        println!("cargo:rustc-env=ZED_COMMIT_SHA={git_sha}");

        if let Some(build_identifier) = option_env!("GITHUB_RUN_NUMBER") {
            println!("cargo:rustc-env=ZED_BUILD_ID={build_identifier}");
        }

        if let Ok(build_profile) = std::env::var("PROFILE")
            && build_profile == "release"
        {
            // This is currently the best way to make `cargo build ...`'s build script
            // to print something to stdout without extra verbosity.
            println!("cargo::warning=Info: using '{git_sha}' hash for ZED_COMMIT_SHA env var");
        }
    }

    #[cfg(target_os = "windows")]
    {
        #[cfg(target_env = "msvc")]
        {
            // todo(windows): This is to avoid stack overflow. Remove it when solved.
            println!("cargo:rustc-link-arg=/stack:{}", 8 * 1024 * 1024);
        }

        if cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64") {
            let out_dir = std::env::var("OUT_DIR").unwrap();
            let out_dir: &std::path::Path = out_dir.as_ref();
            let target_dir = std::path::Path::new(&out_dir)
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .expect("Failed to find target directory");

            let conpty_dll_target = target_dir.join("conpty.dll");
            let open_console_target = target_dir.join("OpenConsole.exe");

            let conpty_url = "https://github.com/microsoft/terminal/releases/download/v1.23.13503.0/Microsoft.Windows.Console.ConPTY.1.23.251216003.nupkg";
            let nupkg_path = out_dir.join("conpty.nupkg.zip");
            let extract_dir = out_dir.join("conpty");

            let download_script = format!(
                "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                conpty_url,
                nupkg_path.display()
            );

            let download_result = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    &download_script,
                ])
                .output();

            match download_result {
                Ok(output) if output.status.success() => {
                    println!("Downloaded conpty nupkg successfully");

                    let extract_script = format!(
                        "$ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                        nupkg_path.display(),
                        extract_dir.display()
                    );

                    let extract_result = Command::new("powershell")
                        .args(["-NoProfile", "-NonInteractive", "-Command", &extract_script])
                        .output();

                    match extract_result {
                        Ok(output) if output.status.success() => {
                            let (conpty_dll_source, open_console_source) =
                                if cfg!(target_arch = "x86_64") {
                                    (
                                        extract_dir.join("runtimes/win-x64/native/conpty.dll"),
                                        extract_dir
                                            .join("build/native/runtimes/x64/OpenConsole.exe"),
                                    )
                                } else {
                                    (
                                        extract_dir.join("runtimes/win-arm64/native/conpty.dll"),
                                        extract_dir
                                            .join("build/native/runtimes/arm64/OpenConsole.exe"),
                                    )
                                };

                            match std::fs::copy(&conpty_dll_source, &conpty_dll_target) {
                                Ok(_) => {
                                    println!("Copied conpty.dll to {}", conpty_dll_target.display())
                                }
                                Err(e) => println!(
                                    "cargo::warning=Failed to copy conpty.dll from {}: {}",
                                    conpty_dll_source.display(),
                                    e
                                ),
                            }

                            match std::fs::copy(&open_console_source, &open_console_target) {
                                Ok(_) => println!(
                                    "Copied OpenConsole.exe to {}",
                                    open_console_target.display()
                                ),
                                Err(e) => println!(
                                    "cargo::warning=Failed to copy OpenConsole.exe from {}: {}",
                                    open_console_source.display(),
                                    e
                                ),
                            }
                        }
                        Ok(output) => {
                            println!(
                                "cargo::warning=Failed to extract conpty nupkg: {}",
                                String::from_utf8_lossy(&output.stderr)
                            );
                        }
                        Err(e) => {
                            println!(
                                "cargo::warning=Failed to run PowerShell for extraction: {}",
                                e
                            );
                        }
                    }
                }
                Ok(output) => {
                    println!(
                        "cargo::warning=Failed to download conpty nupkg: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                Err(e) => {
                    println!(
                        "cargo::warning=Failed to run PowerShell for download: {}",
                        e
                    );
                }
            }
        }

        let release_channel = option_env!("RELEASE_CHANNEL").unwrap_or("dev");
        let icon = match release_channel {
            "stable" => "resources/windows/app-icon.ico",
            "preview" => "resources/windows/app-icon-preview.ico",
            "nightly" => "resources/windows/app-icon-nightly.ico",
            "dev" => "resources/windows/app-icon-dev.ico",
            _ => "resources/windows/app-icon-dev.ico",
        };
        let icon = std::path::Path::new(icon);

        println!("cargo:rerun-if-env-changed=RELEASE_CHANNEL");
        println!("cargo:rerun-if-changed={}", icon.display());

        let mut res = winresource::WindowsResource::new();

        // Depending on the security applied to the computer, winresource might fail
        // fetching the RC path. Therefore, we add a way to explicitly specify the
        // toolkit path, allowing winresource to use a valid RC path.
        if let Some(explicit_rc_toolkit_path) = std::env::var("ZED_RC_TOOLKIT_PATH").ok() {
            res.set_toolkit_path(explicit_rc_toolkit_path.as_str());
        }
        res.set_icon(icon.to_str().unwrap());
        res.set("FileDescription", "Zed");
        res.set("ProductName", "Zed");

        if let Err(e) = res.compile() {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "rn-chat-demo")]
fn link_rngpui_for_rn_chat_demo() {
    use rngpui_build::{BuildConfig, build, emit_link_directives, existing_build_output};

    // crates/zed -> crates -> zed -> rngpui -> react-native-gpui
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let rngpui_root = manifest_dir
        .join("..")
        .join("..")
        .join("..")
        .join("react-native-gpui");

    // Cargo won't automatically re-run this build script when out-of-tree RNGPUI
    // sources change, so explicitly track the key C++/CMake inputs.
    println!(
        "cargo:rerun-if-changed={}",
        rngpui_root
            .join("native/cpp/scripts/generate_view_config_patches_from_codegen.py")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        rngpui_root.join("native/cpp/cmake/HostExecutable.cmake").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        rngpui_root.join("native/cpp/src/fabric/ViewConfig.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        rngpui_root
            .join("native/cpp/src/fabric/ViewConfigRegistry.cpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        rngpui_root
            .join("native/cpp/src/fabric/ViewConfigRegistry.h")
            .display()
    );

    let config = BuildConfig {
        native_sources_dir: rngpui_root.join("native"),
        // Point the C++ build at the RN Chat Panel JS workspace so optional
        // integrations (like react-native-svg) are detected from the correct
        // node_modules tree.
        app_dir: Some(manifest_dir.join("..").join("rn_chat_panel").join("js")),
        cmake_build_name: "zed".to_string(),
        features: rngpui_build::BuildFeatures {
            rnsvg: true,
            reanimated: true,
            worklets: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let output = match build(config.clone()) {
        Ok(output) => output,
        Err(e) => {
            println!("cargo:warning=rn-chat-demo: RNGPUI build failed: {}", e);
            match existing_build_output(&config) {
                Ok(output) => {
                    println!(
                        "cargo:warning=rn-chat-demo: using existing RNGPUI build at {}",
                        output.build_dir.display()
                    );
                    output
                }
                Err(err) => {
                    panic!(
                        "rn-chat-demo: RNGPUI build failed and no existing build output found: {}",
                        err
                    );
                }
            }
        }
    };

    if !output.bundled_lib.is_file() {
        panic!(
            "rn-chat-demo: missing RNGPUI bundled library at {}",
            output.bundled_lib.display()
        );
    }

    #[cfg(target_os = "macos")]
    if output.jsi_lib.is_none() {
        let expected = output.build_dir.join("react/jsi/libjsi.dylib");
        panic!(
            "rn-chat-demo: missing React Native JSI dylib at {}",
            expected.display()
        );
    }

    // Use force_load on macOS to avoid link-order issues
    #[cfg(target_os = "macos")]
    {
        println!(
            "cargo:rustc-link-arg=-Wl,-force_load,{}",
            output.bundled_lib.display()
        );
    }

    emit_link_directives(&output);

    // Link the Craby-generated gpui_modules archives *after* the C++ runtime library.
    // The RNGPUI C++ staticlib (`librngpui_bundled.a`) references symbols defined in these
    // archives, and static link order matters on macOS.
    link_craby_modules();
}

/// Link the gpui_modules staticlib (Craby-generated TurboModule FFI).
/// The staticlib provides Rust implementations called from C++ via CXX bridge.
#[cfg(feature = "rn-chat-demo")]
fn link_craby_modules() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(&out_dir);

    // Get the profile name (debug/release)
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    // Navigate from OUT_DIR to the profile directory:
    // OUT_DIR is typically: target/{profile}/build/{crate}-{hash}/out
    let profile_dir = out_path
        .ancestors()
        .find(|p| {
            p.file_name()
                .is_some_and(|n| n == "debug" || n == "release")
        })
        .map(|p| p.to_path_buf())
        .expect("Could not find profile directory (debug/release)");

    let deps_dir = profile_dir.join("deps");

    // Also check craby-modules workspace target directory
    // crates/zed -> crates -> zed -> rngpui -> react-native-gpui -> craby-modules
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let craby_modules_target = manifest_dir
        .join("..")
        .join("..")
        .join("..")
        .join("react-native-gpui")
        .join("craby-modules")
        .join("target")
        .join(&profile);
    let craby_modules_deps = craby_modules_target.join("deps");

    // Collect all directories to search
    let mut search_dirs: Vec<PathBuf> = vec![profile_dir.clone(), deps_dir.clone()];

    // Add craby-modules directories if they exist
    // Prefer artifacts built in Zed's target dir first; only fall back to the
    // react-native-gpui/craby-modules target dir if we can't find anything.
    let mut fallback_dirs: Vec<PathBuf> = Vec::new();
    if craby_modules_target.exists() {
        fallback_dirs.push(craby_modules_target.clone());
    }
    if craby_modules_deps.exists() {
        fallback_dirs.push(craby_modules_deps.clone());
    }

    // Also link the CXX bridge C++ glue archive produced by gpui_modules' build script.
    // It defines the `craby::gpuimodules::bridging::*` shims and `rust::cxxbridge1::*`
    // specializations used by the generated C++ code.
    let gpui_modules_build_dir = profile_dir.join("build");
    if let Some(cxxbridge_path) = newest_matching_file(
        &gpui_modules_build_dir,
        "gpui_modules-",
        &["out", "libcxxbridge.a"],
    ) {
        println!(
            "cargo:warning=Linking Craby modules C++ bridge archive: {}",
            cxxbridge_path.display()
        );
        println!("cargo:rustc-link-arg={}", cxxbridge_path.display());
    } else {
        println!(
            "cargo:warning=Could not find gpui_modules libcxxbridge.a under {}",
            gpui_modules_build_dir.display()
        );
    }

    // Then link the gpuimodules Rust static archive. Use an exact path so it lands at the end of
    // the final link line (after librngpui_bundled.a).
    let mut newest_gpuimodules: Option<(std::time::SystemTime, PathBuf)> = None;
    for search_dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(search_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with("libgpuimodules") || !name_str.ends_with(".a") {
                    continue;
                }

                let path = entry.path();
                let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
                    continue;
                };
                match &newest_gpuimodules {
                    Some((best_time, _)) if modified <= *best_time => {}
                    _ => newest_gpuimodules = Some((modified, path)),
                }
            }
        }
    }

    if newest_gpuimodules.is_none() {
        for search_dir in &fallback_dirs {
            // Prefer the exact name without hash for fallback builds that run in that workspace.
            let exact_path = search_dir.join("libgpuimodules.a");
            if let Ok(modified) = exact_path.metadata().and_then(|m| m.modified()) {
                newest_gpuimodules = Some((modified, exact_path));
                continue;
            }

            if let Ok(entries) = std::fs::read_dir(search_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if !name_str.starts_with("libgpuimodules") || !name_str.ends_with(".a") {
                        continue;
                    }

                    let path = entry.path();
                    let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
                        continue;
                    };
                    match &newest_gpuimodules {
                        Some((best_time, _)) if modified <= *best_time => {}
                        _ => newest_gpuimodules = Some((modified, path)),
                    }
                }
            }
        }
    }

    if let Some((_, path)) = newest_gpuimodules {
        println!("cargo:warning=Linking Craby modules archive: {}", path.display());
        println!("cargo:rustc-link-arg={}", path.display());
        return;
    }

    // If not found, emit a detailed warning
    println!("cargo:warning=Could not find libgpuimodules staticlib");
    println!("cargo:warning=  Searched directories:");
    for dir in &search_dirs {
        println!("cargo:warning=    - {}", dir.display());
    }
    println!("cargo:warning=  OUT_DIR: {}", out_dir);

    // List what's actually in the deps directory for debugging
    if let Ok(entries) = std::fs::read_dir(&deps_dir) {
        let libs: Vec<_> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".a") {
                    Some(name_str.to_string())
                } else {
                    None
                }
            })
            .take(10)
            .collect();
        println!("cargo:warning=  .a files in Zed deps (first 10): {:?}", libs);
    }
}

#[cfg(not(feature = "rn-chat-demo"))]
fn link_rngpui_for_rn_chat_demo() {
    // No-op when feature is disabled
}

#[cfg(feature = "rn-chat-demo")]
fn newest_matching_file(
    dir: &PathBuf,
    prefix: &str,
    suffix_components: &[&str],
) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(prefix) {
            continue;
        }

        let mut path = entry.path();
        for component in suffix_components {
            path = path.join(component);
        }
        if !path.is_file() {
            continue;
        }

        let Ok(modified) = path.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        match &newest {
            Some((best_time, _)) if modified <= *best_time => {}
            _ => newest = Some((modified, path)),
        }
    }

    newest.map(|(_, p)| p)
}
