use std::path::PathBuf;

fn main() {
    // ./target/debug/build/ocl_vkfft-.../out/
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-search={}", out_path.to_str().unwrap());
    println!("cargo:rustc-link-lib=static=foo");
    println!("cargo:rerun-if-changed=src/test.c");

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("vkFFT/vkFFT.h")
        .clang_arg("-IvkFFT")
        .clang_arg("-DVKFFT_BACKEND=3")
        .clang_arg("-DCL_TARGET_OPENCL_VERSION=300")
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_path.join("foo.c"))
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_recursively(true)
        .allowlist_type("VkFFTConfiguration")
        .allowlist_type("VkFFTApplication")
        .allowlist_type("VkFFTLaunchParams")
        .allowlist_type("VkFFTResult")
        .blocklist_type("cl_platform_id")
        .blocklist_type("_cl_platform_id")
        .blocklist_type("cl_device_id")
        .blocklist_type("_cl_device_id")
        .blocklist_type("cl_command_queue")
        .blocklist_type("_cl_command_queue")
        .blocklist_type("cl_mem")
        .blocklist_type("_cl_mem")
        .blocklist_type("cl_context")
        .blocklist_type("_cl_context")
        .blocklist_type("cl_program")
        .blocklist_type("_cl_program")
        .blocklist_type("cl_kernel")
        .blocklist_type("_cl_kernel")
        .allowlist_function("initializeVkFFT")
        .allowlist_function("VkFFTAppend")
        .allowlist_function("deleteVkFFT")
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Compile the generated wrappers into an object file.
    let clang_output = std::process::Command::new("clang")
        .arg("-IvkFFT")
        .arg("-I.")
        .arg("-O3") // Change to "-g" for debugging 
        .arg("-DVKFFT_BACKEND=3")
        .arg("-DCL_TARGET_OPENCL_VERSION=300")
        .arg("-o")
        .arg(out_path.join("foo.o"))
        .arg("-c")
        .arg(out_path.join("foo.c"))
        .output()
        .unwrap();
        
    if !clang_output.status.success() {
        panic!(
            "Could not compile object file:\n{}",
            String::from_utf8_lossy(&clang_output.stderr)
        );
    }
    let clang_output2 = std::process::Command::new("clang")
        .arg("-IvkFFT")
        .arg("-DVKFFT_BACKEND=3")
        .arg("-DCL_TARGET_OPENCL_VERSION=300")
        .arg("-o")
        .arg(out_path.join("test.o"))
        .arg("-c")
        .arg("src/test.c")
        .output()
        .unwrap();
        
    if !clang_output2.status.success() {
        panic!(
            "Could not compile object file:\n{}",
            String::from_utf8_lossy(&clang_output2.stderr)
        );
    }

    
    let ar_output = std::process::Command::new("ar")
        .arg("rcs")
        .arg(out_path.join("libfoo.a"))
        .arg(out_path.join("foo.o"))
        .arg(out_path.join("test.o"))
        .output()
        .unwrap();

    if !ar_output.status.success() {
        panic!(
            "Could not emit library file:\n{}",
            String::from_utf8_lossy(&ar_output.stderr)
        );
    }

    //cc::Build::new()
    //    .file("src/foo.c")
    //    .include(Path::new("vkFFT"))
    //    .flag("-DVKFFT_BACKEND=3")
    //    .flag("-DCL_TARGET_OPENCL_VERSION=300")
    //    .flag("-fkeep-inline-functions")
    //    .flag("-lOpenCL")
    //    .flag("-lm")
    //    .static_flag(true)
    //    .cargo_metadata(true)
    //    .compile("foo");
}
