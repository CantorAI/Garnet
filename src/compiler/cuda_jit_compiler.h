#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <vector>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <mutex>
#include <chrono>

// CUDA Runtime & Driver API
#include <cuda.h>
#include <cuda_runtime.h>
// NVRTC for runtime compilation
#include <nvrtc.h>

namespace Garnet {

    class CudaJitCompiler {
    private:
        // Cache for loaded modules
        struct ModuleCache {
            CUmodule module = nullptr;
            std::string ptx_code;
        };

        std::string base_dir_;
        std::string jit_dir_;
        std::vector<std::string> include_dirs_;
        std::vector<std::string> cu_dependencies_;
        std::vector<std::string> nvrtc_options_;
        std::vector<std::pair<std::string, std::string>> header_files_;
        std::unordered_map<std::string, ModuleCache> module_cache_;
        std::mutex cache_mutex_;
        bool initialized_ = false;
        bool verbose_ = false;

        std::string get_cuda_include_dir() {
#if defined(WIN32) || defined(_WIN32)
            // On Windows, use the CUDA_PATH environment variable.
            const char* cudaPath = std::getenv("CUDA_PATH");
            if (cudaPath) {
                std::filesystem::path includePath = std::filesystem::path(cudaPath) / "include";
                if (std::filesystem::exists(includePath)) {
                    return includePath.string();
                }
                else {
                    std::cerr << "CUDA include directory not found at: " << includePath.string() << std::endl;
                }
            }
            else {
                std::cerr << "CUDA_PATH environment variable not set on Windows." << std::endl;
            }
            // Fallback for Windows (update the path as needed)
            return "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include";
#else
            // On non-Windows systems (Linux/macOS), use the CUDA_HOME environment variable.
            const char* cudaPath = std::getenv("CUDA_HOME");
            if (cudaPath) {
                std::filesystem::path includePath = std::filesystem::path(cudaPath) / "include";
                if (std::filesystem::exists(includePath)) {
                    return includePath.string();
                }
                else {
                    std::cerr << "CUDA include directory not found at: " << includePath.string() << std::endl;
                }
            }
            else {
                std::cerr << "CUDA_HOME environment variable not set on non-Windows system." << std::endl;
            }
            // Fallback for non-Windows (update the path as needed)
            return "/usr/local/cuda/include";
#endif
        }

        // Check CUDA driver API errors
        void check_cuda_error(CUresult result, const char* call) const {
            if (result != CUDA_SUCCESS) {
                const char* error_string;
                cuGetErrorString(result, &error_string);
                throw std::runtime_error(std::string(call) + " failed with error: " + error_string);
            }
        }

        // Check NVRTC errors
        void check_nvrtc_error(nvrtcResult result, const char* call) const {
            if (result != NVRTC_SUCCESS) {
                throw std::runtime_error(std::string(call) + " failed with error: " + nvrtcGetErrorString(result));
            }
        }

        // Initialize CUDA context if not already done
        void ensure_cuda_initialized() {
            if (!initialized_) {
                check_cuda_error(cuInit(0), "cuInit");

                CUdevice device;
                check_cuda_error(cuDeviceGet(&device, 0), "cuDeviceGet");

                CUcontext context;
                check_cuda_error(cuCtxCreate(&context, 0, device), "cuCtxCreate");

                initialized_ = true;
            }
        }

        // Get the PTX file path for a given module name
        std::string get_ptx_path(const std::string& module_name) const {
            return (std::filesystem::path(jit_dir_) / (module_name + ".ptx")).string();
        }
        // Get the hash file path for a given module name
        std::string get_hash_path(const std::string& module_name) const {
            return (std::filesystem::path(jit_dir_) / (module_name + ".md5")).string();
        }

        // Check if the PTX file and hash file exist for a given module name
        bool ptx_file_exists(const std::string& module_name) const {
            return std::filesystem::exists(get_ptx_path(module_name));
        }

        bool hash_file_exists(const std::string& module_name) const {
            return std::filesystem::exists(get_hash_path(module_name));
        }

        // Read the stored hash from file for a given module name
        std::string read_hash_from_file(const std::string& module_name) const {
            std::string hash_path = get_hash_path(module_name);
            if (!std::filesystem::exists(hash_path)) {
                return "";
            }

            try {
                std::ifstream file(hash_path);
                if (!file.is_open()) {
                    return "";
                }

                std::string hash;
                std::getline(file, hash);
                return hash;
            }
            catch (const std::exception& e) {
                std::cerr << "Error reading hash file: " << e.what() << std::endl;
                return "";
            }
        }

        // Write hash to file for a given module name
        bool write_hash_to_file(const std::string& module_name, const std::string& hash) const {
            try {
                std::string hash_path = get_hash_path(module_name);
                std::ofstream file(hash_path);
                if (!file.is_open()) {
                    std::cerr << "Failed to open hash file for writing: " << hash_path << std::endl;
                    return false;
                }

                file << hash;
                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "Error writing hash file: " << e.what() << std::endl;
                return false;
            }
        }

        // Load PTX from file for a given module name
        bool load_ptx_from_file(const std::string& module_name, ModuleCache& cache) {
            try {
                std::string ptx_path = get_ptx_path(module_name);
                if (verbose_) {
                    std::cout << "Loading PTX from file: " << ptx_path << std::endl;
                }

                std::ifstream file(ptx_path, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                // Read the file content into a string
                file.seekg(0, std::ios::end);
                size_t size = file.tellg();
                file.seekg(0, std::ios::beg);

                cache.ptx_code.resize(size);
                file.read(&cache.ptx_code[0], size);

                // Load the PTX code
                if (cache.module) {
                    cuModuleUnload(cache.module);
                    cache.module = nullptr;
                }

                check_cuda_error(
                    cuModuleLoadData(&cache.module, cache.ptx_code.c_str()),
                    "cuModuleLoadData"
                );

                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading PTX from file: " << e.what() << std::endl;
                return false;
            }
        }

        // Save PTX to file for a given module name
        bool save_ptx_to_file(const std::string& module_name, const std::string& ptx_code) const {
            try {
                std::string ptx_path = get_ptx_path(module_name);
                if (verbose_) {
                    std::cout << "Saving PTX to file: " << ptx_path << std::endl;
                }

                std::ofstream file(ptx_path, std::ios::binary);
                if (!file.is_open()) {
                    std::cerr << "Failed to open file for writing: " << ptx_path << std::endl;
                    return false;
                }

                file.write(ptx_code.c_str(), ptx_code.size());
                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "Error saving PTX to file: " << e.what() << std::endl;
                return false;
            }
        }

        // Create the JIT directory if it doesn't exist
        void ensure_jit_directory() {
            if (!std::filesystem::exists(jit_dir_)) {
                if (verbose_) {
                    std::cout << "Creating JIT directory: " << jit_dir_ << std::endl;
                }
                std::filesystem::create_directories(jit_dir_);
            }
        }

        // Compile CUDA code to PTX using NVRTC
        bool compile_cuda_to_ptx(const std::string& cuda_code,
            const std::string& module_name,
            std::string& ptx_code) {
            try {
                if (verbose_) {
                    std::cout << "Compiling CUDA code for module '" << module_name << "' to PTX..." << std::endl;
                }

                auto start = std::chrono::high_resolution_clock::now();

                // Create NVRTC program
                nvrtcProgram prog;
                check_nvrtc_error(
                    nvrtcCreateProgram(&prog,
                        cuda_code.c_str(),
                        (module_name + ".cu").c_str(),
                        0, nullptr, nullptr),
                    "nvrtcCreateProgram"
                );

                // Add stored header files
                for (const auto& [name, content] : header_files_) {
                    if (verbose_) {
                        std::cout << "Adding header: " << name << std::endl;
                    }
                    check_nvrtc_error(
                        nvrtcAddNameExpression(prog, name.c_str()),
                        "nvrtcAddNameExpression"
                    );
                }

                // Prepare compiler options
                std::vector<std::string> options = nvrtc_options_;

                // Add include directories
                for (const auto& dir : include_dirs_) {
                    options.push_back("-I" + dir);
                }

                // Add dependency file paths directly
                for (const auto& dependency : cu_dependencies_) {
                    if (std::filesystem::exists(dependency)) {
                        if (verbose_) {
                            std::cout << "Adding dependency: " << dependency << std::endl;
                        }
                        // Extract the file name without directory
                        std::string filename = std::filesystem::path(dependency).filename().string();
                        options.push_back("--include-path=" + std::filesystem::path(dependency).parent_path().string());
                    }
                }

                // Always generate relocatable PTX code
                options.push_back("--relocatable-device-code=true");
                options.push_back("-default-device");

                // Convert options vector to char* array
                std::vector<const char*> option_ptrs;
                for (const auto& opt : options) {
                    option_ptrs.push_back(opt.c_str());
                }

                // Print NVRTC options if verbose
                if (verbose_) {
                    std::cout << "NVRTC options for module '" << module_name << "':" << std::endl;
                    for (const auto& opt : options) {
                        std::cout << "  " << opt << std::endl;
                    }
                }

                // Compile the CUDA code to PTX
                nvrtcResult compile_result = nvrtcCompileProgram(
                    prog, option_ptrs.size(), option_ptrs.data()
                );

                // Get compilation log regardless of success
                size_t log_size;
                check_nvrtc_error(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");

                std::string log(log_size, '\0');
                check_nvrtc_error(nvrtcGetProgramLog(prog, &log[0]), "nvrtcGetProgramLog");

                // Write the compile log to file in the jit directory (.log)
                {
                    std::string log_path = (std::filesystem::path(jit_dir_) / (module_name + ".log")).string();
                    std::ofstream log_file(log_path, std::ios::binary);
                    if (log_file.is_open()) {
                        log_file << log;
                        log_file.close();
                        if (verbose_) {
                            std::cout << "Saved NVRTC compile log to file: " << log_path << std::endl;
                        }
                    }
                    else {
                        std::cerr << "Failed to write NVRTC compile log to file: " << log_path << std::endl;
                    }
                }

                if (compile_result != NVRTC_SUCCESS) {
                    std::cerr << "NVRTC Compilation failed for module '" << module_name << "':\n" << log << std::endl;
                    return false;
                }
                else if (verbose_ && !log.empty()) {
                    std::cout << "NVRTC Compilation log for module '" << module_name << "':\n" << log << std::endl;
                }

                // Get the PTX code
                size_t ptx_size;
                check_nvrtc_error(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");

                ptx_code.resize(ptx_size);
                check_nvrtc_error(nvrtcGetPTX(prog, &ptx_code[0]), "nvrtcGetPTX");

                // Destroy the program
                check_nvrtc_error(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                if (verbose_) {
                    std::cout << "Compilation of module '" << module_name << "' completed in " << duration << " ms" << std::endl;
                }

                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "Error during CUDA compilation of module '" << module_name << "': " << e.what() << std::endl;
                return false;
            }
        }

        // Load the PTX code into a CUDA module
        bool load_ptx_to_module(const std::string& ptx_code, CUmodule& module) {
            try {
                if (module) {
                    cuModuleUnload(module);
                    module = nullptr;
                }

                check_cuda_error(
                    cuModuleLoadData(&module, ptx_code.c_str()),
                    "cuModuleLoadData"
                );

                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading PTX to CUDA module: " << e.what() << std::endl;
                return false;
            }
        }

    public:
        // Constructor with base directory
        CudaJitCompiler() {}
        void Init(const std::string& base_dir)
        {
            base_dir_ = base_dir;
            // Ensure base directory exists
            if (!std::filesystem::exists(base_dir_)) {
                throw std::runtime_error("Base directory does not exist: " + base_dir_);
            }

            // Setup paths
            std::filesystem::path jit_path = std::filesystem::path(base_dir_) / "_jit_";
            jit_dir_ = jit_path.string();

            // Automatically add the CUDA include directory
            std::string cudaInclude = get_cuda_include_dir();
            add_include_directory(cudaInclude);

            std::filesystem::path cuda_lib_path = std::filesystem::path(base_dir_) / "Lib/Garnet/cuda";
            std::string lib_include = cuda_lib_path.string();
            add_include_directory(lib_include);

            // Initialize CUDA
            ensure_cuda_initialized();

            // Ensure JIT directory exists
            ensure_jit_directory();
        }
        // Destructor
        ~CudaJitCompiler() {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            for (auto& [name, cache] : module_cache_) {
                if (cache.module) {
                    cuModuleUnload(cache.module);
                    cache.module = nullptr;
                }
            }
        }

        // Enable verbose mode
        void set_verbose(bool verbose) {
            verbose_ = verbose;
        }

        // Add include directory
        void add_include_directory(const std::string& dir) {
            if (std::filesystem::exists(dir)) {
                include_dirs_.push_back(dir);
            }
            else {
                std::cerr << "Warning: Include directory does not exist: " << dir << std::endl;
            }
        }

        // Add a CU file dependency
        void add_cu_dependency(const std::string& cu_file) {
            if (std::filesystem::exists(cu_file)) {
                cu_dependencies_.push_back(cu_file);
            }
            else {
                std::cerr << "Warning: CU dependency file does not exist: " << cu_file << std::endl;
            }
        }

        // Add compiler option
        void add_option(const std::string& option) {
            nvrtc_options_.push_back(option);
        }

        // Add predefined macro
        void add_define(const std::string& name, const std::string& value = "") {
            if (value.empty()) {
                nvrtc_options_.push_back("-D" + name);
            }
            else {
                nvrtc_options_.push_back("-D" + name + "=" + value);
            }
        }

        // Add a header file
        void add_header_file(const std::string& name, const std::string& content) {
            // Check if header with this name already exists
            for (auto it = header_files_.begin(); it != header_files_.end(); ++it) {
                if (it->first == name) {
                    // Replace existing header
                    *it = std::make_pair(name, content);
                    if (verbose_) {
                        std::cout << "Replaced existing header file: " << name << std::endl;
                    }
                    return;
                }
            }

            // Add new header
            header_files_.emplace_back(name, content);
            if (verbose_) {
                std::cout << "Added header file: " << name << std::endl;
            }
        }
        // Check if a module exists and has a matching hash - thread-safe
        bool check_module_hash(const std::string& module_name, const std::string& md5_hash) {
            // Lock for thread safety
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // First, check if the module files exist
            if (!ptx_file_exists(module_name) || !hash_file_exists(module_name)) {
                return false;
            }
            // Read the stored hash and compare
            std::string stored_hash = read_hash_from_file(module_name);
            // Return true if hashes match
            return (stored_hash == md5_hash);
        }
        // Compile CUDA code or load from cached PTX - thread-safe
        bool compile_or_load(const std::string& cuda_code,
            const std::string& module_name,
            const std::string& md5_hash) {
            ensure_cuda_initialized();

            // Always write out the CUDA source code to a file (.cu) in the JIT directory
            {
                std::string cu_path = (std::filesystem::path(jit_dir_) / (module_name + ".cu")).string();
                std::ofstream cu_file(cu_path, std::ios::binary);
                if (cu_file.is_open()) {
                    cu_file << cuda_code;
                    cu_file.close();
                    if (verbose_) {
                        std::cout << "Saved CUDA source to file: " << cu_path << std::endl;
                    }
                }
                else {
                    std::cerr << "Failed to write CUDA source to file: " << cu_path << std::endl;
                }
            }

            // Lock for thread safety
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Check if this module is already in memory cache
            auto cache_it = module_cache_.find(module_name);
            bool in_memory = (cache_it != module_cache_.end());

            if (verbose_) {
                std::cout << "Module '" << module_name << "' in-memory cache "
                    << (in_memory ? "hit" : "miss") << std::endl;
            }

            // Check if PTX file already exists
            if (ptx_file_exists(module_name) && hash_file_exists(module_name)) {
                if (verbose_) {
                    std::cout << "Found existing PTX file for module '" << module_name << "'" << std::endl;
                }

                // Read the stored hash and compare
                std::string stored_hash = read_hash_from_file(module_name);

                if (verbose_) {
                    std::cout << "Stored hash: " << stored_hash << std::endl;
                    std::cout << "Current hash: " << md5_hash << std::endl;
                }

                // If hashes match, we can use the cached PTX
                if (stored_hash == md5_hash) {
                    if (verbose_) {
                        std::cout << "Hashes match for module '" << module_name << "', using cached PTX." << std::endl;
                    }

                    // If not in memory, load from file
                    if (!in_memory) {
                        ModuleCache new_cache;
                        if (load_ptx_from_file(module_name, new_cache)) {
                            module_cache_[module_name] = std::move(new_cache);
                            return true;
                        }

                        // If loading failed, fall back to recompilation
                        if (verbose_) {
                            std::cout << "Failed to load existing PTX for '" << module_name << "', recompiling..." << std::endl;
                        }
                    }
                    else {
                        // Already in memory cache
                        return true;
                    }
                }
                else {
                    if (verbose_) {
                        std::cout << "Hashes don't match for module '" << module_name << "', recompiling..." << std::endl;
                    }
                }
            }
            else {
                if (verbose_) {
                    if (!ptx_file_exists(module_name)) {
                        std::cout << "PTX file not found for module '" << module_name << "', compiling..." << std::endl;
                    }
                    else {
                        std::cout << "Hash file not found for module '" << module_name << "', recompiling..." << std::endl;
                    }
                }
            }

            // Compile CUDA code to PTX
            ModuleCache new_cache;
            if (!compile_cuda_to_ptx(cuda_code, module_name, new_cache.ptx_code)) {
                return false;
            }

            // Save PTX and hash to files for future use
            save_ptx_to_file(module_name, new_cache.ptx_code);
            write_hash_to_file(module_name, md5_hash);

            // Load the PTX code into a CUDA module
            if (!load_ptx_to_module(new_cache.ptx_code, new_cache.module)) {
                return false;
            }

            // Store in cache
            module_cache_[module_name] = std::move(new_cache);
            return true;
        }

        // Force recompilation regardless of existing PTX - thread-safe
        bool force_recompile(const std::string& cuda_code,
            const std::string& module_name,
            const std::string& md5_hash) {
            ensure_cuda_initialized();

            // Lock for thread safety
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Compile CUDA code to PTX
            ModuleCache new_cache;
            if (!compile_cuda_to_ptx(cuda_code, module_name, new_cache.ptx_code)) {
                return false;
            }

            // Save PTX and hash to files for future use
            save_ptx_to_file(module_name, new_cache.ptx_code);
            write_hash_to_file(module_name, md5_hash);

            // Load the PTX code into a CUDA module
            if (!load_ptx_to_module(new_cache.ptx_code, new_cache.module)) {
                return false;
            }

            // Update cache
            module_cache_[module_name] = std::move(new_cache);
            return true;
        }

        // Get the PTX code for a specific module
        std::string get_ptx_code(const std::string& module_name) const {
            std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache_mutex_));
            auto it = module_cache_.find(module_name);
            if (it != module_cache_.end()) {
                return it->second.ptx_code;
            }
            return "";
        }

        // Get a kernel function from a specific module
        CUfunction get_kernel(const std::string& module_name, const std::string& kernel_name) {
            // Lock for thread safety
            std::lock_guard<std::mutex> lock(cache_mutex_);

            // Try to find the module in the cache
            auto it = module_cache_.find(module_name);
            if (it == module_cache_.end() || !it->second.module) {
                // Module not in memory or not loaded, try loading from the PTX file.
                if (ptx_file_exists(module_name)) {
                    ModuleCache new_cache;
                    if (load_ptx_from_file(module_name, new_cache)) {
                        // Cache the loaded module and update the iterator.
                        module_cache_[module_name] = std::move(new_cache);
                        it = module_cache_.find(module_name);
                    }
                    else {
                        throw std::runtime_error("Failed to load PTX for module '" + module_name + "'");
                    }
                }
                else {
                    throw std::runtime_error("No compiled CUDA module '" + module_name + "' loaded, and no PTX file found.");
                }
            }

            // Retrieve the kernel from the loaded module.
            CUfunction kernel;
            check_cuda_error(
                cuModuleGetFunction(&kernel, it->second.module, kernel_name.c_str()),
                "cuModuleGetFunction"
            );
            return kernel;
        }


        // Launch a kernel with the specified grid and block dimensions
        void launch_kernel(CUfunction kernel,
            dim3 grid_dim, dim3 block_dim,
            void** args, size_t shared_mem = 0,
            CUstream stream = nullptr) const {
            check_cuda_error(
                cuLaunchKernel(kernel,
                    grid_dim.x, grid_dim.y, grid_dim.z,
                    block_dim.x, block_dim.y, block_dim.z,
                    shared_mem, stream, args, nullptr),
                "cuLaunchKernel"
            );
        }
    };

} // namespace Garnet
