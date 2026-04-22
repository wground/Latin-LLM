#!/usr/bin/env python3
"""
System detection script for LatinLLM
Detects hardware capabilities, dependencies, and generates configuration for optimal training.
"""
import torch
import sys
import os
import re
import json
import importlib.util
import subprocess
import platform
from typing import Dict, List, Any, Optional


def _detect_apple_silicon_chip() -> Dict[str, Any]:
    """
    Detect specific Apple Silicon chip via sysctl. Returns dict with:
    - chip_name: e.g. "Apple M3 Pro" (as reported by sysctl), or None if not Apple Silicon
    - chip_family: "M1" | "M2" | "M3" | "M4" | "M5" | None
    - chip_variant: "" | "Pro" | "Max" | "Ultra"
    - gpu_cores: int, best-effort from system_profiler, or None
    - unified_memory: int bytes, or None
    """
    result = {"chip_name": None, "chip_family": None, "chip_variant": "",
              "gpu_cores": None, "unified_memory": None}

    if platform.system() != "Darwin":
        return result

    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        result["chip_name"] = brand

        # Parse family and variant (e.g. "Apple M3 Pro" -> family="M3", variant="Pro")
        match = re.match(r"Apple\s+(M\d+)(?:\s+(Pro|Max|Ultra))?", brand, re.IGNORECASE)
        if match:
            result["chip_family"] = match.group(1).upper()
            result["chip_variant"] = (match.group(2) or "").capitalize()
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        memsize = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        result["unified_memory"] = int(memsize)
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Try to get GPU core count from system_profiler (slower, best-effort)
    try:
        sp_out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        core_match = re.search(r"Total Number of Cores:\s*(\d+)", sp_out)
        if core_match:
            result["gpu_cores"] = int(core_match.group(1))
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result

def check_pytorch_installation() -> Dict[str, Any]:
    """Check PyTorch installation and capabilities."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": False,
        "gpu_count": 0,
        "gpu_devices": [],
        "backend_type": "cpu"  # default
    }
    
    # Check for Metal Performance Shaders (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["backend_type"] = "mps"
        info["gpu_count"] = 1  # Apple Silicon has integrated GPU

        try:
            chip_info = _detect_apple_silicon_chip()
            chip_name = chip_info.get("chip_name") or "Apple Silicon"

            device_info = {
                "id": 0,
                "name": f"{chip_name} GPU",
                "type": "integrated",
                "backend": "mps",
                "chip_family": chip_info.get("chip_family"),
                "chip_variant": chip_info.get("chip_variant"),
                "gpu_cores": chip_info.get("gpu_cores"),
            }

            # Use sysctl unified memory if available, else fall back to psutil
            if chip_info.get("unified_memory"):
                total_memory = chip_info["unified_memory"]
            else:
                try:
                    import psutil
                    total_memory = psutil.virtual_memory().total
                except ImportError:
                    total_memory = None

            if total_memory:
                device_info["unified_memory"] = total_memory
                # Apple Silicon unified memory: GPU can safely use ~75% for training
                # (rest needed for OS, CPU workloads, and system overhead)
                device_info["memory_estimated"] = int(total_memory * 0.75)
            else:
                device_info["memory_estimated"] = "unknown"

            info["gpu_devices"].append(device_info)
        except Exception as e:
            info["gpu_devices"].append({"id": 0, "error": str(e), "backend": "mps"})
    
    elif torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        
        # Detect if we're using CUDA or ROCm
        try:
            # Try to get CUDA version - this will fail on ROCm
            cuda_version = torch.version.cuda
            if cuda_version:
                info["backend_type"] = "cuda"
                info["cuda_version"] = cuda_version
        except:
            pass
            
        # Check for ROCm
        try:
            # ROCm uses HIP, check for HIP-related attributes
            if hasattr(torch.version, 'hip') and torch.version.hip:
                info["backend_type"] = "rocm"
                info["hip_version"] = torch.version.hip
        except:
            pass
            
        # If we still don't know, try to detect based on device names
        if info["backend_type"] == "cpu" and info["gpu_count"] > 0:
            try:
                device_name = torch.cuda.get_device_name(0).lower()
                if "radeon" in device_name or "amd" in device_name:
                    info["backend_type"] = "rocm"
                elif "nvidia" in device_name or "geforce" in device_name or "tesla" in device_name or "quadro" in device_name:
                    info["backend_type"] = "cuda"
            except:
                # Default to cuda if we have GPUs but can't determine type
                info["backend_type"] = "cuda"
        
        # Get GPU device information
        for i in range(info["gpu_count"]):
            try:
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_free": torch.cuda.memory_reserved(i),
                    "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                }
                info["gpu_devices"].append(device_info)
            except Exception as e:
                info["gpu_devices"].append({"id": i, "error": str(e)})
    
    return info

def check_mixed_precision_support() -> Dict[str, bool]:
    """Check support for different mixed precision types."""
    support = {
        "float16": True,  # Generally supported
        "bfloat16": False,
        "tf32": False
    }
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS supports fp16 universally. bf16 requires PyTorch 2.1+ and macOS 14+.
        # Raw fp16 without a GradScaler is unsafe for training (overflow → NaN),
        # so we probe bf16 at runtime and prefer it when available.
        support["float16"] = True
        support["tf32"] = False
        try:
            probe = torch.ones(4, 4, dtype=torch.bfloat16, device='mps')
            _ = (probe @ probe).float().sum().item()
            support["bfloat16"] = True
        except Exception:
            support["bfloat16"] = False
    elif torch.cuda.is_available():
        try:
            # Test bfloat16 support
            support["bfloat16"] = torch.cuda.is_bf16_supported()
        except:
            pass
            
        try:
            # TF32 is supported on Ampere and newer (compute capability >= 8.0)
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:
                support["tf32"] = True
        except:
            pass
    
    return support

def check_dependencies() -> Dict[str, Any]:
    """Check required and optional dependencies."""
    dependencies = {
        "required": {},
        "optional": {},
        "missing": []
    }
    
    # Required dependencies
    required_packages = [
        ("torch", "PyTorch deep learning framework"),
        ("numpy", "Numerical computing library")
    ]
    
    # Optional dependencies
    optional_packages = [
        ("tiktoken", "Tokenizer for GPT models"),
        ("wandb", "Weights & Biases for experiment tracking"),
        ("transformers", "Hugging Face transformers library")
    ]
    
    def check_package(package_name: str) -> Optional[str]:
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                module = importlib.import_module(package_name)
                return getattr(module, '__version__', 'unknown')
            return None
        except:
            return None
    
    # Check required packages
    for package, description in required_packages:
        version = check_package(package)
        if version:
            dependencies["required"][package] = {
                "version": version,
                "description": description,
                "status": "installed"
            }
        else:
            dependencies["required"][package] = {
                "description": description,
                "status": "missing"
            }
            dependencies["missing"].append(package)
    
    # Check optional packages
    for package, description in optional_packages:
        version = check_package(package)
        if version:
            dependencies["optional"][package] = {
                "version": version,
                "description": description,
                "status": "installed"
            }
        else:
            dependencies["optional"][package] = {
                "description": description,
                "status": "missing"
            }
    
    return dependencies

def test_gpu_operations() -> Dict[str, Any]:
    """Test basic GPU operations if available."""
    test_results = {
        "tensor_operations": False,
        "mixed_precision": False,
        "memory_allocation": False,
        "errors": []
    }
    
    # Check for MPS (Apple Silicon) first
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = 'mps'
            
            # Test basic tensor operations
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            _ = torch.matmul(x, y)
            test_results["tensor_operations"] = True
            
            # Test mixed precision (MPS supports float16)
            try:
                with torch.amp.autocast('cpu', dtype=torch.float16):  # MPS uses CPU autocast
                    _ = torch.matmul(x, y)
                test_results["mixed_precision"] = True
            except Exception:
                # MPS mixed precision might not be fully supported, but tensor ops work
                test_results["mixed_precision"] = False
            
            # Test memory allocation
            large_tensor = torch.zeros(1000, 1000, device=device)
            del large_tensor
            # Note: MPS doesn't have empty_cache() equivalent
            test_results["memory_allocation"] = True
            
        except Exception as e:
            test_results["errors"].append(f"MPS test failed: {str(e)}")
    
    elif torch.cuda.is_available():
        try:
            device = 'cuda:0'
            
            # Test basic tensor operations
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            _ = torch.matmul(x, y)
            test_results["tensor_operations"] = True
            
            # Test mixed precision
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _ = torch.matmul(x, y)
            test_results["mixed_precision"] = True
            
            # Test memory allocation
            large_tensor = torch.zeros(1000, 1000, device=device)
            del large_tensor
            torch.cuda.empty_cache()
            test_results["memory_allocation"] = True
            
        except Exception as e:
            test_results["errors"].append(f"CUDA test failed: {str(e)}")
    
    else:
        test_results["errors"].append("No GPU available")
    
    return test_results

def get_system_info() -> Dict[str, str]:
    """Get basic system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "machine": platform.machine()
    }

def generate_training_config(pytorch_info: Dict, mixed_precision: Dict, gpu_tests: Dict) -> Dict[str, Any]:
    """Generate optimal training configuration based on detected hardware."""
    config = {
        "device": "cpu",
        "dtype": "float32",
        "compile": False,
        "backend": "cpu",
        "multi_gpu": False,
        "recommended_batch_size": 4,
        "recommended_block_size": 256,
        "use_fused_adamw": False,
        "enable_tf32": False
    }
    
    # Windows: torch.compile / Triton has limited support — auto-disable unconditionally.
    is_windows = platform.system() == "Windows"
    config["os"] = platform.system().lower()

    # Check for Apple Silicon MPS first
    if pytorch_info.get("mps_available") and pytorch_info["backend_type"] == "mps" and gpu_tests["tensor_operations"]:
        config["device"] = "mps"
        config["backend"] = "mps"
        config["compile"] = False  # torch.compile still flaky on MPS as of PyTorch 2.9
        config["multi_gpu"] = False  # Apple Silicon is single integrated GPU

        # Record detected chip for downstream FLOPS lookup
        if pytorch_info["gpu_devices"]:
            dev = pytorch_info["gpu_devices"][0]
            config["apple_chip_family"] = dev.get("chip_family")
            config["apple_chip_variant"] = dev.get("chip_variant", "")
            config["apple_gpu_cores"] = dev.get("gpu_cores")

        # bf16 is strongly preferred on MPS — wider dynamic range, no GradScaler
        # needed, and raw fp16 training on MPS diverges because PyTorch's
        # GradScaler is CUDA-only. If bf16 isn't supported, use fp32 rather
        # than risking overflow.
        if mixed_precision["bfloat16"]:
            config["dtype"] = "bfloat16"
        else:
            config["dtype"] = "float32"

        # MPS doesn't support TF32 or fused AdamW
        config["enable_tf32"] = False
        config["use_fused_adamw"] = False

        # Batch/block tuning uses the 75%-of-unified-memory estimate.
        # M3+ benefits most from larger block_size; older chips cap earlier.
        if pytorch_info["gpu_devices"]:
            memory_info = pytorch_info["gpu_devices"][0]
            chip_family = memory_info.get("chip_family")
            modern_chip = chip_family in ("M3", "M4", "M5")

            if "memory_estimated" in memory_info and isinstance(memory_info["memory_estimated"], int):
                estimated_memory = memory_info["memory_estimated"]
                if estimated_memory > 48 * 1024**3:  # > 48GB (Max/Ultra tier)
                    config["recommended_batch_size"] = 24
                    config["recommended_block_size"] = 1024
                elif estimated_memory > 24 * 1024**3:  # > 24GB (Pro / high Max)
                    config["recommended_batch_size"] = 16
                    config["recommended_block_size"] = 1024 if modern_chip else 512
                elif estimated_memory > 12 * 1024**3:  # > 12GB (base Pro)
                    config["recommended_batch_size"] = 12
                    config["recommended_block_size"] = 512
                elif estimated_memory > 6 * 1024**3:  # > 6GB (base chips)
                    config["recommended_batch_size"] = 8
                    config["recommended_block_size"] = 512 if modern_chip else 256
                else:
                    config["recommended_batch_size"] = 4
                    config["recommended_block_size"] = 256
            else:
                config["recommended_batch_size"] = 6
                config["recommended_block_size"] = 512
    
    elif pytorch_info["cuda_available"] and gpu_tests["tensor_operations"]:
        config["device"] = "cuda"
        config["backend"] = pytorch_info["backend_type"]
        # torch.compile uses Triton, which has poor Windows support — disable there.
        config["compile"] = not is_windows

        if pytorch_info["gpu_count"] > 1:
            config["multi_gpu"] = True
        
        # Set optimal dtype based on hardware support
        if mixed_precision["bfloat16"]:
            config["dtype"] = "bfloat16"
        elif mixed_precision["float16"]:
            config["dtype"] = "float16"
        
        # Enable TF32 if supported
        if mixed_precision["tf32"]:
            config["enable_tf32"] = True
        
        # Adjust batch sizes based on GPU memory (optimized for larger vocab)
        if pytorch_info["gpu_devices"]:
            total_memory = pytorch_info["gpu_devices"][0].get("memory_total", 0)
            if total_memory > 32 * 1024**3:  # > 32GB (A100, RTX 6000 Ada)
                config["recommended_batch_size"] = 32
                config["recommended_block_size"] = 1024
            elif total_memory > 16 * 1024**3:  # > 16GB (RTX 4090, etc)
                config["recommended_batch_size"] = 24
                config["recommended_block_size"] = 1024
            elif total_memory > 12 * 1024**3:  # > 12GB (RTX 4080, etc)
                config["recommended_batch_size"] = 16
                config["recommended_block_size"] = 512
            elif total_memory > 8 * 1024**3:  # > 8GB
                config["recommended_batch_size"] = 12
                config["recommended_block_size"] = 512
            else:  # <= 8GB
                config["recommended_batch_size"] = 6
                config["recommended_block_size"] = 256
        
        config["use_fused_adamw"] = True
    
    return config

def main():
    """Main detection and configuration generation function."""
    print("=== LatinLLM System Detection ===\n")
    
    # Gather system information
    print("Gathering system information...")
    system_info = get_system_info()
    pytorch_info = check_pytorch_installation()
    mixed_precision = check_mixed_precision_support()
    dependencies = check_dependencies()
    gpu_tests = test_gpu_operations()
    
    # Generate training configuration
    training_config = generate_training_config(pytorch_info, mixed_precision, gpu_tests)
    
    # Create comprehensive report
    report = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "system_info": system_info,
        "pytorch": pytorch_info,
        "mixed_precision_support": mixed_precision,
        "dependencies": dependencies,
        "gpu_tests": gpu_tests,
        "recommended_config": training_config
    }
    
    # Print summary
    print(f"System: {system_info['platform']}")
    print(f"Python: {system_info['python_version']}")
    print(f"PyTorch: {pytorch_info['pytorch_version']}")
    
    if pytorch_info.get("mps_available"):
        print(f"MPS Available: {pytorch_info['mps_available']}")
        print(f"Backend: {pytorch_info['backend_type'].upper()}")
        print(f"GPU Count: {pytorch_info['gpu_count']} (Apple Silicon integrated)")
        for gpu in pytorch_info["gpu_devices"]:
            if "name" in gpu:
                if "memory_estimated" in gpu and isinstance(gpu["memory_estimated"], int):
                    memory_gb = gpu["memory_estimated"] / (1024**3)
                    print(f"  GPU {gpu['id']}: {gpu['name']} (Est. {memory_gb:.1f}GB shared)")
                else:
                    print(f"  GPU {gpu['id']}: {gpu['name']}")
    elif pytorch_info["cuda_available"]:
        print(f"CUDA Available: {pytorch_info['cuda_available']}")
        print(f"Backend: {pytorch_info['backend_type'].upper()}")
        print(f"GPU Count: {pytorch_info['gpu_count']}")
        for gpu in pytorch_info["gpu_devices"]:
            if "name" in gpu:
                memory_gb = gpu["memory_total"] / (1024**3)
                print(f"  GPU {gpu['id']}: {gpu['name']} ({memory_gb:.1f}GB)")
    else:
        print(f"GPU Available: False")
    
    if pytorch_info.get("mps_available") or pytorch_info["cuda_available"]:
        print(f"Mixed Precision Support:")
        print(f"  Float16: {mixed_precision['float16']}")
        print(f"  BFloat16: {mixed_precision['bfloat16']}")
        print(f"  TF32: {mixed_precision['tf32']}")
    
    # Check for missing dependencies
    if dependencies["missing"]:
        print(f"\n⚠️  Missing required dependencies: {', '.join(dependencies['missing'])}")
        return 1
    
    # Save configuration BEFORE printing results (to prevent Unicode crashes on Windows)
    config_path = "latin_training_config.json"
    with open(config_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Show GPU test results (use ASCII fallbacks for Windows compatibility)
    def safe_check(val):
        try:
            return '\u2705' if val else '\u274c'
        except (UnicodeEncodeError, UnicodeDecodeError):
            return '[OK]' if val else '[FAIL]'

    if pytorch_info.get("mps_available") or pytorch_info["cuda_available"]:
        print(f"\nGPU Tests:")
        print(f"  Tensor Operations: {safe_check(gpu_tests['tensor_operations'])}")
        print(f"  Mixed Precision: {safe_check(gpu_tests['mixed_precision'])}")
        print(f"  Memory Allocation: {safe_check(gpu_tests['memory_allocation'])}")

        if gpu_tests["errors"]:
            print(f"  Errors: {', '.join(gpu_tests['errors'])}")

    print(f"\nSystem detection complete!")
    print(f"Full report saved to: {config_path}")
    print(f"\nRecommended training configuration:")
    print(f"  Device: {training_config['device']}")
    print(f"  Backend: {training_config['backend']}")
    print(f"  Data Type: {training_config['dtype']}")
    print(f"  Batch Size: {training_config['recommended_batch_size']}")
    print(f"  Block Size: {training_config['recommended_block_size']}")
    print(f"  Compile Model: {training_config['compile']}")

    return 0 if not dependencies["missing"] and (not (pytorch_info["cuda_available"] or pytorch_info.get("mps_available")) or gpu_tests["tensor_operations"]) else 1

if __name__ == "__main__":
    sys.exit(main())