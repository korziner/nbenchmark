# nbenchmark
bench Pascal gen GPUs +reveal broken Tensor Cores / FMA on NVIDIA CMP mining cards

```
GPU: NVIDIA P102-100 (SM 6.1)

=== FP32 explicit MUL+ADD, no FMA ===
N: 67108864  Reps: 16384  Checksum: 4432853991424.000
Time: 1.559487 s  Perf: 5.640 TFLOP/s

=== FP16 f16x2 explicit MUL+ADD, no FMA ===
N: 67108864  Reps: 16384  Checksum: 0x1C001C000000000
Time: 94.100346 s  Perf: 0.187 TFLOP/s

=== INT8 dp4a ===
N: 67108864  Reps: 16384  Checksum: 17592186044416
Time: 0.844726 s  Perf: 41.652 TOP/s

--- FMA comparison ---

=== FP32 FMA ===
N: 67108864  Reps: 16384  Checksum: 4432853467136.000
Time: 0.834646 s  Perf: 10.539 TFLOP/s

=== FP16 f16x2 FMA ===
N: 67108864  Reps: 16384  Checksum: 0x1C001C000000000
Time: 130.585421 s  Perf: 0.135 TFLOP/s

=== INT8 dp4a ===
N: 67108864  Reps: 16384  Checksum: 17592186044416
Time: 1.510173 s  Perf: 23.298 TOP/s

--- cuBLAS Tensor Core probe ---

==========================================================================
  cuBLAS GEMM Tensor Core Probe  (2048x2048x2048, 50 iters)
  Timing: CUDA events (accurate GPU time, not CPU wall-clock)
  Correctness: C = A*B where A=B=ones, expect C[0]==2048
==========================================================================
Mode                                           Time(s)   TFLOP/s Check
--------------------------------------------------------------------------
FP16 | 16F DEFAULT                              8.3240      0.10  BAD!
FP16 | 16F TENSOR_OP                            8.4530      0.10  BAD!
FP16 | 16F_PEDANTIC (no TC/FMA)                 7.5251      0.11  BAD! *
FP16 | 16F ALGO_1 (CUDA cores)                  8.7967      0.10  BAD!
FP16 | 16F ALGO_2                               7.5790      0.11  BAD!
FP16 | 16F ALGO_3                               9.2164      0.09  BAD!
FP16 | 32F_PEDANTIC                             0.2330      3.69    OK *
FP16 | 32F DEFAULT (uses FMA)                   0.1897      4.53    OK
FP16 | 32F_FAST_16F                             0.2008      4.28    OK
FP16 | 32F_FAST_TF32                            0.2050      4.19    OK
FP32 | 32F_PEDANTIC                             0.1299      6.61    OK *
FP32 | 32F DEFAULT                              0.1249      6.88    OK
FP32 | 32F ALGO_5                               0.2333      3.68    OK
--------------------------------------------------------------------------
Best correct: FP32 | 32F DEFAULT (6.88 TFLOP/s)
```

Compared to P104:
```
cat P104.log
GPU: NVIDIA P104-100 (SM 6.1)

=== FP32 explicit MUL+ADD, no FMA ===
N: 67108864  Reps: 16384  Checksum: 4432853991424.000
Time: 2.613210 s  Perf: 3.366 TFLOP/s

=== FP16 f16x2 explicit MUL+ADD, no FMA ===
N: 67108864  Reps: 16384  Checksum: 0x1C001C000000000
Time: 212.367001 s  Perf: 0.083 TFLOP/s

=== INT8 dp4a ===
N: 67108864  Reps: 16384  Checksum: 17592186044416
Time: 2.220874 s  Perf: 15.843 TOP/s

--- FMA comparison ---
...
```

```
claude-opus-4-6-thinking_nbenchmark --help
Benchmark that tests raw PTX kernels AND cuBLAS GEMM with various compute
modes to reveal broken Tensor Cores / FMA on NVIDIA CMP mining cards.

KNOWN RESULTS on CMP 50HX (Turing, SM 7.5):
  FP32 explicit mul+add  ~5.9 TFLOP/s    <-- use this for FP32 on CMP
  FP32 FMA               ~0.42 TFLOP/s   <-- 14x SLOWER, broken!
  FP16 f16x2 mul+add     ~22.3 TFLOP/s   <-- best for inference
  INT8 dp4a              ~1.7 TOP/s

EXAMPLES:

  nbenchmark all --n 134217728 --reps 16384
      Full PTX benchmark with large workload

  nbenchmark fp32 --reps 16384
  nbenchmark fp32-fma --reps 16384
      Compare FMA vs non-FMA (reveals CMP breakage)

  nbenchmark cublas-probe
      KEY TEST: probes cuBLAS GEMM with many compute modes.
      Uses CUDA events for accurate GPU timing.
      Checks correctness of results.
      Reveals whether Tensor Cores produce wrong results.

  nbenchmark cublas-probe --m 4096 --gemm-iters 50
      Larger matrix, more iterations for stable results

  nbenchmark full-report --n 134217728
      Everything: PTX + FMA comparison + cuBLAS probe

HOW TO USE RESULTS FOR nanochatGPT PRETRAIN:

  Rust:
    // If cublas-probe shows PEDANTIC fastest or DEFAULT produces BAD results:
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    cublasGemmEx(..., CUBLAS_COMPUTE_16F_PEDANTIC, CUBLAS_GEMM_DEFAULT);

  Julia (CUDA.jl):
    CUBLAS.cublasSetMathMode(handle, CUBLAS.CUBLAS_PEDANTIC_MATH)


Usage: claude-opus-4-6-thinking_nbenchmark [OPTIONS] <COMMAND>

Commands:
  fp32          FP32 explicit mul+add (no FMA) -- best for CMP
  fp32-fma      FP32 FMA -- usually broken on CMP
  f16x2         FP16 f16x2 explicit mul+add
  f16x2-fma     FP16 f16x2 FMA
  int8          INT8 dp4a (requires SM 6.1+)
  all           Non-FMA PTX: FP32 + FP16 + INT8
  all-fma       FMA PTX: FP32-FMA + FP16-FMA + INT8
  cublas-probe  KEY TEST: probe cuBLAS GEMM modes, check Tensor Cores
  full-report   Full report: all PTX + cuBLAS probe
  help          Print this message or the help of the given subcommand(s)

Options:
      --n <N>
          Number of elements for PTX benchmarks
          
          [default: 67108864]

      --reps <REPS>
          Arithmetic reps inside each GPU thread
          
          [default: 16384]

      --m <M>
          GEMM matrix dimension M=N=K (for cublas-probe)
          
          [default: 2048]

      --gemm-iters <GEMM_ITERS>
          GEMM iterations for timing (for cublas-probe)
          
          [default: 50]

  -h, --help
  ```
          Print help (see a summary with '-h')

  -V, --version
