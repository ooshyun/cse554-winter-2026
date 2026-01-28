# CSE554 HW1
FYI. I used my other resources, so it includes gpu configuration also including RTX6000.

## Resources
- Following questions (.pdf): CSE554_Assignment1_Conan_Oh.pdf
- Profiling results: profiling_result
- Code
.
├── common
│   ├── gpu_specs.h
│   └── gpu_specs.py
├── common.mk
├── host_GPU
│   ├── CMakeLists.txt
│   ├── copy_first_column.cu
│   ├── copy_first_column.h
│   ├── main.cu
│   ├── Makefile
│   └── memory_transfer.cu
├── profiling_result
│   ├── bandwidth_data.csv
│   ├── bandwidth_plot_combined.pdf
│   ├── bandwidth_plot_combined.png
│   ├── log.log
│   ├── profiling_results_host_GPU
│   ├── profiling_results_rms_norm_matrix
│   ├── profiling_results_rms_norm_vector
│   └── profiling_results_silu_py
│       └── cuda_silu.qdstrm
├── requirements.txt
├── rms_norm
│   ├── matrix
│   │   ├── CMakeLists.txt
│   │   ├── main.cu
│   │   ├── Makefile
│   │   ├── profiling_results
│   │   ├── rms_norm_matrix.cu
│   │   └── rms_norm_matrix.h
│   └── vector
│       ├── CMakeLists.txt
│       ├── main.cu
│       ├── Makefile
│       ├── rms_norm_vector.cu
│       ├── rms_norm_vector.h
│       ├── test_simple.cu
│       └── test_w2l3.cu
├── run_all_tests.sh
├── scripts
│   ├── plot_bandwidth.py
│   ├── run_section_1.sh
│   ├── run_section_2.sh
│   └── run_section_3.sh
└── silu
    ├── CUDA
    │   ├── CMakeLists.txt
    │   ├── main.cu
    │   ├── Makefile
    │   ├── silu.cu
    │   └── silu.h
    ├── gpu_config.py
    ├── silu_torch.py
    ├── silu_triton_kernel.py
    └── silu_triton_test.py


## Run the code
```bash
uv venv
uv pip install -r requirements.txt
./run_all_tests.sh
```