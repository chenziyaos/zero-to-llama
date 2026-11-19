#!/bin/bash
cd "$(dirname "$0")"
bash exp06.sh native
bash exp06.sh generic -DGGML_NATIVE=OFF
bash exp06.sh sse42 -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF
bash exp06.sh blas -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
bash exp06.sh nollamafile -DGGML_LLAMAFILE=OFF
bash exp06.sh dl -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
echo ALLDONE
