#!/usr/bin/env bash
export CUDA_PATH=/data00/Softwares/cuda-8.0
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

export PATH=/data00/UserHome/zwang/cuda-9.0${PATH:+:${PATH
}
}
export CPATH=/data00/UserHome/zwang/cuda-9.0/include${CPATH:+:${CPATH
}
}
export LD_LIBRARY_PATH=/data00/UserHome/zwang/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH
}
}
cd src/cuda

# compile dcn
nvcc -c -o dcn_v2_im2col_cuda.cu.o dcn_v2_im2col_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o dcn_v2_im2col_cuda_double.cu.o dcn_v2_im2col_cuda_double.cu -x cu -Xcompiler -fPIC

# compile dcn-roi-pooling
nvcc -c -o dcn_v2_psroi_pooling_cuda.cu.o dcn_v2_psroi_pooling_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o dcn_v2_psroi_pooling_cuda_double.cu.o dcn_v2_psroi_pooling_cuda_double.cu -x cu -Xcompiler -fPIC

cd -
python build.py
python build_double.py
