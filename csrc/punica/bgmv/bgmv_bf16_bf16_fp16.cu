#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_ONESIDE(INST_BGMV, nv_bfloat16, nv_bfloat16, nv_half)

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_half)
