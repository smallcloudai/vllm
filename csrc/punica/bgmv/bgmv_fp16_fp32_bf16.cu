#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_ONESIDE(INST_BGMV, nv_half, float, nv_bfloat16)

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, float, nv_bfloat16)