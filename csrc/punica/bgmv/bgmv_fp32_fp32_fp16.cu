#include "bgmv_config.h"
#include "bgmv_impl.cuh"

FOR_BGMV_ONESIDE(INST_BGMV, float, float, nv_half)

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, float, nv_half)