// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_CALIPER

#include <caliper/cali.h>

#define GENDIL_BENCHMARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define GENDIL_BENCHMARK_REGION_BEGIN( region_name ) CALI_MARK_BEGIN( region_name )
#define GENDIL_BENCHMARK_REGION_END( region_name ) CALI_MARK_END( region_name ) 

#else

#define GENDIL_BENCHMARK_FUNCTION
#define GENDIL_BENCHMARK_REGION_BEGIN( region_name )
#define GENDIL_BENCHMARK_REGION_END( region_name )

#endif
