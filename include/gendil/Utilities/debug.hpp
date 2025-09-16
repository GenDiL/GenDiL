// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "types.hpp"

#include <iostream>
#include <assert.h>
#include <cstdlib>

namespace gendil
{

GENDIL_HOST_DEVICE GENDIL_INLINE
void Assert(bool should_be_true, const char msg[] = "")
{
    // FIXME: this debug check doesn't seem to actually work.
    #ifndef NDEBUG
    if (not should_be_true)
    {
        printf("---------- gendil error ----------\n%s\n----------------------------------\n", msg);
        assert(should_be_true);
    }
    #endif
}

inline void Assert(bool condition, const char* condition_str,
                   const char* file, int line, const char* msg = "")
{
   if (!condition)
   {
      std::cerr << "Assertion failed: (" << condition_str << ")"
                << " in file " << file << ", line " << line;
      if (msg && *msg)
      {
         std::cerr << ": " << msg;
      }
      std::cerr << std::endl;
      std::abort();
   }
}

#ifdef NDEBUG
#define GENDIL_ASSERT(cond, ...)
#else
#define GENDIL_ASSERT(cond, ...) Assert((cond), #cond, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define GENDIL_VERIFY(cond, ...) Assert((cond), #cond, __FILE__, __LINE__, ##__VA_ARGS__)

} // namespace gendil
