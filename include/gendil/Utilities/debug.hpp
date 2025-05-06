// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "types.hpp"

#ifdef GENDIL_USE_MFEM
#include <mfem.hpp>
#else
#include <iostream>
#include <assert.h>
#endif

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

} // namespace gendil
