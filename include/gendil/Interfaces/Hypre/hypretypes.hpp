// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifndef GENDIL_USE_HYPRE
#error "gendil/Interfaces/Hypre headers require GENDIL_USE_HYPRE."
#endif

#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/prelude.hpp"

#include <HYPRE.h>
#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#include <HYPRE_utilities.h>

#include "_hypre_utilities.h"
#include "_hypre_seq_mv.h"
#include "_hypre_parcsr_mv.h"

namespace gendil
{

struct HypreCSRHostBackend : HostMatVecBackend
{ };

struct HypreCSRDeviceBackend : DeviceMatVecBackend
{ };

inline MPI_Comm HypreSelfComm()
{
   return hypre_MPI_COMM_SELF;
}

} // namespace gendil
