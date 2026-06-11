// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

// Hypre backend tags and small type-level interface markers.
#include "gendil/Interfaces/Hypre/hypretypes.hpp"
// Error-checking helper for Hypre functions that return integer status codes.
#include "gendil/Interfaces/Hypre/hypreerror.hpp"
// Lightweight RAII guard for Hypre's global initialize/finalize lifecycle.
#include "gendil/Interfaces/Hypre/hypresession.hpp"
// Zero-copy `HYPRE_ParCSRMatrix` view creation for `HypreCSRMatrix`.
#include "gendil/Interfaces/Hypre/hypreparcsrview.hpp"
// Zero-copy `HYPRE_ParVector` views over compatible GenDiL vectors.
#include "gendil/Interfaces/Hypre/hyprevector.hpp"
// Hypre-backed ParCSR matrix-vector multiply entry points.
#include "gendil/Interfaces/Hypre/hyprematvec.hpp"

#endif // GENDIL_USE_HYPRE
