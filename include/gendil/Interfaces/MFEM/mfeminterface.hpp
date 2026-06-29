// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// Factories to build gendil unstructured meshes from MFEM meshes.
#include "meshfactories.hpp"

// Global face connectivity bundle types and builder declarations for MFEM meshes.
#include "GlobalFaceConnectivity/globalfaceconnectivity.hpp"

// Global-facet partition factory for MFEM meshes.
#include "globalpartition.hpp"

// An mfem::Vector wrapper to interface between gendil::Vector and mfem::Vector.
#include "vector.hpp"
