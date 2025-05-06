// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>
#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief A wrapper for mfem::Vector meant to represent degrees of freedom on a given finite element space.
 * 
 */
class FiniteElementVector : public mfem::Vector
{
public:
   template < typename FiniteElementSpace >
   explicit FiniteElementVector( const FiniteElementSpace & finite_element_space ) :
      Vector( finite_element_space.GetNumberOfFiniteElementDofs() )
   { UseDevice(true); }

   using mfem::Vector::operator=;
};

}

#endif
