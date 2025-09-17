// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace std;
using namespace gendil;

bool approximately_equal(Real a, Real b, Real tol = 1e-12)
{
   return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b));
}

bool test_read_write_consistency()
{
   constexpr Integer order = 2;
   constexpr Integer num_quad_1d = order + 2;

   const Real h_space = 1.0;
   Cartesian3DMesh mesh( h_space, 2, 2, 2 );

   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule( num_quads );

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 3;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d, num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto read_operator = MakeFaceSpeedOfLightOperator<KernelPolicy>( fe_space, int_rules );
   auto write_operator = MakeWriteFaceSpeedOfLightOperator<KernelPolicy>( fe_space, int_rules );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   Vector dofs_in( num_dofs );
   Vector dofs_read( num_dofs );
   Vector dofs_write( num_dofs );
   dofs_read = 0.0;
   dofs_write = 0.0;

   // Set a pattern to dofs_in
   auto dofs_in_ptr = dofs_in.WriteHostData();
   for (Integer i = 0; i < num_dofs; ++i)
      dofs_in_ptr[i] = std::sin(i);

   read_operator(dofs_in, dofs_read);
   write_operator(dofs_in, dofs_write);

   bool success = true;
   auto read_ptr = dofs_read.ReadHostData();
   auto write_ptr = dofs_write.ReadHostData();
   for (Integer i = 0; i < num_dofs; ++i)
   {
      if (!approximately_equal(read_ptr[i], write_ptr[i]))
      {
         std::cout << "Mismatch at i=" << i << ": "
                   << read_ptr[i] << " != " << write_ptr[i] << "\n";
         success = false;
      }
   }

   if (success)
      std::cout << "Read/Write Speed-of-Light operators are consistent!\n";
   else
      std::cout << "Discrepancy detected in Read/Write Speed-of-Light operators!\n";

   return success;
}

int main(int argc, char *argv[])
{
   return test_read_write_consistency() ? 0 : 1;
}
