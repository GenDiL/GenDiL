// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Compile each cell header independently, before other library headers can
// supply the oriented-view declarations used by ApplyOrientationToCell.
#if GENDIL_TEST_CELL_DIM == 2
#include <gendil/Meshes/Cells/ReferenceCells/quadcell.hpp>
#elif GENDIL_TEST_CELL_DIM == 3
#include <gendil/Meshes/Cells/ReferenceCells/hexcell.hpp>
#else
#error "GENDIL_TEST_CELL_DIM must be 2 or 3"
#endif

namespace oriented_cell_header_test
{

using namespace gendil;

GENDIL_HOST_DEVICE
bool CheckOrientation(const Real seed, const bool alternate)
{
   constexpr Integer Dim = GENDIL_TEST_CELL_DIM;
   constexpr Integer D1D = 3;
   constexpr Integer Dofs = Dim == 2 ? D1D * D1D : D1D * D1D * D1D;
   Real nodes[Dim * Dofs];
   int restriction[Dofs];
   for (Integer node = 0; node < Dofs; ++node)
   {
      restriction[node] = static_cast<int>(node);
      for (Integer component = 0; component < Dim; ++component)
         nodes[component + Dim * node] = seed + 100.0 * component + node;
   }
   const auto node_view = MakeFIFOView(static_cast<const Real*>(nodes), Dim, Dofs);
   HostDevicePointer<const int> indices;
   indices.host_pointer = restriction;
#ifdef GENDIL_USE_DEVICE
   // Both execution paths use their own stack-local restriction array.
   indices.device_pointer = restriction;
#endif
#if GENDIL_TEST_CELL_DIM == 2
   QuadCell<D1D> cell(node_view, MakeFIFOView(indices, D1D, D1D, Integer{1}), 0);
   const Permutation<Dim> orientation = alternate
      ? Permutation<Dim>{{-2, 1}} : Permutation<Dim>{{2, -1}};
#else
   HexCell<D1D> cell(node_view, MakeFIFOView(indices, D1D, D1D, D1D, Integer{1}), 0);
   const Permutation<Dim> orientation = alternate
      ? Permutation<Dim>{{3, -1, 2}} : Permutation<Dim>{{-2, 3, -1}};
#endif
   ApplyOrientationToCell(orientation, cell);

   // Independently scatter each native index into its signed canonical axes.
   for (Integer node = 0; node < Dofs; ++node)
   {
      std::array<Integer, Dim> canonical{};
      Integer remaining = node;
      for (Integer axis = 0; axis < Dim; ++axis)
      {
         const Integer native = remaining % D1D;
         remaining /= D1D;
         const int mapped = orientation(axis);
         const Integer destination = mapped > 0 ? mapped - 1 : -mapped - 1;
         canonical[destination] = mapped > 0 ? native : D1D - 1 - native;
      }
      for (Integer component = 0; component < Dim; ++component)
      {
#if GENDIL_TEST_CELL_DIM == 2
         const Real actual = cell.nodes[canonical[0]][canonical[1]][component];
#else
         const Real actual = cell.nodes[canonical[0]][canonical[1]][canonical[2]][component];
#endif
         if (actual != seed + 100.0 * component + node) return false;
      }
   }
   return true;
}

} // namespace oriented_cell_header_test

#include <gendil/Algebra/Vector/vector.hpp>
#include <gendil/Utilities/KernelContext/KernelConfigurations/kernelconfigurations.hpp>

int main()
{
   using namespace gendil;
   constexpr Integer count = 5;
   for (Integer i = 0; i < count; ++i)
      if (!oriented_cell_header_test::CheckOrientation(0.25 * (i + 1), i % 2))
         return 1;

#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   Vector results(count);
   Real* values = results.WriteDeviceData();
   Policy::CandidateBlockLoop(count, [=] GENDIL_DEVICE ()
   {
      const auto i = Policy::WorkItemIndex();
      if (Policy::IsActive(count))
         values[i] = oriented_cell_header_test::CheckOrientation(0.25 * (i + 1), i % 2);
   });
   GENDIL_DEVICE_SYNC;
   const Real* host_values = results.ReadHostData();
   for (Integer i = 0; i < count; ++i)
      if (host_values[i] != 1.0) return 1;
#endif
   return 0;
}
