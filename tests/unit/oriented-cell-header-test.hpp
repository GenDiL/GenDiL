// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Each entry point includes its cell header and instantiates its orientation
// function before including this shared host/device test harness.
#include <type_traits>
#include <gendil/Algebra/Vector/vector.hpp>
#include <gendil/Utilities/KernelContext/KernelConfigurations/kernelconfigurations.hpp>

namespace oriented_cell_header_test
{

using namespace gendil;

template<class Cell>
GENDIL_HOST_DEVICE
bool CheckOrientation(const Real seed, const bool alternate)
{
   constexpr Integer Dim = Cell::Dim;
   constexpr Integer D1D = std::extent_v<decltype(Cell::nodes)>;
   static_assert(Dim >= 1 && Dim <= 3);
   constexpr Integer Dofs = Dim == 1 ? D1D :
      (Dim == 2 ? D1D * D1D : D1D * D1D * D1D);
   Real nodes[Dim * Dofs];
   int restriction[Dofs];
   for (Integer node = 0; node < Dofs; ++node)
   {
      restriction[node] = static_cast<int>(node);
      for (Integer component = 0; component < Dim; ++component)
         nodes[component + Dim * node] = seed + 100.0 * component + node;
   }
   HostDevicePointer<const int> indices;
   indices.host_pointer = restriction;
#ifdef GENDIL_USE_DEVICE
   // Both execution paths use their own stack-local restriction array.
   indices.device_pointer = restriction;
#endif
   Cell cell = [&]()
   {
      if constexpr (Dim == 1)
         return Cell(MakeFIFOView(static_cast<const Real*>(nodes), Dofs),
                     MakeFIFOView(indices, D1D, Integer{1}), 0);
      else
      {
         const auto node_view = MakeFIFOView(static_cast<const Real*>(nodes), Dim, Dofs);
         if constexpr (Dim == 2)
            return Cell(node_view, MakeFIFOView(indices, D1D, D1D, Integer{1}), 0);
         else
            return Cell(node_view, MakeFIFOView(indices, D1D, D1D, D1D, Integer{1}), 0);
      }
   }();
   Permutation<Dim> orientation;
   if constexpr (Dim == 1)
      orientation = alternate ? Permutation<Dim>{{-1}} : Permutation<Dim>{{1}};
   else if constexpr (Dim == 2)
      orientation = alternate ? Permutation<Dim>{{-2, 1}} : Permutation<Dim>{{2, -1}};
   else
      orientation = alternate ? Permutation<Dim>{{3, -1, 2}} : Permutation<Dim>{{-2, 3, -1}};
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
         Real actual;
         if constexpr (Dim == 1)
            actual = cell.nodes[canonical[0]];
         else if constexpr (Dim == 2)
            actual = cell.nodes[canonical[0]][canonical[1]][component];
         else
            actual = cell.nodes[canonical[0]][canonical[1]][canonical[2]][component];
         if (actual != seed + 100.0 * component + node) return false;
      }
   }
   return true;
}

template<class Cell>
int Run()
{
   constexpr Integer count = 5;
   for (Integer i = 0; i < count; ++i)
      if (!CheckOrientation<Cell>(0.25 * (i + 1), i % 2))
         return 1;

#ifdef GENDIL_USE_DEVICE
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   Vector results(count);
   Real* values = results.WriteDeviceData();
   Policy::CandidateBlockLoop(count, [=] GENDIL_DEVICE ()
   {
      const auto i = Policy::WorkItemIndex();
      if (Policy::IsActive(count))
         values[i] = CheckOrientation<Cell>(0.25 * (i + 1), i % 2);
   });
   GENDIL_DEVICE_SYNC;
   const Real* host_values = results.ReadHostData();
   for (Integer i = 0; i < count; ++i)
      if (host_values[i] != 1.0) return 1;
#endif
   return 0;
}

} // namespace oriented_cell_header_test
