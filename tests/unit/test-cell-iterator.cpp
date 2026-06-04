// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <iostream>

using namespace gendil;

namespace
{
struct TinyMesh
{
   GlobalIndex num_cells;

   GlobalIndex GetNumberOfCells() const
   {
      return num_cells;
   }

   GlobalIndex GetCell( const GlobalIndex cell_index ) const
   {
      return cell_index;
   }
};

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}
} // namespace

int main()
{
   constexpr GlobalIndex num_cells = 7;
   TinyMesh mesh{ num_cells };

   std::array< int, num_cells > visits{};
   mesh::CellIterator<HostKernelConfiguration>(
      mesh,
      [&] ( const GlobalIndex cell_index )
      {
         visits[ cell_index ] += 1;
      } );

   bool success = true;
   for ( GlobalIndex i = 0; i < num_cells; ++i )
   {
      success = Check(
         visits[ i ] == 1,
         "CellIterator one-index body did not visit each cell once." ) &&
         success;
   }

   return success ? 0 : 1;
}
