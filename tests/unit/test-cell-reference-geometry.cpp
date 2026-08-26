// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <type_traits>

using namespace gendil;

namespace {

struct CountOnlyMesh
{
   GlobalIndex GetNumberOfCells() const
   {
      return 1;
   }

   GlobalIndex GetCell(GlobalIndex cell_index) const
   {
      return cell_index;
   }
};

using Product2D = ProductCell<SegmentCell, LineCell<3>>;
using NestedProduct3D = ProductCell<Product2D, SegmentCell>;

static_assert(std::is_same_v<SegmentCell::geometry, HyperCube<1>>);
static_assert(std::is_same_v<LineCell<3>::geometry, HyperCube<1>>);
static_assert(std::is_same_v<SquareCell::geometry, HyperCube<2>>);
static_assert(std::is_same_v<QuadCell<3>::geometry, HyperCube<2>>);
static_assert(std::is_same_v<CubeCell::geometry, HyperCube<3>>);
static_assert(std::is_same_v<HexCell<3>::geometry, HyperCube<3>>);
static_assert(
   std::is_same_v<HyperCubeCell<4>::geometry, HyperCube<4>>);
static_assert(std::is_same_v<Product2D::geometry, HyperCube<2>>);
static_assert(
   std::is_same_v<NestedProduct3D::geometry, HyperCube<3>>);

static_assert(mesh::Mesh<CountOnlyMesh>);
static_assert(!mesh::MeshWithCellGeometry<CountOnlyMesh>);
static_assert(mesh::MeshWithCellGeometry<Cartesian1DMesh>);
static_assert(
   std::is_same_v<
      mesh::mesh_geometry_t<Cartesian1DMesh>,
      HyperCube<1>>);

} // namespace

int main()
{
   Cartesian1DMesh mesh(0.5, 2);

   int interior_count = 0;
   int boundary_count = 0;
   FaceLoop(
      mesh,
      0,
      [&] (const auto&)
      {
         ++interior_count;
      },
      [&] (const auto&)
      {
         ++boundary_count;
      });

   int interior_only_count = 0;
   InteriorFaceLoop(
      mesh,
      0,
      [&] (const auto&)
      {
         ++interior_only_count;
      });

   int boundary_only_count = 0;
   BoundaryFaceLoop(
      mesh,
      0,
      [&] (const auto&)
      {
         ++boundary_only_count;
      });

   const bool counts_match =
      interior_count == 1 &&
      boundary_count == 1 &&
      interior_only_count == 1 &&
      boundary_only_count == 1;
   return counts_match ? 0 : 1;
}
