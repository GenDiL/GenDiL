// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "generalized-boundary-test-utils.hpp"

namespace observed_facet_vectors
{
using namespace gendil;
struct VectorRef
{
   Vector& value;
   mutable int reads = 0, writes = 0, read_writes = 0;
};
size_t GetVectorSize(const VectorRef& vector) { return vector.value.Size(); }
const Real* ReadHostVector(const VectorRef& vector)
{ ++vector.reads; return vector.value.ReadHostData(); }
Real* WriteHostVector(VectorRef& vector)
{ ++vector.writes; return vector.value.WriteHostData(); }
Real* ReadWriteHostVector(VectorRef& vector)
{ ++vector.read_writes; return vector.value.ReadWriteHostData(); }
const Real* ReadDeviceVector(const VectorRef& vector)
{ ++vector.reads; return vector.value.ReadDeviceData(); }
Real* WriteDeviceVector(VectorRef& vector)
{ ++vector.writes; return vector.value.WriteDeviceData(); }
Real* ReadWriteDeviceVector(VectorRef& vector)
{ ++vector.read_writes; return vector.value.ReadWriteDeviceData(); }
}

namespace interior_test
{
using namespace gendil;
using boundary_test::PointData;
using boundary_test::AddP1;
using boundary_test::Check;
using boundary_test::CheckLoad;

// Enumerate both current-cell visits to every interior facet independently of
// GenDiL topology and orientation helpers. Minus is the current cell here.
template<class Callback>
void InteriorPoints(size_t nx, size_t ny, Real hx, Real hy,
                    Real ox, Real oy, Callback callback)
{
   for (size_t cy = 0; cy < ny; ++cy)
   for (size_t cx = 0; cx < nx; ++cx)
   for (size_t axis = 0; axis < 2; ++axis)
   for (size_t side = 0; side < 2; ++side)
   {
      const size_t index = axis == 0 ? cx : cy;
      const size_t count = axis == 0 ? nx : ny;
      if ((side == 0 && index == 0) || (side == 1 && index+1 == count)) { continue; }
      const size_t adjacent = side == 0 ? index-1 : index+1;
      const size_t neighbor = axis == 0 ? adjacent+nx*cy : cx+nx*adjacent;
      for (size_t q = 0; q < boundary_test::points.size(); ++q)
      {
         const Real xi = axis == 0 ? Real(side) : boundary_test::points[q];
         const Real eta = axis == 1 ? Real(side) : boundary_test::points[q];
         callback(PointData{cx+nx*cy, xi, eta, ox+hx*(cx+xi), oy+hy*(cy+eta),
            axis == 0 ? 2*Real(side)-1 : 0,
            axis == 1 ? 2*Real(side)-1 : 0,
            boundary_test::weights[q]*(axis == 0 ? hy : hx)}, neighbor);
      }
   }
}

inline Real U(size_t cell, Real x, Real y)
{
   return 1 + 3*Real(cell) + (1+Real(cell))*x - 2*y;
}
inline Real P(size_t cell, Real x, Real y)
{
   return 2 - Real(cell) + x*x + 3*y;
}
inline Real F(size_t cell, Real x, Real y)
{
   return 4 + 2*Real(cell) + (2+Real(cell))*x - (1+Real(cell))*y;
}
}
