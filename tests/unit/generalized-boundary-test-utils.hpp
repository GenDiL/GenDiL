// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <gendil/gendil.hpp>
#include <array>
#include <cmath>
#include <iostream>

namespace boundary_test
{
using namespace gendil;

// Independent four-point Gauss rule and explicit bilinear shape functions.
// The reference does not call GenDiL's basis, geometry, or restriction kernels.
constexpr std::array<Real, 4> points{
   0.0694318442029737124, 0.330009478207571868,
   0.669990521792428132, 0.930568155797026288};
constexpr std::array<Real, 4> weights{
   0.173927422568726929, 0.326072577431273071,
   0.326072577431273071, 0.173927422568726929};

struct PointData
{
   size_t cell;
   Real xi, eta, x, y, nx, ny, weight;
};

template<class Callback>
void BoundaryPoints(size_t nx, size_t ny, Real hx, Real hy,
                    Real ox, Real oy, Callback callback)
{
   for (size_t j = 0; j < ny; ++j)
   for (size_t i = 0; i < nx; ++i)
   for (size_t axis = 0; axis < 2; ++axis)
   for (size_t side = 0; side < 2; ++side)
   {
      const size_t index = axis == 0 ? i : j;
      const size_t size = axis == 0 ? nx : ny;
      if ((side == 0 && index != 0) || (side == 1 && index + 1 != size)) { continue; }
      for (size_t q = 0; q < points.size(); ++q)
      {
         const Real xi = axis == 0 ? Real(side) : points[q];
         const Real eta = axis == 1 ? Real(side) : points[q];
         callback(PointData{i + nx*j, xi, eta, ox + hx*(i+xi), oy + hy*(j+eta),
            axis == 0 ? 2*Real(side)-1 : 0,
            axis == 1 ? 2*Real(side)-1 : 0,
            weights[q]*(axis == 0 ? hy : hx)});
      }
   }
}

template<class Callback>
void CellPoints(size_t nx, size_t ny, Real hx, Real hy,
                Real ox, Real oy, Callback callback)
{
   for (size_t j = 0; j < ny; ++j)
   for (size_t i = 0; i < nx; ++i)
   for (size_t qy = 0; qy < points.size(); ++qy)
   for (size_t qx = 0; qx < points.size(); ++qx)
   {
      const Real xi = points[qx], eta = points[qy];
      callback(PointData{i + nx*j, xi, eta, ox + hx*(i+xi), oy + hy*(j+eta),
         0, 0, weights[qx]*weights[qy]*hx*hy});
   }
}

template<size_t N>
void AddP1(std::array<Real, N>& rows, const PointData& p, Real hx, Real hy,
           Real value, Real flux_x = 0, Real flux_y = 0)
{
   for (size_t j = 0; j < 2; ++j)
   for (size_t i = 0; i < 2; ++i)
   {
      const Real bx = i ? p.xi : 1-p.xi, by = j ? p.eta : 1-p.eta;
      rows[4*p.cell+i+2*j] += p.weight *
         (value*bx*by + flux_x*(i ? 1 : -1)*by/hx + flux_y*bx*(j ? 1 : -1)/hy);
   }
}

template<size_t N>
bool Check(const Vector& vector, const std::array<Real, N>& expected,
           const char* name, Real scale = 1, Real initial = 0)
{
   bool ok = true;
   const auto* actual = vector.ReadHostData();
   for (size_t i = 0; i < size_t(vector.Size()); ++i)
   {
      const Real reference = initial + (i < N ? scale*expected[i] : 0);
      if (!std::isfinite(actual[i]) || !std::isfinite(reference) ||
          std::abs(actual[i] - reference) > 3e-11)
      {
         std::cerr << name << " coordinate " << i << ": " << actual[i]
                   << " expected " << reference << '\n';
         ok = false;
      }
   }
   return ok;
}

template<class Operator, size_t N>
bool CheckLoad(const Operator& op, const std::array<Real, N>& expected, const char* name)
{
   bool ok = op.InputSize() == 0 && op.OutputSize() == N;
   Vector y(N+3);
   y = 7;
   op(y); ok = Check(y, expected, name) && ok;
   op(y); ok = Check(y, expected, name) && ok;
   y = -2;
   op.ApplyAdd(y); ok = Check(y, expected, name, 1, -2) && ok;
   op.ApplyAdd(y); ok = Check(y, expected, name, 2, -2) && ok;
   return ok;
}
} // namespace boundary_test
