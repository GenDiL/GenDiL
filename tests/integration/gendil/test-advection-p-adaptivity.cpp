// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace gendil;

namespace
{

struct CellIndex3D
{
   Integer i;
   Integer j;
   Integer k;
};

CellIndex3D DecodeCell3D( GlobalIndex cell, Integer nx, Integer ny )
{
   const Integer c = static_cast<Integer>( cell );
   return {
      c % nx,
      ( c / nx ) % ny,
      c / ( nx * ny )
   };
}

struct SegmentStats
{
   Integer count = 0;
   Integer nonzero = 0;
   Integer nonfinite = 0;
   Real sum = 0.0;
   Real l1 = 0.0;
   Real l2_sq = 0.0;
   Real linf = 0.0;
   Real min = std::numeric_limits<Real>::infinity();
   Real max = -std::numeric_limits<Real>::infinity();
};

void AddSample( SegmentStats & stats, Real value )
{
   stats.count++;
   if ( !std::isfinite( value ) )
   {
      stats.nonfinite++;
      return;
   }

   const Real abs_value = std::abs( value );
   stats.sum += value;
   stats.l1 += abs_value;
   stats.l2_sq += value * value;
   stats.linf = std::max( stats.linf, abs_value );
   stats.min = std::min( stats.min, value );
   stats.max = std::max( stats.max, value );
   if ( abs_value > 0.0 )
   {
      stats.nonzero++;
   }
}

void PrintStats( const std::string & label, const SegmentStats & stats )
{
   const Real l2 = std::sqrt( stats.l2_sq );
   const Real min_value = stats.count > stats.nonfinite ? stats.min : 0.0;
   const Real max_value = stats.count > stats.nonfinite ? stats.max : 0.0;

   std::cout
      << "  " << std::setw(20) << std::left << label
      << " count=" << std::setw(5) << stats.count
      << " nonzero=" << std::setw(5) << stats.nonzero
      << " nonfinite=" << std::setw(3) << stats.nonfinite
      << " sum=" << std::setw(18) << stats.sum
      << " l1=" << std::setw(18) << stats.l1
      << " l2=" << std::setw(18) << l2
      << " linf=" << std::setw(18) << stats.linf
      << " min=" << std::setw(18) << min_value
      << " max=" << std::setw(18) << max_value
      << "\n";
}

struct DecodedDof
{
   std::string side;
   Integer global_index = 0;
   Integer side_index = 0;
   Integer cell = 0;
   CellIndex3D cell_index{};
   Integer local_dof = 0;
   Real value = 0.0;
};

DecodedDof DecodeDof(
   Integer global_index,
   Real value,
   Integer ndofsL,
   Integer nldofsL,
   Integer nldofsR,
   Integer nxL,
   Integer nxR,
   Integer ny )
{
   DecodedDof decoded;
   decoded.global_index = global_index;
   decoded.value = value;

   if ( global_index < ndofsL )
   {
      decoded.side = "L";
      decoded.side_index = global_index;
      decoded.cell = decoded.side_index / nldofsL;
      decoded.local_dof = decoded.side_index % nldofsL;
      decoded.cell_index = DecodeCell3D( decoded.cell, nxL, ny );
   }
   else
   {
      decoded.side = "R";
      decoded.side_index = global_index - ndofsL;
      decoded.cell = decoded.side_index / nldofsR;
      decoded.local_dof = decoded.side_index % nldofsR;
      decoded.cell_index = DecodeCell3D( decoded.cell, nxR, ny );
   }

   return decoded;
}

void PrintDecodedDof( const DecodedDof & decoded )
{
   std::cout
      << "    g=" << std::setw(5) << decoded.global_index
      << " side=" << decoded.side
      << " side_i=" << std::setw(5) << decoded.side_index
      << " cell=" << std::setw(4) << decoded.cell
      << " cell=(" << decoded.cell_index.i
      << "," << decoded.cell_index.j
      << "," << decoded.cell_index.k << ")"
      << " local=" << std::setw(4) << decoded.local_dof
      << " value=" << decoded.value
      << "\n";
}

void PrintLargestEntries(
   const std::string & label,
   const Real * data,
   Integer size,
   Integer ndofsL,
   Integer nldofsL,
   Integer nldofsR,
   Integer nxL,
   Integer nxR,
   Integer ny,
   Integer max_entries = 12 )
{
   std::vector< Integer > order( size );
   for ( Integer i = 0; i < size; ++i )
   {
      order[i] = i;
   }

   std::sort(
      order.begin(),
      order.end(),
      [=]( Integer a, Integer b )
      {
         return std::abs( data[a] ) > std::abs( data[b] );
      } );

   std::cout << label << "\n";
   for ( Integer entry = 0; entry < std::min( max_entries, size ); ++entry )
   {
      const Integer i = order[entry];
      PrintDecodedDof(
         DecodeDof( i, data[i], ndofsL, nldofsL, nldofsR, nxL, nxR, ny ) );
   }
}

void PrintLargestDifferences(
   const std::string & label,
   const Real * a,
   const Real * b,
   Integer size,
   Integer ndofsL,
   Integer nldofsL,
   Integer nldofsR,
   Integer nxL,
   Integer nxR,
   Integer ny,
   Integer max_entries = 12 )
{
   std::vector< Integer > order( size );
   for ( Integer i = 0; i < size; ++i )
   {
      order[i] = i;
   }

   std::sort(
      order.begin(),
      order.end(),
      [=]( Integer lhs, Integer rhs )
      {
         return std::abs( a[lhs] - b[lhs] ) >
                std::abs( a[rhs] - b[rhs] );
      } );

   std::cout << label << "\n";
   for ( Integer entry = 0; entry < std::min( max_entries, size ); ++entry )
   {
      const Integer i = order[entry];
      auto decoded = DecodeDof(
         i, a[i] - b[i], ndofsL, nldofsL, nldofsR, nxL, nxR, ny );
      PrintDecodedDof( decoded );
      std::cout
         << "      a=" << a[i]
         << " b=" << b[i]
         << " abs_diff=" << std::abs( a[i] - b[i] )
         << "\n";
   }
}

void PrintValidity( const std::string & label, const Vector & u, const Vector & r )
{
   std::cout
      << label
      << " u(host=" << u.IsHostValid()
      << ", device=" << u.IsDeviceValid()
      << ") r(host=" << r.IsHostValid()
      << ", device=" << r.IsDeviceValid()
      << ")\n";
}

void FillConstantHost( Vector & v, Real value )
{
   Real * data = v.WriteHostData();
   for ( Integer i = 0; i < v.Size(); ++i )
   {
      data[i] = value;
   }
}

template < typename FaceMesh >
std::vector< char > MarkLeftInterfaceCells(
   const FaceMesh & face_mesh,
   Integer num_left_cells )
{
   std::vector< char > marked( num_left_cells, 0 );
   for ( GlobalIndex f = 0; f < face_mesh.GetNumberOfFaces(); ++f )
   {
      const auto face_info = face_mesh.GetGlobalFaceInfo( f );
      const GlobalIndex cell = face_info.MinusSide().GetCellIndex();
      if ( cell < static_cast<GlobalIndex>( marked.size() ) )
      {
         marked[cell] = 1;
      }
   }
   return marked;
}

template < typename FaceMesh >
std::vector< char > MarkRightInterfaceCells(
   const FaceMesh & face_mesh,
   Integer num_right_cells )
{
   std::vector< char > marked( num_right_cells, 0 );
   for ( GlobalIndex f = 0; f < face_mesh.GetNumberOfFaces(); ++f )
   {
      const auto face_info = face_mesh.GetGlobalFaceInfo( f );
      const GlobalIndex cell = face_info.PlusSide().GetCellIndex();
      if ( cell < static_cast<GlobalIndex>( marked.size() ) )
      {
         marked[cell] = 1;
      }
   }
   return marked;
}

void PrintResidualReport(
   const std::string & label,
   const Vector & r,
   Integer ndofsL,
   Integer ndofsR,
   Integer nldofsL,
   Integer nldofsR,
   Integer nxL,
   Integer nxR,
   Integer ny,
   const std::vector< char > & left_interface_cells,
   const std::vector< char > & right_interface_cells )
{
   const Real * data = r.ReadHostData();

   SegmentStats left;
   SegmentStats right;
   SegmentStats total;
   SegmentStats left_interface;
   SegmentStats right_interface;
   SegmentStats left_off_interface;
   SegmentStats right_off_interface;

   for ( Integer i = 0; i < ndofsL; ++i )
   {
      AddSample( left, data[i] );
      AddSample( total, data[i] );
      const Integer cell = i / nldofsL;
      AddSample(
         left_interface_cells[cell] ? left_interface : left_off_interface,
         data[i] );
   }

   for ( Integer i = 0; i < ndofsR; ++i )
   {
      const Integer global = ndofsL + i;
      AddSample( right, data[global] );
      AddSample( total, data[global] );
      const Integer cell = i / nldofsR;
      AddSample(
         right_interface_cells[cell] ? right_interface : right_off_interface,
         data[global] );
   }

   std::cout << "\n=== Residual report: " << label << " ===\n";
   PrintStats( "left", left );
   PrintStats( "right", right );
   PrintStats( "total", total );
   PrintStats( "left interface", left_interface );
   PrintStats( "right interface", right_interface );
   PrintStats( "left off-interface", left_off_interface );
   PrintStats( "right off-interface", right_off_interface );

   PrintLargestEntries(
      "  Largest residual entries:",
      data,
      ndofsL + ndofsR,
      ndofsL,
      nldofsL,
      nldofsR,
      nxL,
      nxR,
      ny );
}

template < typename FaceMesh >
void PrintFacePairReport(
   const FaceMesh & face_mesh,
   const Vector & r,
   Integer ndofsL,
   Integer nldofsL,
   Integer nldofsR,
   Integer nxL,
   Integer nxR,
   Integer ny,
   Integer max_faces = 12 )
{
   const Real * data = r.ReadHostData();
   const GlobalIndex num_faces = face_mesh.GetNumberOfFaces();

   std::cout << "\n=== Per-interface-face cell sums ===\n";
   for ( GlobalIndex f = 0; f < std::min<GlobalIndex>( num_faces, max_faces ); ++f )
   {
      const auto face_info = face_mesh.GetGlobalFaceInfo( f );
      const Integer left_cell =
         static_cast<Integer>( face_info.MinusSide().GetCellIndex() );
      const Integer right_cell =
         static_cast<Integer>( face_info.PlusSide().GetCellIndex() );
      const auto left_ijk = DecodeCell3D( left_cell, nxL, ny );
      const auto right_ijk = DecodeCell3D( right_cell, nxR, ny );

      Real sum_left = 0.0;
      Real sum_right = 0.0;
      for ( Integer local = 0; local < nldofsL; ++local )
      {
         sum_left += data[left_cell * nldofsL + local];
      }
      for ( Integer local = 0; local < nldofsR; ++local )
      {
         sum_right += data[ndofsL + right_cell * nldofsR + local];
      }

      std::cout
         << "  face=" << std::setw(3) << f
         << " minus_cell=" << std::setw(3) << left_cell
         << " (" << left_ijk.i << "," << left_ijk.j << "," << left_ijk.k << ")"
         << " plus_cell=" << std::setw(3) << right_cell
         << " (" << right_ijk.i << "," << right_ijk.j << "," << right_ijk.k << ")"
         << " sum_minus=" << std::setw(18) << sum_left
         << " sum_plus=" << std::setw(18) << sum_right
         << " pair_sum=" << sum_left + sum_right
         << "\n";
   }
}

SegmentStats DifferenceStats( const Vector & a, const Vector & b )
{
   const Real * a_data = a.ReadHostData();
   const Real * b_data = b.ReadHostData();
   SegmentStats stats;

   for ( Integer i = 0; i < a.Size(); ++i )
   {
      AddSample( stats, a_data[i] - b_data[i] );
   }

   return stats;
}

} // namespace

int main(int, char**)
{
   std::cout.setf( std::ios::scientific );
   std::cout << std::setprecision( 16 ) << std::boolalpha;

   // --------------------------
   // Two side-by-side Cartesian meshes in 3D
   // Left: [0,1]^3, Right: [1,2]x[0,1]^2 (same h, geometry-conforming)
   // --------------------------
   const Integer num_elem_1d = 3;
   const Real h = 1.0 / num_elem_1d;

   const Integer nxL = num_elem_1d;
   const Integer ny = num_elem_1d;
   const Integer nz = num_elem_1d;
   const Integer nxR = num_elem_1d; // p-nonconforming only, same tangential partition

   Point<3> originL = {0.0, 0.0, 0.0};
   Point<3> originR = {1.0, 0.0, 0.0}; // shift by +1 in x

   Cartesian3DMesh meshL(h, h, h, nxL, ny, nz, originL);
   Cartesian3DMesh meshR(h, h, h, nxR, ny, nz, originR);

   // --------------------------
   // FE spaces with different polynomial orders (p-nonconforming)
   // --------------------------
   constexpr Integer pL = 1;
   constexpr Integer pR = 3;

   auto feL = MakeLobattoFiniteElement(FiniteElementOrders<pL,pL,pL>{});
   auto feR = MakeLobattoFiniteElement(FiniteElementOrders<pR,pR,pR>{});

   L2Restriction resL{ 0 };
   auto fe_space_L = MakeFiniteElementSpace(meshL, feL, resL);
   const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();
   L2Restriction resR{ ndofsL };
   auto fe_space_R = MakeFiniteElementSpace(meshR, feR, resR);
   const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();
   const Integer ndofs_total = ndofsL + ndofsR;

   constexpr Integer Dim = GetDim(fe_space_L);
   static_assert(Dim == 3, "This test is set up for 3D.");

   constexpr Integer q1d = (pL > pR ? pL : pR) + 2;
   auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d>{});
   using IntegrationRule = std::remove_cvref_t< decltype( int_rules ) >;

   std::array<GlobalIndex, Dim> sizesL{(GlobalIndex)nxL, (GlobalIndex)ny, (GlobalIndex)nz};
   std::array<GlobalIndex, Dim> sizesR{(GlobalIndex)nxR, (GlobalIndex)ny, (GlobalIndex)nz};

   // Keep this connectivity as-is while debugging. The diagnostics below print
   // the actual minus/plus cell pairs selected by this convention.
   CartesianIntermeshFaceConnectivity<Dim, 0> face_mesh(sizesL, sizesR);
   using FaceMesh = std::remove_cvref_t< decltype( face_mesh ) >;

   auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X, Real (&v)[Dim]) {
      const Real x = X[0], y = X[1], z = X[2];
      v[0] = y; v[1] = z; v[2] = x;
   };

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout  = ThreadBlockLayout<q1d,q1d,q1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   constexpr size_t required_shared_mem =
      required_shared_memory_v<KernelPolicy, IntegrationRule>;
   constexpr Integer nldofsL = (pL + 1) * (pL + 1) * (pL + 1);
   constexpr Integer nldofsR = (pR + 1) * (pR + 1) * (pR + 1);

   std::cout << "=== test-advection-p-adaptivity diagnostics ===\n";
#if defined(GENDIL_USE_DEVICE)
   std::cout << "Backend              = device\n";
   std::cout << "ThreadLayout         = <" << q1d << "," << q1d << "," << q1d << ">\n";
   std::cout << "threads/block        = " << q1d * q1d * q1d << "\n";
#else
   std::cout << "Backend              = serial\n";
#endif
   std::cout << "Dim                  = " << Dim << "\n";
   std::cout << "pL, pR, q1d          = " << pL << ", " << pR << ", " << q1d << "\n";
   std::cout << "mesh L               = (" << nxL << "," << ny << "," << nz << ")\n";
   std::cout << "mesh R               = (" << nxR << "," << ny << "," << nz << ")\n";
   std::cout << "ndofs L/R/total      = " << ndofsL << ", " << ndofsR << ", " << ndofs_total << "\n";
   std::cout << "local dofs L/R       = " << nldofsL << ", " << nldofsR << "\n";
   std::cout << "FaceMesh LFI         = 0\n";
   std::cout << "FaceMesh axis/sign   = " << FaceMesh::axis << ", " << FaceMesh::sign << "\n";
   std::cout << "FaceMesh num faces   = " << face_mesh.GetNumberOfFaces() << "\n";
   std::cout << "required shared mem  = " << required_shared_mem
             << " doubles (" << required_shared_mem * sizeof(Real) << " bytes)\n";

   std::cout << "\n=== First face pairs selected by connectivity ===\n";
   for ( GlobalIndex f = 0; f < std::min<GlobalIndex>( face_mesh.GetNumberOfFaces(), 12 ); ++f )
   {
      const auto face_info = face_mesh.GetGlobalFaceInfo( f );
      const auto left_ijk = DecodeCell3D( face_info.MinusSide().GetCellIndex(), nxL, ny );
      const auto right_ijk = DecodeCell3D( face_info.PlusSide().GetCellIndex(), nxR, ny );
      std::cout
         << "  face=" << std::setw(3) << f
         << " minus_cell=" << std::setw(3) << face_info.MinusSide().GetCellIndex()
         << " (" << left_ijk.i << "," << left_ijk.j << "," << left_ijk.k << ")"
         << " plus_cell=" << std::setw(3) << face_info.PlusSide().GetCellIndex()
         << " (" << right_ijk.i << "," << right_ijk.j << "," << right_ijk.k << ")"
         << "\n";
   }

   auto face_op = MakeAdvectionFaceOperator<KernelPolicy>(
      fe_space_L, fe_space_R, face_mesh, int_rules, adv);

   auto serial_face_op = MakeAdvectionFaceOperator<SerialKernelConfiguration>(
      fe_space_L, fe_space_R, face_mesh, int_rules, adv);

   const auto left_interface_cells =
      MarkLeftInterfaceCells( face_mesh, nxL * ny * nz );
   const auto right_interface_cells =
      MarkRightInterfaceCells( face_mesh, nxR * ny * nz );

   auto run_case = [&]( const std::string & label, bool explicit_host_init, auto & op )
   {
      Vector u( ndofs_total );
      Vector r( ndofs_total );

      if ( explicit_host_init )
      {
         FillConstantHost( u, 1.0 );
         FillConstantHost( r, 0.0 );
      }
      else
      {
         u = 1.0;
         r = 0.0;
      }

      std::cout << "\n=== Apply case: " << label << " ===\n";
      PrintValidity( "  before apply     ", u, r );
      op( u, r );
      GENDIL_DEVICE_SYNC;
      PrintValidity( "  after apply      ", u, r );
      r.ReadHostData();
      PrintValidity( "  after ReadHost   ", u, r );

      return r;
   };

   Vector r_current = run_case( "current vector initialization", false, face_op );
   Vector r_host_init = run_case( "explicit host initialization", true, face_op );
   Vector r_serial = run_case( "serial reference", true, serial_face_op );

   PrintResidualReport(
      "current vector initialization",
      r_current,
      ndofsL,
      ndofsR,
      nldofsL,
      nldofsR,
      nxL,
      nxR,
      ny,
      left_interface_cells,
      right_interface_cells );
   PrintFacePairReport(
      face_mesh,
      r_current,
      ndofsL,
      nldofsL,
      nldofsR,
      nxL,
      nxR,
      ny );

   const SegmentStats current_vs_host = DifferenceStats( r_current, r_host_init );
   const SegmentStats current_vs_serial = DifferenceStats( r_current, r_serial );

   std::cout << "\n=== Difference reports ===\n";
   PrintStats( "current - host-init", current_vs_host );
   PrintStats( "current - serial", current_vs_serial );

   const Real * current_data = r_current.ReadHostData();
   const Real * host_data = r_host_init.ReadHostData();
   const Real * serial_data = r_serial.ReadHostData();

   PrintLargestDifferences(
      "  Largest current-vs-host-init differences:",
      current_data,
      host_data,
      ndofs_total,
      ndofsL,
      nldofsL,
      nldofsR,
      nxL,
      nxR,
      ny );

   PrintLargestDifferences(
      "  Largest current-vs-serial differences:",
      current_data,
      serial_data,
      ndofs_total,
      ndofsL,
      nldofsL,
      nldofsR,
      nxL,
      nxR,
      ny );

   SegmentStats total_stats;
   for ( Integer i = 0; i < ndofs_total; ++i )
   {
      AddSample( total_stats, current_data[i] );
   }

   const Real conservation_error = std::abs( total_stats.sum );
   constexpr Real conservation_tol = 1e-10;
   constexpr Real comparison_tol = 1e-10;

   std::cout << "\n=== Final checks ===\n";
   std::cout
      << "  conservation |sum| = " << conservation_error
      << " tolerance = " << conservation_tol
      << " -> " << ( conservation_error <= conservation_tol ? "PASS" : "FAIL" )
      << "\n";
   std::cout
      << "  current-vs-host linf = " << current_vs_host.linf
      << " tolerance = " << comparison_tol
      << " -> " << ( current_vs_host.linf <= comparison_tol ? "PASS" : "FAIL" )
      << "\n";
   std::cout
      << "  current-vs-serial linf = " << current_vs_serial.linf
      << " tolerance = " << comparison_tol
      << " -> " << ( current_vs_serial.linf <= comparison_tol ? "PASS" : "FAIL" )
      << "\n";

   if ( total_stats.nonfinite > 0 )
   {
      return 4;
   }
   if ( current_vs_serial.linf > comparison_tol )
   {
      return 3;
   }
   if ( conservation_error > conservation_tol )
   {
      return 2;
   }

   return 0;
}
