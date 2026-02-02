// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace gendil;

namespace {

// ---------------------------------------------------------------
// Pretty-print a 2D field's DOFs as per-cell (p+1)x(p+1) blocks.
// Cells laid out with x-fastest (i,j), local DOFs a-fastest then b.
// ---------------------------------------------------------------
static inline Integer lin_cell_2d(Integer i, Integer j, Integer nx) {
  return i + nx*j; // x-fastest
}

void print_dofs_grid(const std::string& title,
                     const Vector& v,          // vector in FULL layout (nx × ny cells)
                     Integer nx, Integer ny,   // cell counts
                     Integer p)                // polynomial degree
{
  const Integer nldofs = (p+1)*(p+1);
  const Real* data = v.ReadHostData();

  std::cout << "\n=== " << title << " ===\n";
  std::cout << std::fixed << std::setprecision(6);

  // top row first, safe reverse for possibly unsigned Integer
  for (Integer row = ny; row-- > 0; ) {
    const Integer j = row;

    // local b = p..0 (top to bottom)
    for (Integer b_row = p+1; b_row-- > 0; ) {
      const Integer b = b_row;

      // across cells in row
      for (Integer i = 0; i < nx; ++i) {
        const Integer base = lin_cell_2d(i, j, nx) * nldofs;

        std::cout << " |";
        // local a = 0..p (left to right)
        for (Integer a = 0; a <= p; ++a) {
          const Integer l = a + (p+1)*b;
          const Real val = data[ base + l ];
          std::cout << " " << std::setw(10) << val;
        }
      }
      std::cout << " |\n";
    }

    // row separator
    for (Integer i = 0; i < nx; ++i) {
      std::cout << " +" << std::string(int((p+1)*12), '-');
    }
    std::cout << " +\n";
  }
}

} // namespace

int main(int, char**)
{
  // --------------------------
  // Config
  // --------------------------
  constexpr Integer Dim = 2;
  constexpr Integer MeshFaceIndex = Dim;
  const Integer nxL = 2, nyL = 2;   // coarse left
  const Integer nxR = 2, nyR = 4;   // fine right (ry = 2)

  constexpr Integer p   = 2;
  constexpr Integer q1d = p + 2;
  const Integer nldofs  = (p+1)*(p+1);

  const Real hx = 1.0 / nxL; // domain length in x = 2
  const Real hyL = 1.0 / nyL;
  const Real hyR = 1.0 / (nyR); // same physical height

  // --------------------------
  // Meshes
  // --------------------------
  Cartesian2DMesh meshL(hx, hyL, nxL, nyL, Point<2>{0.0, 0.0});
  Cartesian2DMesh meshR(hx, hyR, nxR, nyR, Point<2>{1.0, 0.0}); // shifted +1 in x

  // --------------------------
  // FE (no p-adaptivity)
  // --------------------------
  auto fe = MakeLobattoFiniteElement(FiniteElementOrders<p,p>{});

  // Layout: split vector = [L | R]
  L2Restriction resL{0};
  auto fe_space_L = MakeFiniteElementSpace(meshL, fe, resL);
  const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();

  L2Restriction resR{ndofsL};
  auto fe_space_R = MakeFiniteElementSpace(meshR, fe, resR);
  const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

  const Integer ndofs_split = ndofsL + ndofsR;

  // --------------------------
  // Rules
  // --------------------------
  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d>{});

  // Interior face meshes (each block)
  auto face_mesh_L = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxL,(GlobalIndex)nyL});
  auto face_mesh_R = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxR,(GlobalIndex)nyR});

  // Nonconforming intermesh (minus=coarse L, plus=fine R)
  NonconformingCartesianIntermeshFaceConnectivity<Dim,MeshFaceIndex> iface(
      /*coarse*/{(GlobalIndex)nxL,(GlobalIndex)nyL},
      /*fine  */{(GlobalIndex)nxR,(GlobalIndex)nyR});

  // --------------------------
  // Advection β = (+1, 0)
  // --------------------------
  auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>&, Real (&v)[Dim]) {
    v[0] = 1.0; v[1] = 0.0;
  };

  // --------------------------
  // Kernel policy
  // --------------------------
#if defined(GENDIL_USE_DEVICE)
  using ThreadLayout = ThreadBlockLayout<q1d,q1d>;
  constexpr size_t NumSharedDimensions = Dim;
  using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
  using KernelPolicy = SerialKernelConfiguration;
#endif

  // --------------------------
  // Operators
  // --------------------------
  auto face_L  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, face_mesh_L, int_rules, adv);
  auto face_R  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_R, face_mesh_R, int_rules, adv);
  auto face_IF = MakeAdvectionFaceOperator<KernelPolicy>(
                    /*minus*/fe_space_L, /*plus*/fe_space_R, iface, int_rules, adv);

  // --------------------------
  // Helpers to slice [L | R] into compact views for printing
  // --------------------------
  auto slice_left = [&](const Vector& r_split) {
    Vector rL(ndofsL);
    rL = 0.0;
    const Real* src = r_split.ReadHostData();
    Real*       dst = rL.ReadWriteHostData();
    for (Integer j=0; j<nyL; ++j)
      for (Integer i=0; i<nxL; ++i) {
        const Integer base_src = /*L offset*/ 0 + lin_cell_2d(i, j, nxL)*nldofs;
        const Integer base_dst = lin_cell_2d(i, j, nxL)*nldofs;
        for (Integer l=0; l<nldofs; ++l) dst[base_dst + l] = src[base_src + l];
      }
    return rL;
  };

  auto slice_right = [&](const Vector& r_split) {
    Vector rR(ndofsR);
    rR = 0.0;
    const Real* src = r_split.ReadHostData();
    Real*       dst = rR.ReadWriteHostData();
    for (Integer j=0; j<nyR; ++j)
      for (Integer i=0; i<nxR; ++i) {
        const Integer base_src = ndofsL + lin_cell_2d(i, j, nxR)*nldofs; // start at R offset
        const Integer base_dst = lin_cell_2d(i, j, nxR)*nldofs;
        for (Integer l=0; l<nldofs; ++l) dst[base_dst + l] = src[base_src + l];
      }
    return rR;
  };

  // --------------------------
  // Cases
  // --------------------------
  auto apply_all = [&](const Vector& u_split, Vector& r_split){
    r_split = 0.0;
    face_L(u_split, r_split);
    face_R(u_split, r_split);
    face_IF(u_split, r_split);
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "[2D h-nonconforming] p=" << p << " q1d=" << q1d
            << "  nxL=" << nxL << " nyL=" << nyL
            << "  nxR=" << nxR << " nyR=" << nyR << "\n"
            << "  ndofsL=" << ndofsL << "  ndofsR=" << ndofsR
            << "  ndofs_split=" << ndofs_split << "\n";

  // (1) Constant → conservation (sum ≈ 0)
  {
    Vector u(ndofs_split), r(ndofs_split);
    u = 1.0;
    r = 0.0;
    apply_all(u, r);
    Real sum = 0.0;
    const Real* pr = r.ReadHostData();
    for (Integer g=0; g<ndofs_split; ++g) sum += pr[g];
    std::cout << "  [Const u=1] sum(r) = " << sum << "  → "
              << (std::abs(sum) <= 1e-12 ? "PASS" : "FAIL") << "\n";
  }

  // (2) Interface pulse (coarse left interface column only)
  {
    Vector u(ndofs_split), r(ndofs_split);
    u = 0.0;
    r = 0.0;
    // Set u=1 on coarse L cells with i==nxL-1
    {
      Real* pu = u.ReadWriteHostData();
      for (Integer j=0; j<nyL; ++j) {
        const Integer i = nxL-1;
        const Integer base = /*L offset*/ 0 + lin_cell_2d(i, j, nxL)*nldofs;
        for (Integer a=0; a<=p; ++a)
          for (Integer b=0; b<=p; ++b)
            pu[ base + (a + (p+1)*b) ] = 1.0;
      }
    }
    apply_all(u, r);

    std::cout << "\n[Interface pulse on coarse-left column]\n";
    auto rL = slice_left(r);
    auto rR = slice_right(r);
    print_dofs_grid("r (Left block)",  rL, nxL, nyL, p);
    print_dofs_grid("r (Right block)", rR, nxR, nyR, p);
  }

  // (3) r=1 fallback → matches conforming intermesh (interface-only)
  {
    const Integer ny_equal = nyL;
    Cartesian2DMesh meshR1(hx, 1.0/ny_equal, nxR, ny_equal, Point<2>{1.0,0.0});
    auto fe_space_R1 = MakeFiniteElementSpace(meshR1, fe, L2Restriction{ndofsL});
    const Integer ndofsR1 = fe_space_R1.GetNumberOfFiniteElementDofs();

    // Nonconforming iface with ry=1 (degenerates to conforming)
    NonconformingCartesianIntermeshFaceConnectivity<Dim, MeshFaceIndex> iface1(
        {(GlobalIndex)nxL,(GlobalIndex)nyL},
        {(GlobalIndex)nxR,(GlobalIndex)ny_equal});

    // Conforming iface with the same minus/plus convention (minus=L, plus=R1)
    CartesianIntermeshFaceConnectivity<Dim, MeshFaceIndex> iface_conf(
        {(GlobalIndex)nxL,(GlobalIndex)nyL},
        {(GlobalIndex)nxR,(GlobalIndex)ny_equal});

    auto face_IF1  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, fe_space_R1, iface1,  int_rules, adv);
    auto face_IFc  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, fe_space_R1, iface_conf, int_rules, adv);

    // Random-but-deterministic u on split layout [L | R1]
    Vector u(ndofsL + ndofsR1), rA(u.Size()), rB(u.Size());
    {
      Real* pu = u.WriteHostData();
      uint32_t s = 0xC0FFEEu;
      auto rnd = [&](){ s^=s<<13; s^=s>>17; s^=s<<5; return Real((s&0xFFFF))/Real(0xFFFF); };
      for (Integer g=0; g<u.Size(); ++g) pu[g] = rnd();
    }

    // Compare INTERFACE ONLY
    rA = 0.0; face_IF1(u, rA);
    rB = 0.0; face_IFc(u, rB);

    Real l2 = 0.0;
    const Real *a = rA.ReadHostData(), *b = rB.ReadHostData();
    for (Integer g=0; g<rA.Size(); ++g) {
      const Real d = a[g] - b[g];
      l2 += d*d;
    }
    l2 = std::sqrt(l2);
    std::cout << "\n[r=1 fallback: interface-only] ||Δ||_2 = " << l2
              << "  → " << (l2 <= 1e-12 ? "PASS" : "FAIL") << "\n";
  }

  return 0;
}
