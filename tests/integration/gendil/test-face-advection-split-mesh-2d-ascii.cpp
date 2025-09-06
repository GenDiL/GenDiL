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
template<typename T> inline T sqr(T x) { return x*x; }
static inline Integer lin_cell_2d(Integer i, Integer j, Integer nx) { return i + nx*j; }

// Pretty-print a 2D field's DOFs as per-cell (p+1)x(p+1) blocks.
// Assumptions:
//  - Dim = 2
//  - global cells ordered lexicographically with x fastest (i,j)
//  - per-cell DOFs contiguous, local order a (x) fastest then b (y)
void print_dofs_grid(const std::string& title,
                     const Vector& v,          // vector in FULL layout
                     Integer nx, Integer ny,   // cells in x/y in FULL mesh
                     Integer p)
{
  const Integer nldofs = (p+1)*(p+1);
  const Real* data = v.ReadHostData();

  std::cout << "\n=== " << title << " ===\n";
  std::cout << std::fixed << std::setprecision(6);

  // Print from top row (j=ny-1) down to 0 (safe for unsigned Integer).
  for (Integer row = ny; row-- > 0; ) {
    const Integer j = row;

    // For each local b-row inside the cell, from top (b=p) to bottom (b=0)
    for (Integer b_row = p+1; b_row-- > 0; ) {
      const Integer b = b_row;

      // Across all cells in this mesh row
      for (Integer i = 0; i < nx; ++i) {
        const Integer cell = lin_cell_2d(i, j, nx);
        const Integer base = cell * nldofs;

        std::cout << " |"; // left border for this cell

        // Local a from left (a=0) to right (a=p)
        for (Integer a = 0; a <= p; ++a) {
          const Integer l = a + (p+1)*b;   // local dof index in the cell
          const Integer g = base + l;      // global dof index
          std::cout << " " << std::setw(10) << data[g];
        }
      }
      std::cout << " |\n";
    }

    // Separator under the cell row
    for (Integer i = 0; i < nx; ++i) {
      std::cout << " +" << std::string((int)((p+1)*12), '-') ;
    }
    std::cout << " +\n";
  }
}
} // namespace

int main(int, char**)
{
  // --------------------------
  // Config: 2D, split along x (Axis=0)
  // --------------------------
  constexpr Integer Dim = 2;
  const Integer nxL = 2, nxR = 2, ny = 2;  // tiny for readable prints
  const Integer nx_full = nxL + nxR;

  constexpr Integer p   = 2;               // polynomial degree
  constexpr Integer q1d = p + 2;           // quadrature chooser
  const Integer nldofs  = (p+1)*(p+1);

  const Real hx = 1.0 / nxL; // full domain length = 2.0 in x
  const Real hy = 1.0 / ny;

  // What to print (toggle as needed)
  constexpr bool PRINT_RANDOM   = false;
  constexpr bool PRINT_HALFJUMP = false;
  constexpr bool PRINT_PULSE    = true;

  // --------------------------
  // Meshes
  // --------------------------
  Cartesian2DMesh mesh_full(hx, hy, nx_full, ny, Point<2>{0.0, 0.0});
  Cartesian2DMesh meshL    (hx, hy, nxL,     ny, Point<2>{0.0, 0.0});
  Cartesian2DMesh meshR    (hx, hy, nxR,     ny, Point<2>{1.0, 0.0}); // shift +1 in x

  // --------------------------
  // FE: no p-adaptivity
  // --------------------------
  auto fe = MakeLobattoFiniteElement(FiniteElementOrders<p,p>{});

  // Layouts (full vs split)
  L2Restriction res_full{0};
  auto fe_space_full = MakeFiniteElementSpace(mesh_full, fe, res_full);
  const Integer ndofs_full = fe_space_full.GetNumberOfFiniteElementDofs();

  L2Restriction resL{0};
  auto fe_space_L = MakeFiniteElementSpace(meshL, fe, resL);
  const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();

  L2Restriction resR{ndofsL}; // split layout = [L | R]
  auto fe_space_R = MakeFiniteElementSpace(meshR, fe, resR);
  const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

  if (ndofs_full != ndofsL + ndofsR) {
    std::cerr << "Layout mismatch: ndofs_full=" << ndofs_full
              << "  ndofsL+ndofsR=" << (ndofsL + ndofsR) << "\n";
    return 10;
  }

  // --------------------------
  // Integration rules
  // --------------------------
  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d>{});

  // --------------------------
  // Face meshes
  // --------------------------
  auto face_mesh_full = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nx_full,(GlobalIndex)ny});
  auto face_mesh_L = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxL,(GlobalIndex)ny});
  auto face_mesh_R = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxR,(GlobalIndex)ny});

  // Intermesh: lhs=minus=RIGHT, rhs=plus=LEFT (matches operator semantics)
  CartesianIntermeshFaceConnectivity<Dim,0> iface(
      /*sizes_minus = R*/{(GlobalIndex)nxR,(GlobalIndex)ny},
      /*sizes_plus  = L*/{(GlobalIndex)nxL,(GlobalIndex)ny});

  // --------------------------
  // Advection field: constant +ex
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
  auto face_full = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_full, face_mesh_full, int_rules, adv);
  auto face_L    = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L,    face_mesh_L,    int_rules, adv);
  auto face_R    = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_R,    face_mesh_R,    int_rules, adv);
  auto face_IF   = MakeAdvectionFaceOperator<KernelPolicy>(/*minus*/fe_space_R,
                                                           /*plus */fe_space_L,
                                                           iface, int_rules, adv);

  // --------------------------
  // Permutation FULL ↔ SPLIT
  // --------------------------
  // auto decode_full = [&](Integer g, Integer& i, Integer& j, Integer& a, Integer& b) {
  //   const Integer cell = g / nldofs;
  //   const Integer l    = g % nldofs;
  //   i =  cell % nx_full;
  //   j =  cell / nx_full;
  //   a =  l % (p+1);
  //   b =  l / (p+1);
  // };

  std::vector<Integer> P_full_to_split(ndofs_full), P_split_to_full(ndofs_full);
  for (Integer j=0; j<ny; ++j)
    for (Integer i=0; i<nx_full; ++i) {
      const bool inL = (i < nxL);
      const Integer iLR = inL ? i : (i - nxL);
      const Integer cellF  = lin_cell_2d(i, j, nx_full);
      const Integer cellLR = inL ? lin_cell_2d(iLR, j, nxL)
                                 : lin_cell_2d(iLR, j, nxR);
      const Integer baseF  = cellF  * nldofs;
      const Integer baseLR = (inL ? 0 : ndofsL) + cellLR * nldofs;
      for (Integer a=0; a<=p; ++a)
        for (Integer b=0; b<=p; ++b) {
          const Integer l    = a + (p+1)*b;
          const Integer gF   = baseF  + l;
          const Integer gLR  = baseLR + l;
          P_full_to_split[gF]  = gLR;
          P_split_to_full[gLR] = gF;
        }
    }
#ifndef NDEBUG
  for (Integer i=0;i<ndofs_full;++i) GENDIL_ASSERT(P_split_to_full[P_full_to_split[i]] == i);
  for (Integer j=0;j<ndofs_full;++j) GENDIL_ASSERT(P_full_to_split[P_split_to_full[j]] == j);
#endif

  auto apply_perm = [](const Vector& x, Vector& y, const std::vector<Integer>& f2s) {
    const Real* px = x.ReadHostData(); Real* py = y.ReadWriteHostData();
    const Integer n = (Integer)f2s.size();
    for (Integer i=0;i<n;++i) py[ f2s[i] ] = px[i];
  };
  auto inv_apply_perm = [](const Vector& x, Vector& y, const std::vector<Integer>& f2s) {
    const Real* px = x.ReadHostData(); Real* py = y.ReadWriteHostData();
    const Integer n = (Integer)f2s.size();
    for (Integer i=0;i<n;++i) py[i] = px[ f2s[i] ];
  };

  auto header = [&](){
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[2D DOF Print] Dim=" << Dim
              << "  p=" << p << "  q1d=" << q1d << "\n"
              << "  nx_full=" << nx_full << " (nxL=" << nxL << ", nxR=" << nxR << ")"
              << "  ny=" << ny << "\n"
              << "  ndofs_full=" << ndofs_full
              << "  ndofsL=" << ndofsL << "  ndofsR=" << ndofsR << "\n";
  };

  header();

  auto run_case = [&](const std::string& label, const Vector& u_full, bool do_print){
    Vector u_split(ndofs_full);
    apply_perm(u_full, u_split, P_full_to_split);

    Vector r_full(ndofs_full), r_split(ndofs_full), r_split_as_full(ndofs_full);
    r_full = 0.0; face_full(u_full, r_full);
    r_split = 0.0; face_L(u_split, r_split); face_R(u_split, r_split); face_IF(u_split, r_split);
    r_split_as_full = 0.0; inv_apply_perm(r_split, r_split_as_full, P_full_to_split);

    // metrics
    const Real *rf = r_full.ReadHostData(), *rs = r_split_as_full.ReadHostData();
    Real max_err=0.0, l2_err=0.0, l2_ref=0.0;
    for (Integer g=0; g<ndofs_full; ++g) {
      const Real e = rf[g]-rs[g];
      max_err = std::max(max_err, std::abs(e));
      l2_err += e*e; l2_ref += rf[g]*rf[g];
    }
    l2_err = std::sqrt(l2_err);
    l2_ref = std::sqrt(std::max(l2_ref, Real(1e-300)));
    std::cout << "  [" << label << "]  max|Δr|=" << max_err
              << "  relL2=" << (l2_err/l2_ref) << "\n";

    if (do_print) {
      print_dofs_grid("r_full",           r_full,          nx_full, ny, p);
      print_dofs_grid("r_split_as_full",  r_split_as_full, nx_full, ny, p);

      // difference (signed)
      Vector diff(ndofs_full);
      Real* d = diff.ReadWriteHostData();
      for (Integer g=0; g<ndofs_full; ++g) d[g] = rf[g] - rs[g];
      print_dofs_grid("Δr = r_full - r_split_as_full", diff, nx_full, ny, p);
    }
  };

  // ---------- TEST 1: Random ----------
  if (PRINT_RANDOM) {
    Vector u_full(ndofs_full);
    { // deterministic PRNG
      Real* uh = u_full.ReadWriteHostData();
      uint32_t s = 0xC0FFEEu;
      auto rnd = [&](){ s ^= s<<13; s ^= s>>17; s ^= s<<5; return Real((s & 0xFFFF))/Real(0xFFFF); };
      for (Integer i=0;i<ndofs_full;++i) uh[i] = rnd();
    }
    run_case("Random", u_full, /*do_print=*/true);
  }

  // ---------- TEST 2: Half-domain jump (L=1, R=0) ----------
  if (PRINT_HALFJUMP) {
    Vector u_full(ndofs_full); {
      Real* uf = u_full.ReadWriteHostData();
      for (Integer g=0; g<ndofs_full; ++g) {
        Integer I;
        const Integer cell = g / nldofs;
        I =  cell % nx_full;
        uf[g] = (I < nxL) ? Real(1.0) : Real(0.0);
      }
    }
    run_case("Half-jump (L=1,R=0)", u_full, /*do_print=*/true);
  }

  // ---------- TEST 3: Interface pulse (left column only) ----------
  if (PRINT_PULSE) {
    Vector u_full(ndofs_full); {
      Real* uf = u_full.ReadWriteHostData();
      for (Integer g=0; g<ndofs_full; ++g) {
        Integer I;
        const Integer cell = g / nldofs;
        I =  cell % nx_full;
        uf[g] = (I == nxL-1) ? Real(1.0) : Real(0.0);
      }
    }
    run_case("Interface pulse (left column only)", u_full, /*do_print=*/true);
  }

  // ---------- Conservation (u=1) ----------
  {
    Vector u1_full(ndofs_full); u1_full = 1.0;
    Vector u_split(ndofs_full);
    apply_perm(u1_full, u_split, P_full_to_split);

    Vector r1_full(ndofs_full), r1_split(ndofs_full), r1_split_as_full(ndofs_full);
    r1_full = 0.0; face_full(u1_full, r1_full);
    r1_split = 0.0; face_L(u_split, r1_split); face_R(u_split, r1_split); face_IF(u_split, r1_split);
    r1_split_as_full = 0.0; inv_apply_perm(r1_split, r1_split_as_full, P_full_to_split);

    Real sum_full = 0.0, sum_split = 0.0;
    { const Real* a = r1_full.ReadHostData();
      const Real* b = r1_split_as_full.ReadHostData();
      for (Integer g=0; g<ndofs_full; ++g) { sum_full += a[g]; sum_split += b[g]; } }
    std::cout << "  [Const] sum(r_full)=" << sum_full
              << "  sum(r_split)=" << sum_split << "\n";
  }

  return 0;
}
