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
#include <vector>
#include <string>

using namespace gendil;

namespace {

   template<typename T>
   inline T sqr(T x) { return x*x; }

} // namespace

int main(int, char**)
{
   // --------------------------
   // Config: 3D, split along x (Axis=0), no p-adaptivity
   // --------------------------
   constexpr Integer Dim = 3;
   const Integer nxL = 3, nxR = 3, ny = 3, nz = 3;   // cells per dim
   const Integer nx_full = nxL + nxR;
   constexpr Integer p = 2;                          // polynomial degree
   constexpr Integer q1d = p + 2;                    // quadrature chooser

   const Real hx = 1.0 / nxL;  // same h on both halves
   const Real hy = 1.0 / ny;
   const Real hz = 1.0 / nz;

   // --------------------------
   // Meshes
   // --------------------------
   Cartesian3DMesh mesh_full(hx, hy, hz, nx_full, ny, nz, Point<3>{0.0, 0.0, 0.0});
   Cartesian3DMesh meshL    (hx, hy, hz, nxL,     ny, nz, Point<3>{0.0, 0.0, 0.0});
   Cartesian3DMesh meshR    (hx, hy, hz, nxR,     ny, nz, Point<3>{1.0, 0.0, 0.0}); // shifted +1 in x

   // --------------------------
   // FE: no p-adaptivity
   // --------------------------
   auto fe = MakeLobattoFiniteElement(FiniteElementOrders<p,p,p>{});

   // Global layouts via L2Restriction
   L2Restriction res_full{0};
   auto fe_space_full = MakeFiniteElementSpace(mesh_full, fe, res_full);
   const Integer ndofs_full = fe_space_full.GetNumberOfFiniteElementDofs();

   L2Restriction resL{0};
   auto fe_space_L = MakeFiniteElementSpace(meshL, fe, resL);
   const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();

   L2Restriction resR{ndofsL}; // split layout is [u_L | u_R]
   auto fe_space_R = MakeFiniteElementSpace(meshR, fe, resR);
   const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

   if (ndofs_full != ndofsL + ndofsR) {
      std::cerr << "Layout mismatch: ndofs_full=" << ndofs_full
               << "  ndofsL+ndofsR=" << (ndofsL + ndofsR) << "\n";
      return 10;
   }

   // --------------------------
   // Integration rules (volume; face rules deduced internally)
   // --------------------------
   auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d>{});

   // --------------------------
   // Face meshes
   // --------------------------
   auto face_mesh_full = make_cartesian_interior_face_connectivity<Dim>(
         std::array<GlobalIndex,Dim>{(GlobalIndex)nx_full,(GlobalIndex)ny,(GlobalIndex)nz});
   auto face_mesh_L = make_cartesian_interior_face_connectivity<Dim>(
         std::array<GlobalIndex,Dim>{(GlobalIndex)nxL,(GlobalIndex)ny,(GlobalIndex)nz});
   auto face_mesh_R = make_cartesian_interior_face_connectivity<Dim>(
         std::array<GlobalIndex,Dim>{(GlobalIndex)nxR,(GlobalIndex)ny,(GlobalIndex)nz});

   CartesianIntermeshFaceConnectivity<Dim,0> iface(
         /*sizesR*/{(GlobalIndex)nxR,(GlobalIndex)ny,(GlobalIndex)nz},
         /*sizesL*/{(GlobalIndex)nxL,(GlobalIndex)ny,(GlobalIndex)nz});

   // --------------------------
   // Advection field (constant +ex for crisp upwind)
   // --------------------------
   auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>&, Real (&v)[Dim]) {
      v[0] = 1.0; v[1] = 0.0; v[2] = 0.0;
   };

   // --------------------------
   // Kernel policy
   // --------------------------
   #if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<q1d,q1d,q1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
   #else
   using KernelPolicy = SerialKernelConfiguration;
   #endif

   // --------------------------
   // Build FACE operators
   // --------------------------
   auto face_full = MakeAdvectionFaceOperator<KernelPolicy>(
         fe_space_full, face_mesh_full, int_rules, adv);

   auto face_L  = MakeAdvectionFaceOperator<KernelPolicy>(
         fe_space_L,   face_mesh_L,    int_rules, adv);
   auto face_R  = MakeAdvectionFaceOperator<KernelPolicy>(
         fe_space_R,   face_mesh_R,    int_rules, adv);
   auto face_IF = MakeAdvectionFaceOperator<KernelPolicy>(
         fe_space_R, fe_space_L, iface, int_rules, adv);

   // --------------------------
   // Build permutation P: FULL → SPLIT ([L | R])
   // Assumptions:
   //  - per-cell DOFs are contiguous,
   //  - local DOFs are lexicographic (a,b,c) in [0..p]^3 with x fastest,
   //  - global cells ordered lexicographically (i,j,k) with x fastest.
   // --------------------------
   const Integer nldofs = (p+1)*(p+1)*(p+1);

   auto cell_lin = [](Integer i, Integer j, Integer k, Integer nx, Integer ny)->Integer {
      return i + nx*(j + ny*k); // x-fastest
   };

   std::vector<Integer> P_full_to_split(ndofs_full);
   std::vector<Integer> P_split_to_full(ndofs_full);

   for (Integer k=0; k<nz; ++k)
      for (Integer j=0; j<ny; ++j)
         for (Integer i=0; i<nx_full; ++i) {
         const bool inL = (i < nxL);
         const Integer iLR = inL ? i : (i - nxL);

         const Integer cellF  = cell_lin(i,   j, k, nx_full, ny);
         const Integer cellLR = inL
            ? cell_lin(iLR, j, k, nxL, ny)
            : cell_lin(iLR, j, k, nxR, ny);

         const Integer baseF  = cellF  * nldofs;
         const Integer baseLR = (inL ? 0 : ndofsL) + cellLR * nldofs;

         Integer l = 0;
         for (Integer a=0; a<=p; ++a)
            for (Integer b=0; b<=p; ++b)
               for (Integer c=0; c<=p; ++c, ++l) {
               const Integer gF  = baseF  + l;
               const Integer gLR = baseLR + l;
               P_full_to_split[gF]  = gLR;
               P_split_to_full[gLR] = gF;
               }
         }

   #ifndef NDEBUG
   for (Integer i=0; i<ndofs_full; ++i) {
      GENDIL_ASSERT(P_split_to_full[ P_full_to_split[i] ] == i);
   }
   for (Integer j=0; j<ndofs_full; ++j) {
      GENDIL_ASSERT(P_full_to_split[ P_split_to_full[j] ] == j);
   }
   #endif

   auto apply_perm = [](const Vector& x, Vector& y,
                        const std::vector<Integer>& full_to_split) {
      // y_split[ full_to_split[i_full] ] = x_full[i_full]
      const Real* px = x.ReadHostData();
      Real*       py = y.ReadWriteHostData();
      const Integer n = (Integer)full_to_split.size();
      for (Integer i=0;i<n;++i) py[ full_to_split[i] ] = px[i];
   };

   auto inv_apply_perm = [](const Vector& x_split, Vector& y_full,
                              const std::vector<Integer>& full_to_split) {
      // y_full[i] = x_split[ full_to_split[i] ]
      const Real* px = x_split.ReadHostData();
      Real*       py = y_full.ReadWriteHostData();
      const Integer n = (Integer)full_to_split.size();
      for (Integer i=0;i<n;++i) py[i] = px[ full_to_split[i] ];
   };

   auto decode_full = [&](Integer g, Integer& i, Integer& j, Integer& k,
                           Integer& a, Integer& b, Integer& c) {
      const Integer cell = g / nldofs;
      const Integer l    = g % nldofs;
      i =  cell % nx_full;
      j = (cell / nx_full) % ny;
      k =  cell / (nx_full * ny);
      a =  l % (p+1);
      b = (l / (p+1)) % (p+1);
      c =  l / ((p+1)*(p+1));
   };

   auto header = [&](){
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "[Equivalence Test] Dim=" << Dim
               << "  p=" << p << "  q1d=" << q1d << "\n"
               << "  nx_full=" << nx_full << " (nxL=" << nxL << ", nxR=" << nxR << ")"
               << "  ny=" << ny << "  nz=" << nz << "\n"
               << "  ndofs_full=" << ndofs_full
               << "  ndofsL=" << ndofsL << "  ndofsR=" << ndofsR << "\n";
   };

   auto compare_and_print = [&](const std::string& label,
                                 const Vector& r_full_v,
                                 const Vector& r_split_as_full_v) -> bool {
      const Real* rf = r_full_v.ReadHostData();
      const Real* rs = r_split_as_full_v.ReadHostData();
      Real max_err = 0.0, l2_err = 0.0, l2_ref = 0.0, max_rf = 0.0, max_rs = 0.0;
      for (Integer g=0; g<ndofs_full; ++g) {
         const Real e = std::abs(rf[g] - rs[g]);
         max_err = std::max(max_err, e);
         l2_err += e*e;
         l2_ref += rf[g]*rf[g];
         max_rf = std::max(max_rf, std::abs(rf[g]));
         max_rs = std::max(max_rs, std::abs(rs[g]));
      }
      l2_err = std::sqrt(l2_err);
      l2_ref = std::sqrt(std::max(l2_ref, Real(1e-300)));
      const Real rel_l2 = l2_err / l2_ref;
      const Real eps = 200 * std::numeric_limits<Real>::epsilon();

      std::cout << "  [" << label << "]  max|Δr|=" << max_err
               << "  relL2=" << rel_l2
               << "  max|r_full|=" << max_rf
               << "  max|r_split|=" << max_rs << "\n";

      // Per–i-plane L2 of mismatch
      std::vector<Real> plane_l2(nx_full, 0.0), plane_cnt(nx_full, 0.0);
      for (Integer g=0; g<ndofs_full; ++g) {
         Integer I,J,K,A,B,C; decode_full(g,I,J,K,A,B,C);
         const Real e = rf[g] - rs[g];
         plane_l2[I] += e*e;
         plane_cnt[I] += 1.0;
      }
      std::cout << "  [Per-plane L2 of Δr along x]:\n    ";
      for (Integer I=0; I<nx_full; ++I) {
         const Real v = std::sqrt(plane_l2[I] / std::max(plane_cnt[I], 1.0));
         std::cout << v << (I+1<nx_full ? "  " : "\n");
      }

      // Face-DOF breakdown on the left interface column (i = nxL-1)
      Real sum_a0 = 0.0, sum_ap = 0.0; // L2 mismatch sums for a==0 and a==p
      for (Integer g=0; g<ndofs_full; ++g) {
         Integer I,J,K,A,B,C; decode_full(g,I,J,K,A,B,C);
         if (I == nxL-1) {
         const Real e2 = sqr(rf[g] - rs[g]);
         if (A == 0) sum_a0 += e2;
         if (A == p) sum_ap += e2;
         }
      }
      std::cout << "  [Left column i=nxL-1]  L2(Δ|a==0)=" << std::sqrt(sum_a0)
               << "  L2(Δ|a==p)=" << std::sqrt(sum_ap) << "\n";

      const bool pass = (max_err <= 1e-12 || rel_l2 <= eps);
      if (!pass) std::cout << "  RESULT: FAIL (" << label << " mismatch)\n";
      else       std::cout << "  RESULT: PASS (" << label << ")\n";
      return pass;
   };

   header();

   // ========== TEST 1: Random ==========
   Vector u_full(ndofs_full), u_split(ndofs_full);
   Vector r_full(ndofs_full), r_split(ndofs_full), r_split_as_full(ndofs_full);

   // deterministic pseudo-random fill
   {
      Real* uh = u_full.WriteHostData();
      uint32_t s = 0xC0FFEEu;
      auto rnd = [&](){
         s ^= s << 13; s ^= s >> 17; s ^= s << 5;
         return Real( (s & 0xFFFF) ) / Real(0xFFFF);
      };
      for (Integer i=0;i<ndofs_full;++i) uh[i] = rnd();
   }
   u_split = 0.0; apply_perm(u_full, u_split, P_full_to_split);

   r_full = 0.0; face_full(u_full, r_full);
   r_split = 0.0; face_L(u_split, r_split); face_R(u_split, r_split); face_IF(u_split, r_split);
   r_split_as_full = 0.0; inv_apply_perm(r_split, r_split_as_full, P_full_to_split);

   bool pass_random = compare_and_print("Random", r_full, r_split_as_full);

   // ========== TEST 2: Half-domain jump (u=1 on left half, 0 on right) ==========
   Vector u2_full(ndofs_full), u2_split(ndofs_full);
   {
      Real* uf = u2_full.WriteHostData();
      for (Integer g=0; g<ndofs_full; ++g) {
         Integer I,J,K,A,B,C; decode_full(g, I,J,K,A,B,C);
         uf[g] = (I < nxL) ? Real(1.0) : Real(0.0);
      }
   }
   u2_split = 0.0; apply_perm(u2_full, u2_split, P_full_to_split);

   Vector r2_full(ndofs_full), r2_split(ndofs_full), r2_split_as_full(ndofs_full);
   r2_full = 0.0; face_full(u2_full, r2_full);
   r2_split = 0.0; face_L(u2_split, r2_split); face_R(u2_split, r2_split); face_IF(u2_split, r2_split);
   r2_split_as_full = 0.0; inv_apply_perm(r2_split, r2_split_as_full, P_full_to_split);

   bool pass_jump = compare_and_print("Half-jump (L=1,R=0)", r2_full, r2_split_as_full);

   // ========== TEST 3: Interface pulse (only interface-adjacent cells active) ==========
   // Left column at i=nxL-1: set u=1 on those cells only; right column i=nxL: u=0
   Vector u3_full(ndofs_full), u3_split(ndofs_full);
   {
      Real* uf = u3_full.WriteHostData();
      for (Integer g=0; g<ndofs_full; ++g) {
         Integer I,J,K,A,B,C; decode_full(g, I,J,K,A,B,C);
         uf[g] = (I == nxL-1) ? Real(1.0) : Real(0.0);
      }
   }
   u3_split = 0.0; apply_perm(u3_full, u3_split, P_full_to_split);

   Vector r3_full(ndofs_full), r3_split(ndofs_full), r3_split_as_full(ndofs_full);
   r3_full = 0.0; face_full(u3_full, r3_full);
   r3_split = 0.0; face_L(u3_split, r3_split); face_R(u3_split, r3_split); face_IF(u3_split, r3_split);
   r3_split_as_full = 0.0; inv_apply_perm(r3_split, r3_split_as_full, P_full_to_split);

   bool pass_pulse = compare_and_print("Interface pulse (left column only)", r3_full, r3_split_as_full);

   // --------------------------
   // Conservation sanity (constant state) on FULL vs SPLIT
   // --------------------------
   Vector u1(ndofs_full), r1_full(ndofs_full), r1_split(ndofs_full), r1_split_as_full(ndofs_full);
   u1 = 1.0;
   r1_full = 0.0; face_full(u1, r1_full);

   Vector u1_split(ndofs_full);
   u1_split = 0.0; apply_perm(u1, u1_split, P_full_to_split);
   r1_split = 0.0; face_L(u1_split, r1_split); face_R(u1_split, r1_split); face_IF(u1_split, r1_split);
   r1_split_as_full = 0.0; inv_apply_perm(r1_split, r1_split_as_full, P_full_to_split);

   Real sum_full = 0.0, sum_split = 0.0;
   { const Real* a = r1_full.ReadHostData();
      const Real* b = r1_split_as_full.ReadHostData();
      for (Integer g=0; g<ndofs_full; ++g) { sum_full += a[g]; sum_split += b[g]; } }

   std::cout << "  [Const]   sum(r_full)=" << sum_full
               << "  sum(r_split)=" << sum_split << "\n";
   bool pass_cons = (std::abs(sum_full) <= 1e-10 && std::abs(sum_split) <= 1e-10);
   std::cout << "  RESULT: " << (pass_cons ? "PASS" : "FAIL") << " (conservation)\n";

   // Final verdict
   if (pass_random && pass_jump && pass_pulse && pass_cons) {
      std::cout << "OK: Equivalence across single vs split meshes.\n";
      return 0;
   }
   return 20;
}
