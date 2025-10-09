// Copyright GenDiL Project Developers
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
template<typename T> inline T sqr(T x){ return x*x; }
template<typename V> bool is_finite_vec(const V& v){
  const Real* p = v.ReadHostData();
  for (Integer i=0;i<v.Size();++i) if (!std::isfinite(p[i])) return false;
  return true;
}
} // namespace

int main(int, char**)
{
   std::cout.setf(std::ios::fixed); std::cout<<std::setprecision(6);

   // ----------------------------------------------------------
   // Config: Euler on 3D (x), Vlasov on 6D (x,v); test +x / -x
   // ----------------------------------------------------------
   constexpr Integer LowDim  = 3;
   constexpr Integer HighDim = 6;
   static_assert(HighDim == LowDim + LowDim, "Expect 3x3V = 6D for this test");

   constexpr Integer Axis = 0;  // x
   constexpr Integer LFI = Axis + LowDim; // +x face index on the low side

   // cell counts in x and v
   const Integer nx=3, ny=2, nz=2;
   const Integer nvx=4, nvy=3, nvz=3;

   // FE degrees (feel free to raise)
   constexpr Integer pX = 1;    // Euler space degree
   constexpr Integer pF = 1;    // Vlasov space degree

   // 1D quadrature for the *volume* rule (face rules are derived internally).
   // Keep it small: this is a smoke/robustness test.
   constexpr Integer qx  = pX + 2;
   constexpr Integer qv  = pF + 2;

   // -------
   // Meshes
   // -------
   // Euler 3D in [0,1]^3
   const std::array<GlobalIndex,3> sizesX{ (GlobalIndex)nx,
                                          (GlobalIndex)ny,
                                          (GlobalIndex)nz };
   const std::array<Real,3>        hX{ 1.0/nx, 1.0/ny, 1.0/nz };
   const Point<3>                  x0{ 0.0, 0.0, 0.0 };
   // Use your concrete type (e.g., Cartesian3DMesh or CartesianMesh<3>)
   CartesianMesh<3> meshX(sizesX, hX, x0);

   // Vlasov 6D in [0,1]^3_x × [-3,3]^3_v
   const std::array<GlobalIndex,6> sizesXV{ (GlobalIndex)nx,
                                          (GlobalIndex)ny,
                                          (GlobalIndex)nz,
                                          (GlobalIndex)nvx,
                                          (GlobalIndex)nvy,
                                          (GlobalIndex)nvz };
   const std::array<Real,6>        hXV{ 1.0/nx, 1.0/ny, 1.0/nz,
                                       6.0/nvx, 6.0/nvy, 6.0/nvz };
   const Point<6>                  x0v{ 1.0, 0.0, 0.0,  -3.0, -3.0, -3.0 };
   // Use your concrete type (e.g., Cartesian6DMesh or CartesianMesh<6>)
   CartesianMesh<6> meshXV(sizesXV, hXV, x0v);

   // ----------------------------------------------------------
   // Finite element spaces (L2/Qk)
   // ----------------------------------------------------------
   auto scalar_fe_x = MakeLobattoFiniteElement(FiniteElementOrders<pX,pX,pX>{});
   // (rho, mom[3], rhoE) for Euler
   auto feX  = MakeVectorFiniteElement( scalar_fe_x, scalar_fe_x, scalar_fe_x, scalar_fe_x, scalar_fe_x );
   auto feXV = MakeLobattoFiniteElement(FiniteElementOrders<pF,pF,pF,pF,pF,pF>{});

   // Layout: separate vectors (not concatenated)
   auto fe_space_X  = MakeFiniteElementSpace(meshX,  feX);
   auto fe_space_XV = MakeFiniteElementSpace(meshXV, feXV);

   const Integer ndofsX  = fe_space_X .GetNumberOfFiniteElementDofs();
   const Integer ndofsXV = fe_space_XV.GetNumberOfFiniteElementDofs();

   // ----------------------------------------------------------
   // Cross-dimensional face meshes (+x and -x)
   // Tangential sizes (y,z in x; and all v) must match by construction.
   // ----------------------------------------------------------
   CartesianCrossDimConformingIntermesh<LowDim,HighDim,LFI> iface(
         { (GlobalIndex)nx,(GlobalIndex)ny,(GlobalIndex)nz },
         { (GlobalIndex)nx,(GlobalIndex)ny,(GlobalIndex)nz,
         (GlobalIndex)nvx,(GlobalIndex)nvy,(GlobalIndex)nvz } );

   // ----------------------------------------------------------
   // Integration rule (volume). Face rules are deduced internally.
   // ----------------------------------------------------------
   auto int_rule_x = MakeIntegrationRule(
         IntegrationRuleNumPoints<qx,qx,qx>{});
   auto int_rule_xv = MakeIntegrationRule(
         IntegrationRuleNumPoints<qx,qx,qx, qv,qv,qv>{});

   // ----------------------------------------------------------
   // Kernel policy
   // ----------------------------------------------------------
#if defined(GENDIL_USE_DEVICE)
   // Keep thread layout tiny; the operator manages shared memory internally.
   using ThreadLayout = ThreadBlockLayout<qx,qx,qx>;
   constexpr size_t NumSharedDimensions = LowDim; // shared for low-face accumulation
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // ----------------------------------------------------------
   // Interface operator
   // ----------------------------------------------------------
   auto op = MakeEulerVlasovCrossDimFaceOperator<KernelPolicy>(
      fe_space_X, fe_space_XV, iface,  int_rule_x, int_rule_xv /*, gamma, Rgas (defaults)*/);

   // ----------------------------------------------------------
   // Allocate vectors:
   // Euler: rho, mom[3], rhoE.  Vlasov: f
   // ----------------------------------------------------------
   Vector u(ndofsX), f(ndofsXV);

   Vector u_out(ndofsX), f_out(ndofsXV);

   std::cout << "[Euler–Vlasov cross-dim]  "
               << "pX=" << pX << " pF=" << pF
               << "  qx=" << qx << " qv=" << qv
               << "  nx=" << nx << " ny=" << ny << " nz=" << nz
               << "  nvx=" << nvx << " nvy=" << nvy << " nvz=" << nvz << "\n"
               << "  ndofsX=" << ndofsX << "  ndofsXV=" << ndofsXV << "\n";

   // Helpers to build views (Euler has vector momentum)
   auto make_views_in  = [&](Vector& u_, Vector& f_){
      auto u_in = MakeReadOnlyEVectorView<KernelPolicy>(fe_space_X, u_);
      auto f_in = MakeReadOnlyEVectorView<KernelPolicy>(fe_space_XV, f_);
      return std::tuple(u_in, f_in);
   };
   auto make_views_out = [&](Vector& u_, Vector& f_){
      auto u_out_v = MakeWriteOnlyEVectorView<KernelPolicy>(fe_space_X, u_);
      auto f_out_v = MakeReadWriteEVectorView<KernelPolicy>(fe_space_XV, f_);
      return std::tuple(u_out_v, f_out_v);
   };

   auto apply = [&](auto& OP,
                     Vector& u_i, Vector& f_i,
                     Vector& u_o, Vector& f_o)
   {
      u_o = 0.0; f_o = 0.0;
      auto [u_in, f_in] = make_views_in(u_i, f_i);
      auto [u_out_v, f_out_v] = make_views_out(u_o, f_o);
      OP.Apply( u_in, f_in, u_out_v, f_out_v );
   };

   // ---------------------------------------------
   // (1) Zero → zero (strict)
   // ---------------------------------------------
   {
      u = 0.0; f = 0.0;
      u_out = 0.0; f_out = 0.0;

      apply(op, u, f, u_out, f_out);

      auto norm = [&](const Vector& v)
      {
         const Real* p=v.ReadHostData();
         Real s=0;
         for(Integer i=0;i<v.Size();++i) s+=p[i]*p[i];
         return std::sqrt(s);
      };
      const Real s_u  = norm(u_out);
      const Real s_f  = norm(f_out);

      const bool pass = (std::abs(s_u )<=1e-14) &&
                        (std::abs(s_f)<=1e-14);

      std::cout << "[Zero → Zero] norms: "
               << " u="<<s_u
               << " f="<<s_f
               << "  → " << (pass ? "PASS" : "FAIL") << "\n\n";
   }

   // ---------------------------------------------
   // (2) One-sided source: Euler=const, Vlasov=0
   // Expect non-zero transfer towards phase space
   // ---------------------------------------------
   {
   // Fill Euler vector and Vlasov vector
   u = 1.0;     // simple constant (all components) – OK for a smoke test
   f = 1.0;

   u_out = 0.0; f_out = 0.0;
   apply(op, u, f, u_out, f_out);

   // Quick diagnostics: norms and sums (signs should oppose)
   auto l2 = [](const Vector& v){
      const Real* p = v.ReadHostData();
      Real s = 0; for (Integer i=0;i<v.Size();++i) s += p[i]*p[i];
      return std::sqrt(s);
   };
   auto sum = [](const Vector& v){
      const Real* p = v.ReadHostData();
      Real s = 0; for (Integer i=0;i<v.Size();++i) s += p[i];
      return s;
   };

   const Real l2_u  = l2(u_out);
   const Real l2_f  = l2(f_out);
   const Real s_u   = sum(u_out);
   const Real s_f   = sum(f_out);

   const bool nonzero_phase = (l2_f > 1e-12);
   const bool opposite_sgn  = (s_u * s_f) < 0.0; // aggregate signs oppose

   std::cout << "[One-sided source]  ||u_out||=" << l2_u
               << "  ||f_out||=" << l2_f
               << "  sum(u_out)=" << s_u
               << "  sum(f_out)=" << s_f
               << "  → non-zero:" << (nonzero_phase ? "PASS" : "FAIL")
               << ", opposite signs:" << (opposite_sgn ? "PASS" : "FAIL")
               << "\n\n";
   }


   // ---------------------------------------------
   // (3) Random robustness: outputs finite
   // ---------------------------------------------
   {
      uint32_t s = 0xBADC0DEu;
      auto rnd = [&](){ s^= s<<13; s^= s>>17; s^= s<<5; return Real((s & 0x7FFF)) / Real(0x7FFF); };
      {
         Real* pu  = u.ReadWriteHostData();
         Real* pf  = f.ReadWriteHostData();
         for (Integer i=0;i<u.Size(); ++i)  pu[i]  = rnd();
         for (Integer i=0;i<f.Size(); ++i)  pf[i]  = rnd();
      }

      u_out = 0.0; f_out = 0.0;
      apply(op, u, f, u_out, f_out);

      const bool pass = is_finite_vec(u_out) && is_finite_vec(f_out);
      std::cout << "[Random robustness] outputs finite → "
               << (pass ? "PASS" : "FAIL") << "\n";
   }

   return 0;
}
