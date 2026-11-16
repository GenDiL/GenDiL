// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
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

static inline Integer lin_cell_3d(Integer i, Integer j, Integer k,
                                  Integer nx, Integer ny) {
  // x-fastest
  return i + nx*(j + ny*k);
}

// Independent trace assembly for the P1/P3 fixture. Four-point Gauss quadrature
// exactly integrates the degree-six face products, without using operator,
// connectivity, finite-element evaluation, or quadrature implementation helpers.
template<Integer P>
Real TraceBasis(const Integer i, const Real x)
{
  static_assert(P == 1 || P == 3);
  if constexpr (P == 1)
    return i == 0 ? 1.0 - x : x;
  else
  {
    const Real a = 1.0 / std::sqrt(5.0);
    const std::array<Real, 4> nodes{0.0, (1.0-a)/2.0, (1.0+a)/2.0, 1.0};
    Real value = 1.0;
    for (Integer j = 0; j <= P; ++j)
      if (j != i) value *= (x-nodes[j])/(nodes[i]-nodes[j]);
    return value;
  }
}

template<Integer PLeft, Integer PRight>
void AddJumpDissipationFace(
    const Real* u, std::vector<Real>& result,
    const Integer left_base, const Integer right_base,
    const Real area, const Real left_y_scale = 1.0,
    const Real left_y_origin = 0.0)
{
  constexpr std::array<Real, 4> points{
    0.06943184420297371, 0.33000947820757187,
    0.66999052179242813, 0.93056815579702629};
  constexpr std::array<Real, 4> weights{
    0.17392742256872693, 0.32607257743127307,
    0.32607257743127307, 0.17392742256872693};
  for (Integer qz = 0; qz < points.size(); ++qz)
    for (Integer qy = 0; qy < points.size(); ++qy)
    {
      const Real y_left = left_y_origin + left_y_scale*points[qy];
      Real jump = 0.0;
      for (Integer k = 0; k <= PLeft; ++k)
        for (Integer j = 0; j <= PLeft; ++j)
          jump += u[left_base + PLeft + (PLeft+1)*(j+(PLeft+1)*k)]
                * TraceBasis<PLeft>(j, y_left)*TraceBasis<PLeft>(k, points[qz]);
      for (Integer k = 0; k <= PRight; ++k)
        for (Integer j = 0; j <= PRight; ++j)
          jump -= u[right_base + (PRight+1)*(j+(PRight+1)*k)]
                * TraceBasis<PRight>(j, points[qy])*TraceBasis<PRight>(k, points[qz]);
      const Real flux = area*weights[qy]*weights[qz]*jump;
      for (Integer k = 0; k <= PLeft; ++k)
        for (Integer j = 0; j <= PLeft; ++j)
          result[left_base + PLeft + (PLeft+1)*(j+(PLeft+1)*k)]
            += flux*TraceBasis<PLeft>(j, y_left)*TraceBasis<PLeft>(k, points[qz]);
      for (Integer k = 0; k <= PRight; ++k)
        for (Integer j = 0; j <= PRight; ++j)
          result[right_base + (PRight+1)*(j+(PRight+1)*k)]
            -= flux*TraceBasis<PRight>(j, points[qy])*TraceBasis<PRight>(k, points[qz]);
    }
}

} // namespace

int main(int, char**)
{
  std::cout.setf(std::ios::fixed); std::cout<<std::setprecision(6);
  bool success = true;

  // --------------------------
  // Config: 3D, split along +x
  // --------------------------
  constexpr Integer Dim = 3;
  constexpr Integer Axis = 0;          // split direction (x)
  // Local face index for the minus side: +x face so that n^- = +e_x
  // In 3D, faces 0,1,2 → (-x,-y,-z); faces 3,4,5 → (+x,+y,+z)
  constexpr Integer LFI = Axis + Dim;

  // Coarse (minus) vs refined (plus) sizes
  const Integer nxL = 2, nyL = 2, nzL = 2;
  const Integer nxR = 2, nyR = 4, nzR = 2;      // ry = 2 in y, rz = 1 in z
  // const Integer nx_full = nxL + nxR;

  // Polynomial degrees (p-nonconforming)
  constexpr Integer pL = 1;
  constexpr Integer pR = 3;

  // Quadrature: pick something that’s safe for both sides
  constexpr Integer q1d = (pL > pR ? pL : pR) + 2;

  // Grid spacings so the two blocks tile [0,2]×[0,1]×[0,1]
  const Real hx = 1.0 / nxL;        // both blocks use same hx
  const Real hyL = 1.0 / nyL;
  const Real hyR = 1.0 / nyR;       // refined in y on the plus side
  const Real hz = 1.0 / nzL;

  // --------------------------
  // Meshes
  // --------------------------
  Cartesian3DMesh meshL(hx, hyL, hz, nxL, nyL, nzL, Point<3>{0.0, 0.0, 0.0});
  Cartesian3DMesh meshR(hx, hyR, hz, nxR, nyR, nzR, Point<3>{1.0, 0.0, 0.0}); // shift +1 in x

  // --------------------------
  // Finite elements (p-nonconforming)
  // --------------------------
  auto feL = MakeLobattoFiniteElement(FiniteElementOrders<pL,pL,pL>{});
  auto feR = MakeLobattoFiniteElement(FiniteElementOrders<pR,pR,pR>{});

  // Layout: split vector = [L | R]
  ContiguousL2RestrictionSpecification resL{0};
  auto fe_space_L = MakeFiniteElementSpace(meshL, feL, resL);
  const Integer ndofsL = GetNumberOfGlobalDofs( fe_space_L );

  ContiguousL2RestrictionSpecification resR{ndofsL};
  auto fe_space_R = MakeFiniteElementSpace(meshR, feR, resR);
  const Integer ndofsR = GetNumberOfGlobalDofs( fe_space_R );

  const Integer ndofs_split = ndofsL + ndofsR;

  // --------------------------
  // Integration rules (volume; face rules deduced internally)
  // --------------------------
  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d>{});

  // Interior face meshes (each block)
  auto face_mesh_L = MakeCartesianInteriorFaceConnectivity<Dim>(
      std::array<GlobalIndex,Dim>{
        (GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL});
  auto face_mesh_R = MakeCartesianInteriorFaceConnectivity<Dim>(
      std::array<GlobalIndex,Dim>{
        (GlobalIndex)nxR,(GlobalIndex)nyR,(GlobalIndex)nzR});

  // Nonconforming intermesh (minus=coarse L, plus=fine R) using LocalFaceIndex
  NonconformingCartesianIntermeshFaceConnectivity<Dim, LFI> iface_hp(
      /*coarse*/{(GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL},
      /*fine  */{(GlobalIndex)nxR,(GlobalIndex)nyR,(GlobalIndex)nzR});

  // --------------------------
  // Advection β = (+1, 0, 0) and (−1, 0, 0)
  // --------------------------
  auto adv_pos = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>&, Real (&v)[Dim]) {
    v[0] = 1.0; v[1] = 0.0; v[2] = 0.0;
  };
  auto adv_neg = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>&, Real (&v)[Dim]) {
    v[0] = -1.0; v[1] = 0.0; v[2] = 0.0;
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
  // Operators
  // --------------------------
  auto face_L_pos  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, face_mesh_L, int_rules, adv_pos);
  auto face_R_pos  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_R, face_mesh_R, int_rules, adv_pos);
  auto face_IF_pos = MakeAdvectionFaceOperator<KernelPolicy>(
                       fe_space_L, fe_space_R, iface_hp, int_rules, adv_pos);

  auto face_L_neg  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, face_mesh_L, int_rules, adv_neg);
  auto face_R_neg  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_R, face_mesh_R, int_rules, adv_neg);
  auto face_IF_neg = MakeAdvectionFaceOperator<KernelPolicy>(
                       fe_space_L, fe_space_R, iface_hp, int_rules, adv_neg);

  auto apply_all = [&](auto&& faceL, auto&& faceR, auto&& faceIF,
                      const Vector& u_split, Vector& r_split){
    r_split = 0.0;
    faceL(u_split, r_split);
    faceR(u_split, r_split);
    faceIF(u_split, r_split);
  };

  std::cout << "[3D h+p nonconforming] pL=" << pL << " pR=" << pR
            << "  q1d=" << q1d
            << "  nxL=" << nxL << " nyL=" << nyL << " nzL=" << nzL
            << "  nxR=" << nxR << " nyR=" << nyR << " nzR=" << nzR << "\n"
            << "  ndofsL=" << ndofsL << "  ndofsR=" << ndofsR
            << "  ndofs_split=" << ndofs_split << "\n";

  // --------------------------
  // (1) Constant → conservation (sum ≈ 0)
  // --------------------------
  {
    Vector u(ndofs_split), r(ndofs_split);
    u = 1.0;
    apply_all(face_L_pos, face_R_pos, face_IF_pos, u, r);
    Real sum = 0.0;
    const Real* pr = r.ReadHostData();
    for (Integer g=0; g<ndofs_split; ++g) sum += pr[g];
    const bool passed = std::isfinite(sum) && std::abs(sum) <= 1e-12;
    success = passed && success;
    std::cout << "  [Const u=1] sum(r) = " << sum << "  → "
              << (passed ? "PASS" : "FAIL") << "\n\n";
  }

  // --------------------------
  // (2) r=1 fallback (turn off h-nonconforming): compare NC iface vs conforming iface
  //     This should match even when pL != pR (mapping = identity).
  // --------------------------
  {
    // Make a "same partition" right mesh
    Cartesian3DMesh meshR1(hx, 1.0/nyL, hz, nxR, nyL, nzL, Point<3>{1.0,0.0,0.0});
    auto fe_space_R1 = MakeFiniteElementSpace(meshR1, feR, ContiguousL2RestrictionSpecification{ndofsL});

    // Nonconforming connectivity with refine=1 (degenerates to identity)
    NonconformingCartesianIntermeshFaceConnectivity<Dim, LFI> iface_r1(
        {(GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL},
        {(GlobalIndex)nxR,(GlobalIndex)nyL,(GlobalIndex)nzL});

    // Conforming connectivity
    CartesianIntermeshFaceConnectivity<Dim, LFI> iface_conf(
        {(GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL},
        {(GlobalIndex)nxR,(GlobalIndex)nyL,(GlobalIndex)nzL});

    auto face_R1_pos   = MakeAdvectionFaceOperator<KernelPolicy>(
                            fe_space_R1,
                            MakeCartesianInteriorFaceConnectivity<Dim>(
                              std::array<GlobalIndex,Dim>{
                                (GlobalIndex)nxR,(GlobalIndex)nyL,(GlobalIndex)nzL}),
                            int_rules, adv_pos);
    auto face_IF_nc    = MakeAdvectionFaceOperator<KernelPolicy>(
                            fe_space_L, fe_space_R1, iface_r1,  int_rules, adv_pos);
    auto face_IF_conf  = MakeAdvectionFaceOperator<KernelPolicy>(
                            fe_space_L, fe_space_R1, iface_conf, int_rules, adv_pos);

    const Integer ndofsR1 = GetNumberOfGlobalDofs( fe_space_R1 );
    Vector u(ndofsL + ndofsR1), r_nc(u.Size()), r_cf(u.Size());
    {
      Real* pu = u.WriteHostData();
      // deterministic-ish fill
      for (Integer g=0; g<u.Size(); ++g) pu[g] = (Real(g%11)-5.0) / 7.0;
    }

    r_nc = 0.0; face_L_pos(u, r_nc);  face_R1_pos(u, r_nc);  face_IF_nc(u, r_nc);
    r_cf = 0.0; face_L_pos(u, r_cf);  face_R1_pos(u, r_cf);  face_IF_conf(u, r_cf);

    Real l2 = 0.0;
    const Real *a = r_nc.ReadHostData(), *b = r_cf.ReadHostData();
    for (Integer g=0; g<r_nc.Size(); ++g) l2 += sqr(a[g]-b[g]);
    l2 = std::sqrt(l2);
    const bool passed = std::isfinite(l2) && l2 <= 1e-12;
    success = passed && success;
    std::cout << "[r=1 fallback vs conforming] ||Δ||_2 = " << l2
              << "  → " << (passed ? "PASS" : "FAIL") << "\n\n";
  }

  // --------------------------
  // (3) Reversing beta preserves upwind dissipation:
  // r(+beta) + r(-beta) = integral |beta.n| jump(u) jump(v).
  // --------------------------
  {
    Vector u(ndofs_split), r_pos(ndofs_split), r_neg(ndofs_split);
    // Random-but-deterministic fill
    {
      Real* pu = u.WriteHostData();
      uint32_t s = 0xC0FFEEu;
      auto rnd = [&](){ s^= s<<13; s^= s>>17; s^= s<<5; return Real((s & 0xFFFF)) / Real(0xFFFF); };
      for (Integer g=0; g<ndofs_split; ++g) pu[g] = rnd();
    }
    apply_all(face_L_pos, face_R_pos, face_IF_pos, u, r_pos);
    apply_all(face_L_neg, face_R_neg, face_IF_neg, u, r_neg);

    std::vector<Real> expected(ndofs_split, 0.0);
    const Real* pu = u.ReadHostData();
    constexpr Integer nldofsL = (pL+1)*(pL+1)*(pL+1);
    constexpr Integer nldofsR = (pR+1)*(pR+1)*(pR+1);
    // Only x-normal faces contribute, since beta=(1,0,0).
    for (Integer k=0; k<nzL; ++k)
      for (Integer j=0; j<nyL; ++j)
        for (Integer i=0; i+1<nxL; ++i)
          AddJumpDissipationFace<pL,pL>(pu, expected,
            lin_cell_3d(i,j,k,nxL,nyL)*nldofsL,
            lin_cell_3d(i+1,j,k,nxL,nyL)*nldofsL, hyL*hz);
    for (Integer k=0; k<nzR; ++k)
      for (Integer j=0; j<nyR; ++j)
        for (Integer i=0; i+1<nxR; ++i)
          AddJumpDissipationFace<pR,pR>(pu, expected,
            ndofsL+lin_cell_3d(i,j,k,nxR,nyR)*nldofsR,
            ndofsL+lin_cell_3d(i+1,j,k,nxR,nyR)*nldofsR, hyR*hz);
    for (Integer k=0; k<nzR; ++k)
      for (Integer j=0; j<nyR; ++j)
        AddJumpDissipationFace<pL,pR>(pu, expected,
          lin_cell_3d(nxL-1,j/2,k,nxL,nyL)*nldofsL,
          ndofsL+lin_cell_3d(0,j,k,nxR,nyR)*nldofsR,
          hyR*hz, 0.5, 0.5*Real(j%2));

    Real error2 = 0.0, expected_norm2 = 0.0;
    const Real *rp = r_pos.ReadHostData(), *rn = r_neg.ReadHostData();
    for (Integer g=0; g<ndofs_split; ++g)
    {
      error2 += sqr(rp[g] + rn[g] - expected[g]);
      expected_norm2 += sqr(expected[g]);
    }
    const Real error = std::sqrt(error2);
    const Real expected_norm = std::sqrt(expected_norm2);
    const bool passed = std::isfinite(error) && std::isfinite(expected_norm)
      && expected_norm > 1e-8 && error <= 1e-12*std::max(Real{1}, expected_norm);
    success = passed && success;
    std::cout << "[β reversal dissipation] reference norm = " << expected_norm
              << ", ||r(+β) + r(−β) - reference||_2 = " << error
              << "  → " << (passed ? "PASS" : "FAIL") << "\n\n";
  }

  // --------------------------
  // (4) Interface pulse on coarse-left column: nonzero and conservative.
  // --------------------------
  {
    const Integer nldofsL = (pL+1)*(pL+1)*(pL+1);
    Vector u(ndofs_split), r(ndofs_split);
    u = 0.0;

    // Set u=1 in left-block cells on the interface column i=nxL-1
    {
      Real* pu = u.ReadWriteHostData();
      for (Integer k=0; k<nzL; ++k)
        for (Integer j=0; j<nyL; ++j) {
          const Integer i = nxL-1;
          const Integer base = /*L offset*/ 0
              + lin_cell_3d(i,j,k, nxL,nyL) * nldofsL;
          for (Integer a=0; a<=pL; ++a)
            for (Integer b=0; b<=pL; ++b)
              for (Integer c=0; c<=pL; ++c) {
                const Integer l = a + (pL+1)*(b + (pL+1)*c);
                pu[ base + l ] = 1.0;
              }
        }
    }

    apply_all(face_L_pos, face_R_pos, face_IF_pos, u, r);

    // Report a tiny snapshot: max magnitude on minus/plus interface slices
    Real max_minus = 0.0, sum_all = 0.0;
    const Real* pr = r.ReadHostData();
    // minus interface slice (i=nxL-1)
    for (Integer k=0; k<nzL; ++k)
      for (Integer j=0; j<nyL; ++j) {
        const Integer base = 0 + lin_cell_3d(nxL-1,j,k, nxL,nyL) * nldofsL;
        for (Integer l=0; l<nldofsL; ++l) max_minus = std::max(max_minus, std::abs(pr[base+l]));
      }

    // quick global sum
    for (Integer g=0; g<ndofs_split; ++g) sum_all += pr[g];
    const bool passed = std::isfinite(sum_all) && std::isfinite(max_minus)
      && std::abs(sum_all) <= 1e-12 && max_minus > 1e-8;
    success = passed && success;

    // print
    std::cout << "[Interface pulse]  max|r| on minus slice = " << max_minus
              << "   sum(r) = " << sum_all
              << "  → " << (passed ? "PASS" : "FAIL") << "\n";
  }

  return success ? 0 : 1;
}
