// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <chrono>
#include <sstream>
#include <iostream>
#include <limits>

using namespace gendil;

// ---------- helpers ----------

static inline long long interior_faces_3d(int nx,int ny,int nz){
  // total interior faces across the three orientations
  const long long fx = 1LL*(nx-1)*ny*nz;
  const long long fy = 1LL*nx*(ny-1)*nz;
  const long long fz = 1LL*nx*ny*(nz-1);
  return fx + fy + fz;
}

// time a single operator; returns seconds per application
template<class Op>
double time_operator(Op& op, Vector& in, Vector& out, int iters=6){
  // one warmup
  op(in, out);
  GENDIL_DEVICE_SYNC;
  // measure
  const auto t0 = std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++){
    op(in, out);
  }
  GENDIL_DEVICE_SYNC;
  const auto t1 = std::chrono::steady_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;
  return dt.count() / double(iters);
}

// Build & time the three components for one (nyL,nzL) and refinement (ry,rz)
// axis = 0 split (x), minus=L (coarse), plus=R (fine)
template<int p, int q1d, int ry, int rz>
void bench_face_components_once_3D(int nx_half, int nyL, int nzL,
                                   double& tL, double& tR, double& tIF,
                                   long long& facesL, long long& facesR, long long& facesIF,
                                   long long& ndofsL, long long& ndofsR)
{
  constexpr int Dim = 3;
  static_assert(ry>=1 && rz>=1, "refinement must be >= 1");

  const int nxL = nx_half;
  const int nxR = nx_half;                 // same along splitting axis
  const int nyR = ry*nyL;
  const int nzR = rz*nzL;

  const Real hx = 1.0 / nxL;               // same physical size on each half in x
  const Real hyL = 1.0 / nyL;
  const Real hzL = 1.0 / nzL;
  const Real hyR = 1.0 / nyR;
  const Real hzR = 1.0 / nzR;

  // meshes: L at [0,1]×[0,1]×[0,1], R shifted by +1 in x
  Cartesian3DMesh meshL(hx, hyL, hzL, nxL, nyL, nzL, Point<3>{0.0,0.0,0.0});
  Cartesian3DMesh meshR(hx, hyR, hzR, nxR, nyR, nzR, Point<3>{1.0,0.0,0.0});

  // FE (same p on both sides so we isolate h-adaptivity)
  auto fe = MakeLobattoFiniteElement(FiniteElementOrders<p,p,p>{});

  // split layout: [L | R]
  L2Restriction resL{0};
  auto feL = MakeFiniteElementSpace(meshL, fe, resL);
  const Integer dofsL = feL.GetNumberOfFiniteElementDofs();

  L2Restriction resR{dofsL};
  auto feR = MakeFiniteElementSpace(meshR, fe, resR);
  const Integer dofsR = feR.GetNumberOfFiniteElementDofs();

  ndofsL = dofsL;
  ndofsR = dofsR;

  // integration rules (volume → face deduced)
  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d>{});

  // interior face meshes
  auto face_mesh_L = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL});
  auto face_mesh_R = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxR,(GlobalIndex)nyR,(GlobalIndex)nzR});

  // nonconforming interface: minus = L coarse on +x face (LocalFaceIndex = Dim)
  // plus = R fine on -x face
  NonconformingCartesianIntermeshFaceConnectivity<Dim, /*LocalFaceIndex=*/Dim> iface(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL},
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxR,(GlobalIndex)nyR,(GlobalIndex)nzR});

  // advection β = (1,1,1) (any constant is fine; no special casing)
  auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real,Dim>&, Real (&v)[Dim]){
    v[0]=1.0; v[1]=1.0; v[2]=1.0;
  };

#if defined(GENDIL_USE_DEVICE)
  using ThreadLayout  = ThreadBlockLayout<q1d,q1d,q1d>;
  constexpr size_t NumSharedDimensions = Dim;
  using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
  using KernelPolicy = SerialKernelConfiguration;
#endif

  // operators
  auto face_L  = MakeAdvectionFaceOperator<KernelPolicy>(feL, face_mesh_L, int_rules, adv);
  auto face_R  = MakeAdvectionFaceOperator<KernelPolicy>(feR, face_mesh_R, int_rules, adv);
  auto face_IF = MakeAdvectionFaceOperator<KernelPolicy>(feL, feR, iface, int_rules, adv);

  // inputs / outputs
  Vector u(dofsL + dofsR), r(dofsL + dofsR);
  // deterministic fill
  {
    Real* pu = u.ReadWriteHostData();
    for (Integer g=0; g<u.Size(); ++g) pu[g] = Real((g % 7) - 3) / 3.0;
  }
  r = 0.0;

  // time each component separately
  tL  = time_operator(face_L,  u, r);
  tR  = time_operator(face_R,  u, r);
  tIF = time_operator(face_IF, u, r);

  // face counts for x-axis & conversion to faces/sec
  facesL  = interior_faces_3d(nxL, nyL, nzL);
  facesR  = interior_faces_3d(nxR, nyR, nzR);
  facesIF = 1LL * nyR * nzR; // subfaces on the interface slab
}

// One sweep: keep nx fixed and grow ny,nz (alternating), writing (faces, faces/s)
template<int p, int q_extra, int ry, int rz>
void sweep_h_adaptivity(std::ostringstream& outL,
                        std::ostringstream& outR,
                        std::ostringstream& outIF)
{
  constexpr int q1d = p + q_extra;

  outL  << "       % ===== h-adapt (L) p="<<p<<", ry="<<ry<<", rz="<<rz<<", q="<<q1d<<" =====\n"
        << "       \\addplot coordinates {";
  outR  << "       % ===== h-adapt (R) p="<<p<<", ry="<<ry<<", rz="<<rz<<", q="<<q1d<<" =====\n"
        << "       \\addplot coordinates {";
  outIF << "       % ===== h-adapt (IF) p="<<p<<", ry="<<ry<<", rz="<<rz<<", q="<<q1d<<" =====\n"
        << "       \\addplot coordinates {";

  const int nx_half = 2; // small & fixed along split axis
  int nyL = 1, nzL = 1;
  int toggle = 0;

  // stop when total element DoFs becomes too large
  const long long max_dofs = 25'000'000;

  while (true){
    // quick OOM guard (rough)
    const long long ndofs_est =
      1LL * nx_half * nyL * nzL * (p+1)*(p+1)*(p+1) +          // L
      1LL * nx_half * (ry*nyL) * (rz*nzL) * (p+1)*(p+1)*(p+1); // R
    if (ndofs_est > max_dofs) break;

    double tL=0, tR=0, tIF=0;
    long long fL=0, fR=0, fIF=0, dL=0, dR=0;

    bench_face_components_once_3D<p, (p+q_extra), ry, rz>(
        nx_half, nyL, nzL, tL, tR, tIF, fL, fR, fIF, dL, dR);

    const Integer nldofs = (p+1)*(p+1)*(p+1);

    outL  << " (" << fL*2*nldofs << ", " << fL*2*nldofs/tL  << ") ";
    outR  << " (" << fR*2*nldofs << ", " << fR*2*nldofs/tR  << ") ";
    outIF << " (" << fIF*2*nldofs << ", " << fIF*2*nldofs/tIF << ") ";
    // grow only nyL & nzL (alternate)
    if (toggle==0) nyL *= 2; else nzL *= 2;
    toggle ^= 1;
  }

  outL  << "};\n       \\addlegendentry{(L) $p="<<p<<", r_y="<<ry<<", r_z="<<rz<<", q="<<q1d<<"$}\n";
  outR  << "};\n       \\addlegendentry{(R) $p="<<p<<", r_y="<<ry<<", r_z="<<rz<<", q="<<q1d<<"$}\n";
  outIF << "};\n       \\addlegendentry{(IF) $p="<<p<<", r_y="<<ry<<", r_z="<<rz<<", q="<<q1d<<"$}\n";
}

int main(){
  std::ostringstream outL, outR, outIF;

  // a few representative settings:
  //   p=1 with moderate refinement, p=2 with stronger refinement
  sweep_h_adaptivity</*p=*/1, /*q_extra=*/2, /*ry=*/2, /*rz=*/1>(outL,outR,outIF);
  sweep_h_adaptivity</*p=*/1, /*q_extra=*/2, /*ry=*/2, /*rz=*/2>(outL,outR,outIF);
  sweep_h_adaptivity</*p=*/2, /*q_extra=*/2, /*ry=*/2, /*rz=*/2>(outL,outR,outIF);

  std::cout
    << " \\begin{tikzpicture}[scale=0.9]\n"
    << "    \\begin{axis}[\n"
    << "       title={3D Face Advection (h-adaptivity; dofs/sec)},\n"
    << "       xlabel={Number of Dofs},\n"
    << "       ylabel={Throughput [dofs/s]},\n"
    << "       legend pos=outer north east,\n"
    << "       grid=major,\n"
    << "       xmode=log,\n"
    << "       ymode=log,\n"
    << "       cycle list name=color list,\n"
    << "    ]\n"
    << outL.str() << outR.str() << outIF.str()
    << "    \\end{axis}\n"
    << " \\end{tikzpicture}\n";

  return 0;
}
