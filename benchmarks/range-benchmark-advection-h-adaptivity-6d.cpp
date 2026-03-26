// Copyright GenDiL Project Developers
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <chrono>
#include <sstream>
#include <iostream>
#include <limits>

using namespace gendil;

// ---------- helpers ----------

static inline long long interior_faces_6d(int nx,int ny,int nz,int nw,int nv,int nu){
  // total interior faces across the six orientations
  const long long fx = 1LL*(nx-1)*ny*nz*nw*nv*nu;
  const long long fy = 1LL*nx*(ny-1)*nz*nw*nv*nu;
  const long long fz = 1LL*nx*ny*(nz-1)*nw*nv*nu;
  const long long fw = 1LL*nx*ny*nz*(nw-1)*nv*nu;
  const long long fv = 1LL*nx*ny*nz*nw*(nv-1)*nu;
  const long long fu = 1LL*nx*ny*nz*nw*nv*(nu-1);
  return fx + fy + fz + fw + fv + fu;
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

// Build & time the three components for one (nyL,nzL,nwL,nvL,nuL) and refinement (ry,rz,rw,rv,ru)
// axis = 0 split (x), minus=L (coarse), plus=R (fine)
template<int p, int q1d, int ry, int rz, int rw, int rv, int ru>
void bench_face_components_once_6D(int nx_half, int nyL, int nzL, int nwL, int nvL, int nuL,
                                   double& tL, double& tR, double& tIF,
                                   long long& facesL, long long& facesR, long long& facesIF,
                                   long long& ndofsL, long long& ndofsR)
{
  constexpr int Dim = 6;
  static_assert(ry>=1 && rz>=1 && rw>=1 && rv>=1 && ru>=1, "refinement must be >= 1");

  const int nxL = nx_half;
  const int nxR = nx_half;                 // same along splitting axis
  const int nyR = ry*nyL;
  const int nzR = rz*nzL;
  const int nwR = rw*nwL;
  const int nvR = rv*nvL;
  const int nuR = ru*nuL;

  // meshes: 6D CartesianMesh uses (sizes, h, origin)
  const std::array<GlobalIndex,Dim> sizesL{
    (GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL,(GlobalIndex)nwL,(GlobalIndex)nvL,(GlobalIndex)nuL
  };
  const std::array<GlobalIndex,Dim> sizesR{
    (GlobalIndex)nxR,(GlobalIndex)nyR,(GlobalIndex)nzR,(GlobalIndex)nwR,(GlobalIndex)nvR,(GlobalIndex)nuR
  };

  const std::array<Real,Dim> hL{ 1.0/nxL, 1.0/nyL, 1.0/nzL, 1.0/nwL, 1.0/nvL, 1.0/nuL };
  const std::array<Real,Dim> hR{ 1.0/nxR, 1.0/nyR, 1.0/nzR, 1.0/nwR, 1.0/nvR, 1.0/nuR };

  CartesianMesh<Dim> meshL(sizesL, hL, Point<Dim>{0.0,0.0,0.0,0.0,0.0,0.0});
  CartesianMesh<Dim> meshR(sizesR, hR, Point<Dim>{1.0,0.0,0.0,0.0,0.0,0.0}); // shift +1 in x

  // FE (same p on both sides so we isolate h-adaptivity)
  auto fe = MakeLobattoFiniteElement(FiniteElementOrders<p,p,p,p,p,p>{});

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
  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d,q1d,q1d,q1d>{});

  // interior face meshes
  auto face_mesh_L = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxL,(GlobalIndex)nyL,(GlobalIndex)nzL,(GlobalIndex)nwL,(GlobalIndex)nvL,(GlobalIndex)nuL});
  auto face_mesh_R = make_cartesian_interior_face_connectivity<Dim>(
      std::array<GlobalIndex,Dim>{(GlobalIndex)nxR,(GlobalIndex)nyR,(GlobalIndex)nzR,(GlobalIndex)nwR,(GlobalIndex)nvR,(GlobalIndex)nuR});

  // nonconforming interface: minus = L coarse on +x face (LocalFaceIndex = Dim)
  // plus = R fine on -x face
  NonconformingCartesianIntermeshFaceConnectivity<Dim, /*LocalFaceIndex=*/Dim> iface(
      sizesL, sizesR);

  // advection β = (1,1,1,1,1,1)
  auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real,Dim>&, Real (&v)[Dim]){
    v[0]=1.0; v[1]=1.0; v[2]=1.0; v[3]=1.0; v[4]=1.0; v[5]=1.0;
  };

#if defined(GENDIL_USE_DEVICE)
  using ThreadLayout  = ThreadBlockLayout<q1d,q1d,q1d,q1d,q1d,q1d>;
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
    Real* pu = u.WriteHostData();
    for (Integer g=0; g<u.Size(); ++g) pu[g] = Real((g % 7) - 3) / 3.0;
  }
  r = 0.0;

  // time each component separately
  tL  = time_operator(face_L,  u, r);
  tR  = time_operator(face_R,  u, r);
  tIF = time_operator(face_IF, u, r);

  // face counts & IF subfaces on the interface slab
  facesL  = interior_faces_6d(nxL, nyL, nzL, nwL, nvL, nuL);
  facesR  = interior_faces_6d(nxR, nyR, nzR, nwR, nvR, nuR);
  facesIF = 1LL * nyR * nzR * nwR * nvR * nuR; // subfaces on the x-split interface slab
}

// One sweep: keep nx fixed and grow ny,nz,nw,nv,nu (cycling), writing (faces, faces/s)
template<int p, int q_extra, int ry, int rz, int rw, int rv, int ru>
void sweep_h_adaptivity(std::ostringstream& outL,
                        std::ostringstream& outR,
                        std::ostringstream& outIF)
{
  constexpr int q1d = p + q_extra;

  outL  << "       % ===== h-adapt (L, 6D) p="<<p<<", ry="<<ry<<", rz="<<rz<<", rw="<<rw<<", rv="<<rv<<", ru="<<ru<<", q="<<q1d<<" =====\n"
        << "       \\addplot coordinates {";
  outR  << "       % ===== h-adapt (R, 6D) p="<<p<<", ry="<<ry<<", rz="<<rz<<", rw="<<rw<<", rv="<<rv<<", ru="<<ru<<", q="<<q1d<<" =====\n"
        << "       \\addplot coordinates {";
  outIF << "       % ===== h-adapt (IF, 6D) p="<<p<<", ry="<<ry<<", rz="<<rz<<", rw="<<rw<<", rv="<<rv<<", ru="<<ru<<", q="<<q1d<<" =====\n"
        << "       \\addplot coordinates {";

  const int nx_half = 2; // small & fixed along split axis
  int nyL = 1, nzL = 1, nwL = 1, nvL = 1, nuL = 1;
  int toggle = 0;

  // stop when total element DoFs becomes too large
  const long long max_dofs = 25'000'000;

  while (true){
    // quick OOM guard (rough)
    const long long nldofs = 1LL*(p+1)*(p+1)*(p+1)*(p+1)*(p+1)*(p+1);
    const long long ndofs_est =
      1LL * nx_half * nyL * nzL * nwL * nvL * nuL * nldofs +                                   // L
      1LL * nx_half * (ry*nyL) * (rz*nzL) * (rw*nwL) * (rv*nvL) * (ru*nuL) * nldofs;           // R
    if (ndofs_est > max_dofs) break;

    double tL=0, tR=0, tIF=0;
    long long fL=0, fR=0, fIF=0, dL=0, dR=0;

    bench_face_components_once_6D<p, (p+q_extra), ry, rz, rw, rv, ru>(
        nx_half, nyL, nzL, nwL, nvL, nuL, tL, tR, tIF, fL, fR, fIF, dL, dR);

    outL  << " (" << fL*2*nldofs << ", " << (fL*2*nldofs)/tL  << ") ";
    outR  << " (" << fR*2*nldofs << ", " << (fR*2*nldofs)/tR  << ") ";
    outIF << " (" << fIF*2*nldofs << ", " << (fIF*2*nldofs)/tIF << ") ";

    // grow only tangential dims (cycle y→z→w→v→u)
    switch (toggle){
      case 0: nyL *= 2; break;
      case 1: nzL *= 2; break;
      case 2: nwL *= 2; break;
      case 3: nvL *= 2; break;
      default: nuL *= 2; break;
    }
    toggle = (toggle + 1) % 5;
  }

  outL  << "};\n       \\addlegendentry{(L, 6D) $p="<<p<<", r_y="<<ry<<", r_z="<<rz<<", r_w="<<rw<<", r_v="<<rv<<", r_u="<<ru<<", q="<<q1d<<"$}\n";
  outR  << "};\n       \\addlegendentry{(R, 6D) $p="<<p<<", r_y="<<ry<<", r_z="<<rz<<", r_w="<<rw<<", r_v="<<rv<<", r_u="<<ru<<", q="<<q1d<<"$}\n";
  outIF << "};\n       \\addlegendentry{(IF, 6D) $p="<<p<<", r_y="<<ry<<", r_z="<<rz<<", r_w="<<rw<<", r_v="<<rv<<", r_u="<<ru<<", q="<<q1d<<"$}\n";
}

int main(){
  std::ostringstream outL, outR, outIF;

  // Representative settings (tweak as needed)
  sweep_h_adaptivity</*p=*/1, /*q_extra=*/2, /*ry=*/2, /*rz=*/1, /*rw=*/1, /*rv=*/1, /*ru=*/1>(outL,outR,outIF);
  sweep_h_adaptivity</*p=*/1, /*q_extra=*/2, /*ry=*/2, /*rz=*/2, /*rw=*/1, /*rv=*/1, /*ru=*/1>(outL,outR,outIF);
  sweep_h_adaptivity</*p=*/2, /*q_extra=*/2, /*ry=*/2, /*rz=*/2, /*rw=*/2, /*rv=*/2, /*ru=*/2>(outL,outR,outIF);

  std::cout
    << " \\begin{tikzpicture}[scale=0.9]\n"
    << "    \\begin{axis}[\n"
    << "       title={6D Face Advection (h-adaptivity; dofs/sec)},\n"
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
