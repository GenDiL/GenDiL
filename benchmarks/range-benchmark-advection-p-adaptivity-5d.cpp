// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace gendil;

static inline long long interior_faces_5d(int nx,int ny,int nz,int nw,int nt)
{
  const long long fx = 1LL*(nx-1)*ny*nz*nw*nt;
  const long long fy = 1LL*nx*(ny-1)*nz*nw*nt;
  const long long fz = 1LL*nx*ny*(nz-1)*nw*nt;
  const long long fw = 1LL*nx*ny*nz*(nw-1)*nt;
  const long long ft = 1LL*nx*ny*nz*nw*(nt-1);
  return fx + fy + fz + fw + ft;
}

template<class Op>
double time_face_op(Op&& op, Vector& x, Vector& y, int num_iter)
{
  op(x, y); // warmup
  GENDIL_DEVICE_SYNC;
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < num_iter; ++it) {
    op(y, x);
    op(x, y);
  }
  GENDIL_DEVICE_SYNC;
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / double(2 * num_iter);
}

// One size point → compute thrL/thrR/thrIF and DoF counts for each curve (5D)
template<Integer pL, Integer pR, Integer q1d>
void bench_face_components_once_5D(Integer nx_full, Integer ny, Integer nz, Integer nw, Integer nt,
                                   double& out_thrL, double& out_thrR, double& out_thrIF,
                                   Integer& out_ndofsL, Integer& out_ndofsR, Integer& out_ndofsIF)
{
  const Integer nxL = nx_full / 2;
  const Integer nxR = nx_full - nxL;

  const Real hx = 1.0 / nxL;
  const Real hy = 1.0 / ny;
  const Real hz = 1.0 / nz;
  const Real hw = 1.0 / nw;
  const Real ht = 1.0 / nt;

  // 5D Cartesian meshes: left in [0,1]^5, right shifted +1 in x
  CartesianMesh<5> meshL({nxL, ny, nz, nw, nt}, {hx, hy, hz, hw, ht}, Point<5>{0.0, 0.0, 0.0, 0.0, 0.0});
  CartesianMesh<5> meshR({nxR, ny, nz, nw, nt}, {hx, hy, hz, hw, ht}, Point<5>{1.0, 0.0, 0.0, 0.0, 0.0});

  auto feL = MakeLobattoFiniteElement(FiniteElementOrders<pL,pL,pL,pL,pL>{});
  auto feR = MakeLobattoFiniteElement(FiniteElementOrders<pR,pR,pR,pR,pR>{});

  L2Restriction resL{0};
  auto fe_space_L = MakeFiniteElementSpace(meshL, feL, resL);
  const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();

  L2Restriction resR{ndofsL};
  auto fe_space_R = MakeFiniteElementSpace(meshR, feR, resR);
  const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

  out_ndofsL = ndofsL;
  out_ndofsR = ndofsR;

  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d,q1d,q1d>{});

  auto face_mesh_L = make_cartesian_interior_face_connectivity<5>(
      std::array<GlobalIndex,5>{
        (GlobalIndex)nxL,(GlobalIndex)ny,(GlobalIndex)nz,(GlobalIndex)nw,(GlobalIndex)nt});
  auto face_mesh_R = make_cartesian_interior_face_connectivity<5>(
      std::array<GlobalIndex,5>{
        (GlobalIndex)nxR,(GlobalIndex)ny,(GlobalIndex)nz,(GlobalIndex)nw,(GlobalIndex)nt});

  // LocalFaceIndex = +x face of minus side in 5D → Axis(0) + Dim(5) = 5
  constexpr Integer LFI = 5;
  CartesianIntermeshFaceConnectivity<5, LFI> iface(
      {(GlobalIndex)nxL,(GlobalIndex)ny,(GlobalIndex)nz,(GlobalIndex)nw,(GlobalIndex)nt},
      {(GlobalIndex)nxR,(GlobalIndex)ny,(GlobalIndex)nz,(GlobalIndex)nw,(GlobalIndex)nt});

  auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real,5>&, Real (&v)[5]) {
    v[0] = 1.0; v[1] = 1.0; v[2] = 1.0; v[3] = 1.0; v[4] = 1.0;
  };

#if defined(GENDIL_USE_DEVICE)
  using ThreadLayout = ThreadBlockLayout<q1d,q1d,q1d,q1d,q1d>;
  constexpr size_t NumSharedDimensions = 5;
  using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
  using KernelPolicy = SerialKernelConfiguration;
#endif

  auto face_L  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, face_mesh_L, int_rules, adv);
  auto face_R  = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_R, face_mesh_R, int_rules, adv);
  auto face_IF = MakeAdvectionFaceOperator<KernelPolicy>(fe_space_L, fe_space_R, iface, int_rules, adv);

  const Integer ndofs_split = ndofsL + ndofsR;
  Vector x(ndofs_split), y(ndofs_split);
  x = 1.0; y = 0.0;

  const int num_iter = 5;
  const double tpL  = time_face_op(face_L,  x, y, num_iter);
  const double tpR  = time_face_op(face_R,  x, y, num_iter);
  const double tpIF = time_face_op(face_IF, x, y, num_iter);

  // Element dofs (Q_p in 5D)
  const Integer nldofsL = (pL+1)*(pL+1)*(pL+1)*(pL+1)*(pL+1);
  const Integer nldofsR = (pR+1)*(pR+1)*(pR+1)*(pR+1)*(pR+1);

  // Faces for each operator (interior faces in 5D)
  const long long fL  = interior_faces_5d(nxL, ny, nz, nw, nt);
  const long long fR  = interior_faces_5d(nxR, ny, nz, nw, nt);
  const double denomL = double(fL * nldofsL);
  const double denomR = double(fR * nldofsR);

  // IF normalization by *face dofs processed*:
  const GlobalIndex nfacesIF = iface.GetNumberOfFaces(); // equals ny*nz*nw*nt here
  const Integer ndofsIF = static_cast<Integer>(nfacesIF) * (nldofsL + nldofsR);

  out_ndofsIF = ndofsIF;

  out_thrL  = denomL  / tpL;
  out_thrR  = denomR  / tpR;
  out_thrIF = double(ndofsIF) / tpIF;
}

template<Integer pL, Integer pR, Integer extra_q = 2>
void sweep_face_components(std::ostringstream& out_L,
                           std::ostringstream& out_R,
                           std::ostringstream& out_IF)
{
  constexpr Integer q1d = ((pL > pR ? pL : pR) + extra_q);

  out_L  << "       % ===== pL="<<pL<<", pR="<<pR<<", q="<<q1d<<" (5D) =====\n"
         << "       \\addplot coordinates {";
  out_R  << "       % ===== pL="<<pL<<", pR="<<pR<<", q="<<q1d<<" (5D) =====\n"
         << "       \\addplot coordinates {";
  out_IF << "       % ===== pL="<<pL<<", pR="<<pR<<", q="<<q1d<<" (5D) =====\n"
         << "       \\addplot coordinates {";

  const Integer nx_full = 4; // -> nxL=2, nxR=2 (balanced)
  Integer ny = 1, nz = 1, nw = 1, nt = 1;
  const long long max_items = 10'000'000; // safety cap (tune as needed)

  // element dofs (5D)
  const Integer nldofsL = (pL+1)*(pL+1)*(pL+1)*(pL+1)*(pL+1);
  const Integer nldofsR = (pR+1)*(pR+1)*(pR+1)*(pR+1)*(pR+1);

  int toggle = 0;
  while (true) {
    const int nxL = nx_full/2, nxR = nx_full - nxL;

    // rough limiter by total element DoFs to avoid OOM
    const long long ndofs_est =
      1LL*nxL*ny*nz*nw*nt*nldofsL + 1LL*nxR*ny*nz*nw*nt*nldofsR;
    if (ndofs_est > max_items) break;

    double thrL=0.0, thrR=0.0, thrIF=0.0;
    Integer ndofsL=0, ndofsR=0, ndofsIF=0;
    bench_face_components_once_5D<pL,pR,q1d>(nx_full, ny, nz, nw, nt,
                                             thrL, thrR, thrIF,
                                             ndofsL, ndofsR, ndofsIF);

    // faces for each operator
    const long long fL  = interior_faces_5d(nxL, ny, nz, nw, nt);
    const long long fR  = interior_faces_5d(nxR, ny, nz, nw, nt);
    const long long fIF = 1LL*ny*nz*nw*nt; // x-split interface slab in 5D

    // emit (x = #dofs, y = dofs/sec)
    out_L  << " (" << fL*2*nldofsL  << ", " << thrL  << ") ";
    out_R  << " (" << fR*2*nldofsR  << ", " << thrR  << ") ";
    out_IF << " (" << fIF*(nldofsL+nldofsR) << ", " << thrIF << ") ";

    // grow tangential dims cyclically so L/R/IF stay comparable
    if      (toggle == 0) ny *= 2;
    else if (toggle == 1) nz *= 2;
    else if (toggle == 2) nw *= 2;
    else                  nt *= 2;
    toggle = (toggle + 1) % 4;
  }

  out_L  << "};\n       \\addlegendentry{(L, 5D) $p_L="<<pL<<", p_R="<<pR<<", q="<<q1d<<"$}\n";
  out_R  << "};\n       \\addlegendentry{(R, 5D) $p_L="<<pL<<", p_R="<<pR<<", q="<<q1d<<"$}\n";
  out_IF << "};\n       \\addlegendentry{(IF, 5D) $p_L="<<pL<<", p_R="<<pR<<", q="<<q1d<<"$}\n";
}

int main()
{
  std::ostringstream curves_L, curves_R, curves_IF;

  // Choose (pL,pR) combos to sweep
  sweep_face_components</*pL=*/1, /*pR=*/2>(curves_L, curves_R, curves_IF);
  sweep_face_components</*pL=*/1, /*pR=*/3>(curves_L, curves_R, curves_IF);
  sweep_face_components</*pL=*/2, /*pR=*/4>(curves_L, curves_R, curves_IF);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << " \\begin{tikzpicture}[scale=0.9]\n";
  std::cout << "    \\begin{axis}[\n";
  std::cout << "       title={5D Face Advection (p-adaptivity; L/R/IF)},\n";
  std::cout << "       xlabel={Operator DoFs},\n";
  std::cout << "       ylabel={Throughput [DoF/s]},\n";
  std::cout << "       legend pos=outer north east,\n";
  std::cout << "       grid=major,\n";
  std::cout << "       xmode = log,\n";
  std::cout << "       ymode = log,\n";
  std::cout << "       cycle list name=color list,\n";
  std::cout << "       ]\n";
  std::cout << curves_L.str();
  std::cout << curves_R.str();
  std::cout << curves_IF.str();
  std::cout << "    \\end{axis}\n";
  std::cout << " \\end{tikzpicture}\n";
  return 0;
}
