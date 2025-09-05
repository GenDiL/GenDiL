// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace gendil;

static inline long long interior_faces_3d(int nx,int ny,int nz)
{
  long long fx = 1LL*(nx-1)*ny*nz;
  long long fy = 1LL*nx*(ny-1)*nz;
  long long fz = 1LL*nx*ny*(nz-1);
  return fx + fy + fz;
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

// One size point → compute thrL/thrR/thrIF and DoF counts for each curve
template<Integer pL, Integer pR, Integer q1d>
void bench_face_components_once_3D(Integer nx_full, Integer ny, Integer nz,
                                   double& out_thrL, double& out_thrR, double& out_thrIF,
                                   Integer& out_ndofsL, Integer& out_ndofsR, Integer& out_ndofsIF)
{
  const Integer nxL = nx_full / 2;
  const Integer nxR = nx_full - nxL;

  const Real hx = 1.0 / nxL;
  const Real hy = 1.0 / ny;
  const Real hz = 1.0 / nz;

  Cartesian3DMesh meshL(hx, hy, hz, nxL, ny, nz, Point<3>{0.0, 0.0, 0.0});
  Cartesian3DMesh meshR(hx, hy, hz, nxR, ny, nz, Point<3>{1.0, 0.0, 0.0});

  auto feL = MakeLobattoFiniteElement(FiniteElementOrders<pL,pL,pL>{});
  auto feR = MakeLobattoFiniteElement(FiniteElementOrders<pR,pR,pR>{});

  L2Restriction resL{0};
  auto fe_space_L = MakeFiniteElementSpace(meshL, feL, resL);
  const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();

  L2Restriction resR{ndofsL};
  auto fe_space_R = MakeFiniteElementSpace(meshR, feR, resR);
  const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

  out_ndofsL = ndofsL;
  out_ndofsR = ndofsR;

  auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d>{});

  auto face_mesh_L = make_cartesian_interior_face_connectivity<3>(
      std::array<GlobalIndex,3>{(GlobalIndex)nxL,(GlobalIndex)ny,(GlobalIndex)nz});
  auto face_mesh_R = make_cartesian_interior_face_connectivity<3>(
      std::array<GlobalIndex,3>{(GlobalIndex)nxR,(GlobalIndex)ny,(GlobalIndex)nz});

  // LocalFaceIndex = +x face of minus side in 3D → 0 + Dim = 3
  constexpr Integer LFI = 3;
  CartesianIntermeshFaceConnectivity<3, LFI> iface(
      {(GlobalIndex)nxL,(GlobalIndex)ny,(GlobalIndex)nz},
      {(GlobalIndex)nxR,(GlobalIndex)ny,(GlobalIndex)nz});

  auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real,3>&, Real (&v)[3]) {
    v[0] = 1.0; v[1] = 1.0; v[2] = 1.0;
  };

#if defined(GENDIL_USE_DEVICE)
  using ThreadLayout = ThreadBlockLayout<q1d,q1d,q1d>;
  constexpr size_t NumSharedDimensions = 3;
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

  // L/R normalization:
  const Integer nldofsL = (pL+1)*(pL+1)*(pL+1);
  const Integer nldofsR = (pR+1)*(pR+1)*(pR+1);
  // faces for each operator
  const long long fL  = interior_faces_3d(nxL, ny, nz);
  const long long fR  = interior_faces_3d(nxR, ny, nz);
  const double denomL  = double(fL*nldofsL);
  const double denomR  = double(fR*nldofsR);

  // IF normalization by *face dofs processed*:
  const GlobalIndex nfacesIF = iface.GetNumberOfFaces();
  const Integer nfdL = (pL+1)*(pL+1)*(pL+1);
  const Integer nfdR = (pR+1)*(pR+1)*(pR+1);
  const Integer ndofsIF = static_cast<Integer>(nfacesIF) * (nfdL + nfdR);

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

  out_L  << "       % ===== pL="<<pL<<", pR="<<pR<<", q="<<q1d<<" =====\n"
         << "       \\addplot coordinates {";
  out_R  << "       % ===== pL="<<pL<<", pR="<<pR<<", q="<<q1d<<" =====\n"
         << "       \\addplot coordinates {";
  out_IF << "       % ===== pL="<<pL<<", pR="<<pR<<", q="<<q1d<<" =====\n"
         << "       \\addplot coordinates {";

  const Integer nx_full = 4; // -> nxL=2, nxR=2 (balanced)
  Integer ny = 1, nz = 1;
  const Integer max_items = 10'000'000; // safety cap

  // element dofs for time recovery
  const Integer nldofsL = (pL+1)*(pL+1)*(pL+1);
  const Integer nldofsR = (pR+1)*(pR+1)*(pR+1);

  int toggle = 0;
  while (true) {
    const int nxL = nx_full/2, nxR = nx_full - nxL;

    // rough limiter by total element DoFs to avoid OOM
    const long long ndofs_est =
      1LL*nxL*ny*nz*nldofsL + 1LL*nxR*ny*nz*nldofsR;
    if (ndofs_est > max_items) break;

    double thrL=0.0, thrR=0.0, thrIF=0.0;
    Integer ndofsL=0, ndofsR=0, ndofsIF=0;
    bench_face_components_once_3D<pL,pR,q1d>(nx_full, ny, nz,
                                             thrL, thrR, thrIF,
                                             ndofsL, ndofsR, ndofsIF);

    // faces for each operator
    const long long fL  = interior_faces_3d(nxL, ny, nz);
    const long long fR  = interior_faces_3d(nxR, ny, nz);
    const long long fIF = 1LL*ny*nz; // x-split interface slab

    // emit (x = #dofs, y = dofs/sec)
    out_L  << " (" << fL*2*nldofsL  << ", " << thrL  << ") ";
    out_R  << " (" << fR*2*nldofsR  << ", " << thrR  << ") ";
    out_IF << " (" << fIF*(nldofsL+nldofsR) << ", " << thrIF << ") ";

    // grow only ny and nz (alternate) so L/R/IF stay comparable
    if (toggle == 0) ny *= 2; else nz *= 2;
    toggle ^= 1;
  }

  out_L  << "};\n       \\addlegendentry{(L) $p_L="<<pL<<", p_R="<<pR<<", q="<<q1d<<"$}\n";
  out_R  << "};\n       \\addlegendentry{(R) $p_L="<<pL<<", p_R="<<pR<<", q="<<q1d<<"$}\n";
  out_IF << "};\n       \\addlegendentry{(IF) $p_L="<<pL<<", p_R="<<pR<<", q="<<q1d<<"$}\n";
}

int main()
{
  std::ostringstream curves_L, curves_R, curves_IF;

  // Choose (pL,pR) combos to sweep
  // sweep_face_components</*pL=*/1, /*pR=*/1>(curves_L, curves_R, curves_IF);
  sweep_face_components</*pL=*/1, /*pR=*/2>(curves_L, curves_R, curves_IF);
  sweep_face_components</*pL=*/1, /*pR=*/3>(curves_L, curves_R, curves_IF);
  // sweep_face_components</*pL=*/2, /*pR=*/2>(curves_L, curves_R, curves_IF);
  sweep_face_components</*pL=*/2, /*pR=*/4>(curves_L, curves_R, curves_IF);
  // sweep_face_components</*pL=*/3, /*pR=*/3>(curves_L, curves_R, curves_IF);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << " \\begin{tikzpicture}[scale=0.9]\n";
  std::cout << "    \\begin{axis}[\n";
  std::cout << "       title={3D Face Advection (p-adaptivity; L/R/IF)},\n";
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
