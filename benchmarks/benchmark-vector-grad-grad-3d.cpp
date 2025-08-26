// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifdef GENDIL_USE_MFEM
#include <mfem.hpp>
#endif

#include <gendil/gendil.hpp>
#include <chrono>

using namespace std;
using namespace gendil;

int main(int argc, char *argv[])
{
   const Integer num_elem_1d = 40;

   //----------------------------------------------------------------------------
   // Meshes
   const Real h = 1.0;
   Cartesian1DMesh mesh_x(h, num_elem_1d),
                  mesh_y(h, num_elem_1d),
                  mesh_z(h, num_elem_1d);

   auto tensor_mesh = MakeCartesianProductMesh(mesh_x, mesh_y, mesh_z);

#ifdef GENDIL_USE_MFEM
   auto mfem_mesh = mfem::Mesh::MakeCartesian3D(
     num_elem_1d, num_elem_1d, num_elem_1d,
     mfem::Element::HEXAHEDRON, 1.0,1.0,1.0,false
   );
   HexMesh<1> unstruct_mesh = MakeHexMesh<1>(mfem_mesh);
#endif

   //----------------------------------------------------------------------------
   // FE Space
   constexpr Integer order = 3;
   FiniteElementOrders<order,order,order> orders;
   auto fe = MakeLegendreFiniteElement(orders);
   auto vector_fe = MakeVectorFiniteElement(fe, fe, fe);
   auto tensor_fes = MakeFiniteElementSpace(tensor_mesh, vector_fe);

#ifdef GENDIL_USE_MFEM
   auto unstruct_fes = MakeFiniteElementSpace(unstruct_mesh, vector_fe);
#endif

   //----------------------------------------------------------------------------
   // Quadrature
   constexpr Integer nq1d = order + 2;
   IntegrationRuleNumPoints<nq1d,nq1d,nq1d> nq;
   auto int_rules = MakeIntegrationRule(nq);

   //----------------------------------------------------------------------------
   // Kernel config (CPU / CUDA / HIP)
#if defined(GENDIL_USE_DEVICE)
#ifdef GENDIL_USE_CUDA
   const char device_cfg[] = "cuda";
#elif defined(GENDIL_USE_HIP)
   const char device_cfg[] = "hip";
#endif
   constexpr Integer NumSharedDims = 3;
   using ThreadLayout = ThreadBlockLayout<nq1d,nq1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout,NumSharedDims>;
#else
#ifdef GENDIL_USE_MFEM
   const char device_cfg[] = "cpu";
#endif
   using KernelPolicy = SerialKernelConfiguration;
#endif

#ifdef GENDIL_USE_MFEM
   mfem::Device device(device_cfg);
   device.Print();
#endif

   //----------------------------------------------------------------------------
   // Create Grad-Grad Operators
   auto tensor_diff_op = MakeGradGradOperator<KernelPolicy>(
     tensor_fes, int_rules
   );

#ifdef GENDIL_USE_MFEM
   auto unstruct_diff_op = MakeGradGradOperator<KernelPolicy>(
     unstruct_fes, int_rules
   );
#endif

   //----------------------------------------------------------------------------
   // Benchmark loop
   const Integer num_iter = 5;
   double tensor_tp = 0.0, unstruct_tp = 0.0;

   // --- Tensor‐mesh grad-grad ---
   {
      cout << "\n>> Vector grad-grad on tensor mesh\n";
      const Integer N = tensor_fes.GetNumberOfFiniteElementDofs();
      Vector in(N), out(N);

      in = 1.0;
      tensor_diff_op(in, out);
      GENDIL_DEVICE_SYNC;

      auto t0 = chrono::steady_clock::now();
      for ( Integer i = 0; i < num_iter; ++i )
      {
         tensor_diff_op(out, in);
         tensor_diff_op(in, out);
      }
      GENDIL_DEVICE_SYNC;
      auto t1 = chrono::steady_clock::now();

      double elapsed = chrono::duration<double>(t1 - t0).count();
      Integer total_calls = 2 * num_iter;
      double time_per = elapsed / total_calls;
      tensor_tp = N / time_per;

      cout << "Dofs: " << N
           << "  Total time: "  << elapsed << " s"
           << "  It/total: "    << total_calls
           << "  Time/it: "     << time_per << " s"
           << "  Throughput: "  << tensor_tp << " Dofs/s\n";
   }

#ifdef GENDIL_USE_MFEM
   // --- Unstructured‐mesh grad-grad ---
   {
      cout << "\n>> Vector grad-grad on unstructured mesh\n";
      const Integer N = unstruct_fes.GetNumberOfFiniteElementDofs();
      mfem::Vector in(N), out(N);

      in = 1.0;
      unstruct_diff_op.Mult(in, out);
      GENDIL_DEVICE_SYNC;

      auto t0 = chrono::steady_clock::now();
      for ( Integer i = 0; i < num_iter; ++i )
      {
         unstruct_diff_op.Mult(out, in);
         unstruct_diff_op.Mult(in, out);
      }
      GENDIL_DEVICE_SYNC;
      auto t1 = chrono::steady_clock::now();

      double elapsed = chrono::duration<double>(t1 - t0).count();
      Integer total_calls = 2 * num_iter;
      double time_per = elapsed / total_calls;
      unstruct_tp = N / time_per;

      cout << "Dofs: " << N
           << "  Total time: "  << elapsed << " s"
           << "  It/total: "    << total_calls
           << "  Time/it: "     << time_per << " s"
           << "  Throughput: "  << unstruct_tp << " Dofs/s\n";
   }

   cout << "\n>> Speedup tensor/unstruct: "
        << tensor_tp / unstruct_tp << "\n";
#endif

   return 0;
}
