// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace mfem;
using namespace gendil;

mfem::Mesh mesh_2d_orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 2;
   static const int nv = 6;
   static const int nel = 2;
   mfem::Mesh mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   int el[4];
   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   std::rotate(&el[0], &el[face_perm_1], &el[3] + 1);

   mesh.AddQuad(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   std::rotate(&el[0], &el[face_perm_2], &el[3] + 1);
   mesh.AddQuad(el);

   mesh.FinalizeQuadMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

void rotate_3d_vertices(int *v, int ref_face, int rot)
{
   std::vector<int> face_1, face_2;

   switch (ref_face/2)
   {
      case 0:
         face_1 = {v[0], v[1], v[2], v[3]};
         face_2 = {v[4], v[5], v[6], v[7]};
         break;
      case 1:
         face_1 = {v[1], v[5], v[6], v[2]};
         face_2 = {v[0], v[4], v[7], v[3]};
         break;
      case 2:
         face_1 = {v[4], v[5], v[1], v[0]};
         face_2 = {v[7], v[6], v[2], v[3]};
         break;
   }
   if (ref_face % 2 == 0)
   {
      std::reverse(face_1.begin(), face_1.end());
      std::reverse(face_2.begin(), face_2.end());
      std::swap(face_1, face_2);
   }

   std::rotate(face_1.begin(), face_1.begin() + rot, face_1.end());
   std::rotate(face_2.begin(), face_2.begin() + rot, face_2.end());

   for (int i=0; i<4; ++i)
   {
      v[i] = face_1[i];
      v[i+4] = face_2[i];
   }
}

mfem::Mesh mesh_3d_orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;
   mfem::Mesh mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 3.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);

   int el[8];

   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   rotate_3d_vertices(el, face_perm_1/4, face_perm_1%4);
   mesh.AddHex(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   rotate_3d_vertices(el, face_perm_2/4, face_perm_2%4);
   mesh.AddHex(el);

   mesh.FinalizeHexMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

template < Integer order, Integer num_quad_1d >
void test( int fp1, int fp2 )
{
   // const int nx = 2;//fp1 + 1;
   // const int ny = 1;//fp2 + 1;
   // const int nz = 1;
   // // std::cout << "(" << nx << ", " << ny << ", " << nz << ", " << ")" << std::endl;
   // auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( nx, ny, nz, mfem::Element::Type::HEXAHEDRON, nx, ny, nz, false );
   constexpr Integer mesh_order = 1;
   mfem::Mesh non_periodic_mesh = mesh_3d_orientation(fp1, fp2);
   HexMesh< mesh_order > mesh = MakeHexMesh< mesh_order >( non_periodic_mesh );

   ///////////////////////
   // Finite Element Space
   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto adv = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      v[0] = y;
      v[1] = z;
      v[2] = x;
   };
   auto zero = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X ){ return 0.0; };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto advection_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, adv, zero );

   FiniteElementVector dofs_in( fe_space );
   FiniteElementVector dofs_out_mf( fe_space );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs = num_elem * num_elem_dofs;
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in.HostReadWrite();
   // dofs_in[0] = 1.0;
   // dofs_in = 1.0;
   advection_operator.Mult( dofs_in, dofs_out_mf );

   /// Using MFEM partial assembly DG
   int dim = non_periodic_mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&non_periodic_mesh, &fec);
   
   GridFunction dofs_out_pa( &fes );

   auto adv_func = [=](const Vector& x, Vector& v)
   {
      const Real X = x(0);
      const Real Y = x(1);
      const Real Z = x(2);
      v[0] = Y;
      v[1] = Z;
      v[2] = X;
   };
   VectorFunctionCoefficient velocity(3, adv_func);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   constexpr double alpha = 1.0; // Should discretize div(adv*psi)
   blf_pa.AddDomainIntegrator(
      new ConservativeConvectionIntegrator(velocity, alpha));
   blf_pa.AddInteriorFaceIntegrator(
      new ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.AddBdrFaceIntegrator(
      new ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.Assemble();
   blf_pa.Mult(dofs_in, dofs_out_pa);

   // std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   // std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   // std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   // std::cout << "mf: "; dofs_out_mf.Print(std::cout, dofs_out_mf.Size());
   // std::cout << "pa: "; dofs_out_pa.Print(std::cout, dofs_out_pa.Size());


   // std::cout << "min(Apsi_mf), max(Apsi_mf): " << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << endl;
   // std::cout << "min(Apsi_pa), max(Apsi_pa): " << dofs_out_pa.Min() << ", " << dofs_out_pa.Max() << endl;
   // std::cout << "in vector: ";
   // dofs_in.Print( std::cout, num_elem_dofs );
   // std::cout << "\nmf vector:\n";
   // dofs_out_mf.Print( std::cout, num_elem_dofs );
   // std::cout << "\npa vector:\n";
   // dofs_out_pa.Print( std::cout, num_elem_dofs );
   // dofs_out_mf.HostRead();
   // dofs_out_pa.HostRead();
   // for( size_t i=0; i < dofs_out_mf.Size(); i++)
   // {
   //    const Real diff = dofs_out_mf[i] - dofs_out_pa[i];
   //    const Real distance = diff * diff;
   //    if( distance > 1e-10 )
   //    {
   //       std::cout << i << std::endl;
   //    }
   // }

   dofs_out_mf -= dofs_out_pa;

   const double tol = 1e-10;
   const double norm = dofs_out_mf.Norml2();
   if (dofs_out_mf.Norml2() < tol)
   {
      std::cout << "SUCCESS: " << norm << std::endl;
   }
   else
   {
      std::cout << "FAILED: " << norm << std::endl;
   }
   
   // std::cout << "Absolute norm of the diff: " << dofs_out_mf.Norml2() << std::endl;
   // std::cout << "Relative norm of the diff: " << dofs_out_mf.Norml2()/dofs_out_pa.Norml2() << std::endl;

}

int main(int argc, char *argv[])
{
#if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
#elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
#else
   const char device_config[] = "cpu";
#endif
   mfem::Device device(device_config);
   device.Print();

   constexpr Integer order = 1;
   constexpr Integer num_quad_1d = order+2;
   const size_t n_perms = 24;
   for (size_t fp2=0; fp2<n_perms; ++fp2)
   {
      for (size_t fp1=0; fp1<n_perms; ++fp1)
      {
         std::cout << "\nTEST< " << fp1 << ", " << fp2 << " >:\n";
         test<order,num_quad_1d>(fp1, fp2);
      }
   }

   return 0;
}
