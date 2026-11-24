// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>
#include <gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/globalfacesfromlocalconnectivity.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace form_transpose_facet_test
{
using namespace gendil;
using Policy=SerialKernelConfiguration;
using Dense=std::vector<Real>;

bool Near(Real actual,Real expected,const char* label,Real tolerance=1.e-13)
{
   if(std::isfinite(actual) && std::isfinite(expected) &&
      std::abs(actual-expected)<=tolerance*(1+std::abs(expected))) return true;
   std::cerr << label << ": got " << actual << ", expected " << expected << '\n'; return false;
}
Dense Transposed(const Dense& matrix,Integer rows,Integer cols)
{
   Dense result(matrix.size());
   for(Integer i=0;i<rows;++i) for(Integer j=0;j<cols;++j) result[j*rows+i]=matrix[i*cols+j];
   return result;
}
template<class Operator>
bool CheckOperator(const Operator& op,const Dense& matrix,Integer rows,Integer cols,const char* label)
{
   if(op.InputSize()!=cols || op.OutputSize()!=rows)
   { std::cerr << label << ": incorrect extents\n";return false; }
   bool ok=true; Vector x(cols),y(rows);
   for(Integer j=0;j<cols;++j)
   {
      auto* input=x.WriteHostData(); std::fill_n(input,cols,Real(0));input[j]=1;
      y=7;op(x,y);
      for(Integer i=0;i<rows;++i) ok=Near(y.ReadHostData()[i],matrix[i*cols+j],label) && ok;
   }
   auto* input=x.WriteHostData();
   for(Integer j=0;j<cols;++j) input[j]=.2+.4*j-.03*j*j;
   Dense expected(rows,0);
   for(Integer i=0;i<rows;++i) for(Integer j=0;j<cols;++j) expected[i]+=matrix[i*cols+j]*input[j];
   auto* output=y.WriteHostData();for(Integer i=0;i<rows;++i) output[i]=2-.3*i;
   for(Integer repeat=1;repeat<=2;++repeat)
   {
      op.ApplyAdd(x,y);
      for(Integer i=0;i<rows;++i) ok=Near(y.ReadHostData()[i],2-.3*i+repeat*expected[i],label) && ok;
   }
   return ok;
}
template<class Form,class Context,class Rule>
bool CheckForm(const Form& form,const Context& context,const Rule& rule,
   const Dense& matrix,Integer rows,Integer cols,const char* label)
{
   const auto a=MakeGenericOperator<Policy>(form,context,rule);
   const auto at=MakeGenericOperator<Policy>(transpose(form),context,rule);
   const auto att=MakeGenericOperator<Policy>(transpose(transpose(form)),context,rule);
   bool ok=CheckOperator(a,matrix,rows,cols,label);
   ok=CheckOperator(at,Transposed(matrix,rows,cols),cols,rows,label) && ok;
   return CheckOperator(att,matrix,rows,cols,label) && ok;
}

// Explicit left-minus connectivity is essential: Cartesian interior faces use
// the opposite canonical ordering. Faces 0 and 1 are x=0 and x=1 in 1D.
struct UnitIntervalCell: HyperCubeCell<1>
{using HyperCubeCell<1>::HyperCubeCell;};
// This fixture's runtime connectivity stores only identity permutations.
// Provide its geometry hook, as in the runtime facet-equivalence fixture.
template<class Orientation>
void ApplyOrientationToCell(const Orientation&,UnitIntervalCell&) {}
struct LeftMinusMesh: CartesianMesh<1>
{
   using cell_type=UnitIntervalCell;
   cell_type GetCell(GlobalIndex cell) const {return {Point<1>{Real(cell)},std::array<Real,1>{1}};}
   UnstructuredConformingConnectivity<HyperCube<1>> connectivity{2};
   LeftMinusMesh():CartesianMesh<1>({2},{1.},{0.},false)
   {
      for(Integer c=0;c<2;++c) for(Integer f=0;f<2;++f)
         connectivity[c].faces[f]={0,{},MakeReferencePermutation<1>(),{},{},true};
      connectivity[0].faces[1]={1,{},MakeReferencePermutation<1>(),{},{},false};
      connectivity[1].faces[0]={0,{},MakeReferencePermutation<1>(),{},{},false};
   }
   template<Integer Face>
   auto GetLocalFaceInfo(GlobalIndex cell,std::integral_constant<Integer,Face> face) const
   {return connectivity.GetLocalFaceInfo(cell,face);}
};

template<class Context,class Rule>
bool LiteralOracle(const Context& context,const Rule& rule)
{
   const TrialSpace<"u"> u; const TestSpace<"v"> v;
   const auto form=integrate(Cells<"m">{},u*v)
      +integrate(InteriorFacets<"m">{},2.*minus(u)*plus(v)-3.*plus(u)*minus(v));
   const Dense matrix{.5,0,.5,-3,2,.5,0,.5};
   bool ok=CheckForm(form,context,rule,matrix,4,2,"literal rectangular facet matrix");
   const auto a=MakeGenericOperator<Policy>(form,context,rule);
   const auto at=MakeGenericOperator<Policy>(transpose(form),context,rule);
   const auto sum=MakeGenericOperator<Policy>(transpose(form)+transpose(form),context,rule);
   auto doubled=Transposed(matrix,4,2);for(auto& value:doubled)value*=2;
   ok=CheckOperator(sum,doubled,2,4,"rectangular sum containing transpose") && ok;
   Vector x(2),y(4),ax(4),aty(2);
   auto* xv=x.WriteHostData();xv[0]=2;xv[1]=-1;
   auto* yv=y.WriteHostData();yv[0]=3;yv[1]=-2;yv[2]=5;yv[3]=7;
   a(x,ax);at(y,aty);
   const Real expected_ax[4]={1,4,3.5,-.5};
   for(Integer i=0;i<4;++i)ok=Near(ax.ReadHostData()[i],expected_ax[i],"literal vector action") && ok;
   ok=Near(aty.ReadHostData()[0],10.5,"literal transpose action") && ok;
   ok=Near(aty.ReadHostData()[1],12,"literal transpose action") && ok;
   Real lhs=0,rhs=0;
   for(Integer i=0;i<4;++i)lhs+=yv[i]*ax.ReadHostData()[i];
   for(Integer i=0;i<2;++i)rhs+=xv[i]*aty.ReadHostData()[i];
   ok=Near(lhs,9,"literal forward pairing") && ok;
   ok=Near(rhs,9,"literal transposed pairing") && ok;
   return ok;
}

bool RectangularLocalAndGlobal()
{
   const LeftMinusMesh mesh;
   const auto faces=MakeUnstructuredInteriorFaceConnectivity(mesh);
   const auto partition=MakePartition(MakeCellPart(mesh),MakeInteriorFacePart<0,0>(faces));
   const auto p0=MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto p1=MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto local_u=MakeFiniteElementSpace(mesh,p0);
   const auto local_v=MakeFiniteElementSpace(mesh,p1);
   const auto global_u=MakeMixedFiniteElementSpace(partition,std::tuple{p0},DGDirectSumNumbering{});
   const auto global_v=MakeMixedFiniteElementSpace(partition,std::tuple{p1},DGDirectSumNumbering{});
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3>{});
   const auto local=MakeWeakFormContext(MakeTrialField<"u">(local_u),MakeTestField<"v">(local_v),
      MakeIntegrationDomain<"m">(mesh));
   const auto global=MakeWeakFormContext(MakeTrialField<"u">(global_u),MakeTestField<"v">(global_v),
      MakeIntegrationDomain<"m">(partition));
   bool ok=LiteralOracle(local,rule);
   return LiteralOracle(global,rule) && ok;
}

template<class Context,class Rule>
bool InteriorFlux(const Context& context,const Rule& rule)
{
   const TrialSpace<"u"> u;const TestSpace<"v"> v;
   const auto prescribed=MakeCoefficient<"prescribed",FieldValue<"w">>([] GENDIL_HOST_DEVICE (Real value){return value;});
   bool ok=true;
   for(Real speed:{Real(-2),Real(0),Real(2)})
   {
      const auto beta=MakeVectorCoefficient<"beta">([speed] GENDIL_HOST_DEVICE () {return std::array<Real,1>{speed};});
      Dense matrix(16,0);
      const Real positive=std::max(speed,Real(0)),negative=std::min(speed,Real(0));
      matrix[1*4+1]=positive;matrix[1*4+2]=negative;
      matrix[2*4+1]=-positive;matrix[2*4+2]=-negative;
      const auto value_form=integrate(InteriorFacets<"m">{},upwind(beta,u)*jump(v));
      ok=CheckForm(value_form,context,rule,matrix,4,4,"frozen-speed flux signs") && ok;
      const auto direction=MakeVectorCoefficient<"direction">([] GENDIL_HOST_DEVICE () {return std::array<Real,1>{1};});
      const auto gradient_form=integrate(InteriorFacets<"m">{},dot(upwind(beta,grad(u)),direction)*jump(v));
      Dense gradient(16,0);
      // On each unit interval, the physical P1 derivatives are (-1,+1).
      const Integer first=speed>=0?0:2;
      gradient[1*4+first]=-speed;gradient[1*4+first+1]=speed;
      gradient[2*4+first]=speed;gradient[2*4+first+1]=-speed;
      ok=CheckForm(gradient_form,context,rule,gradient,4,4,"frozen gradient transport") && ok;
      matrix[1*4+1]*=3;matrix[2*4+1]*=3;
      matrix[1*4+2]*=6;matrix[2*4+2]*=6;
      ok=CheckForm(integrate(InteriorFacets<"m">{},upwind(beta,prescribed*u)*jump(v)),
         context,rule,matrix,4,4,"selected prescribed-state coefficient") && ok;
   }
   return ok;
}

bool FluxAndBoundary()
{
   const LeftMinusMesh mesh;
   const auto faces=MakeUnstructuredInteriorFaceConnectivity(mesh);
   const auto partition=MakePartition(MakeCellPart(mesh),MakeInteriorFacePart<0,0>(faces));
   const auto p1=MakeLobattoFiniteElement(FiniteElementOrders<1>{});
   const auto local_space=MakeFiniteElementSpace(mesh,p1);
   const auto global_space=MakeMixedFiniteElementSpace(partition,std::tuple{p1},DGDirectSumNumbering{});
   Vector bound(4);auto* w=bound.WriteHostData();w[0]=2;w[1]=3;w[2]=6;w[3]=7;
   const auto local=MakeWeakFormContext(MakeTrialField<"u">(local_space),MakeTestField<"v">(local_space),
      MakeFiniteElementField<"w">(local_space,bound),MakeIntegrationDomain<"m">(mesh));
   const auto global=MakeWeakFormContext(MakeTrialField<"u">(global_space),MakeTestField<"v">(global_space),
      MakeFiniteElementField<"w">(global_space,bound),MakeIntegrationDomain<"m">(partition));
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3>{});
   bool ok=InteriorFlux(local,rule);
   ok=InteriorFlux(global,rule) && ok;
   const TrialSpace<"u"> u;const TestSpace<"v"> v;
   const auto inflow=MakeCoefficient<"inflow",PhysicalCoordinates>([] GENDIL_HOST_DEVICE (const std::array<Real,1>& x){return 2+x[0];});
   for(Real speed:{Real(-2),Real(0),Real(2)})
   {
      const auto beta=MakeVectorCoefficient<"beta">([speed] GENDIL_HOST_DEVICE () {return std::array<float,1>{float(speed)};});
      const auto residual=integrate(BoundaryFacets<"m">{},upwind(beta,u,inflow)*v);
      Dense matrix(16,0);matrix[0]=std::max(-speed,Real(0));matrix[15]=std::max(speed,Real(0));
      ok=CheckForm(lhs(residual),local,rule,matrix,4,4,"homogeneous boundary flux") && ok;
      const auto active_exterior=integrate(BoundaryFacets<"m">{},upwind(beta,u,2.*u)*v);
      Dense exterior_matrix(16,0);
      exterior_matrix[0]=(-speed>0?-speed:-2*speed);
      exterior_matrix[15]=(speed>0?speed:2*speed);
      ok=CheckForm(active_exterior,local,rule,exterior_matrix,4,4,"boundary active exterior state") && ok;
      const auto load=MakeGenericOperator<Policy>(rhs(residual),local,rule);
      Vector output(4);output=9;load(output);
      const Real expected[4]={-2*std::min(-speed,Real(0)),0,0,-4*std::min(speed,Real(0))};
      for(Integer i=0;i<4;++i)ok=Near(output.ReadHostData()[i],expected[i],"prescribed boundary inflow load") && ok;
   }
   // Poison the unselected state to distinguish lazy gating from eager 0*NaN.
   const auto coefficient=MakeCoefficient<"guard",FieldValue<"w">>([] GENDIL_HOST_DEVICE (Real value){return value;});
   for(Real speed:{Real(-2),Real(2)})
   {
      auto* values=bound.WriteHostData();
      for(Integer i=0;i<4;++i)
         values[i]=(speed>0 ? i<2 : i>=2) ? (i<2?3.:6.) : std::numeric_limits<Real>::quiet_NaN();
      const auto beta=MakeVectorCoefficient<"lazy_beta">([speed] GENDIL_HOST_DEVICE () {return std::array<Real,1>{speed};});
      const auto lazy=integrate(InteriorFacets<"m">{},upwind(beta,coefficient*u)*jump(v));
      Dense matrix(16,0); const Integer col=speed>0?1:2;
      matrix[1*4+col]=speed*(speed>0?3.:6.);matrix[2*4+col]=-matrix[1*4+col];
      ok=CheckForm(lazy,local,rule,matrix,4,4,"lazy local inactive flux state") && ok;
      ok=CheckForm(lazy,global,rule,matrix,4,4,"lazy global inactive flux state") && ok;
   }
   // Global evaluation retains its canonical-minus coefficient context, so an
   // index-dependent callback remains valid there and has literal value one.
   const auto index=MakeCoefficient<"index",ElementIndex>([] GENDIL_HOST_DEVICE (GlobalIndex i){return 1.+i;});
   Dense indexed(16,0);indexed[2*4+1]=1;
   ok=CheckForm(integrate(InteriorFacets<"m">{},index*minus(u)*plus(v)),global,rule,indexed,4,4,
      "global index-dependent transpose") && ok;
   // Physical-coordinate callbacks remain valid in local traversal.
   const auto physical=MakeCoefficient<"physical",PhysicalCoordinates>([] GENDIL_HOST_DEVICE (const std::array<Real,1>& x){return 2+x[0];});
   indexed[2*4+1]=3;
   ok=CheckForm(integrate(InteriorFacets<"m">{},physical*minus(u)*plus(v)),local,rule,indexed,4,4,
      "local physical-coordinate transpose") && ok;
   return ok;
}

// The right cell is rotated 180 degrees. Its common face is xi=1 and
// its native tangential coordinate runs opposite to the left cell's.
struct OrientedMeshFixture
{
   std::array<Real,12> nodes{0,0,1,0,0,1,1,1,2,0,2,1};
   std::array<int,8> restriction{0,1,2,3,5,3,4,1};
   HostDevicePointer<const int> pointer{};
   UnstructuredConformingConnectivity<HyperCube<2>> connectivity{2};
   OrientedMeshFixture()
   {
      pointer.host_pointer=restriction.data();
      for(Integer c=0;c<2;++c)for(Integer f=0;f<4;++f)
         connectivity[c].faces[f]={0,{},MakeReferencePermutation<2>(),{},{},true};
      const Permutation<2> rotated{{-1,-2}};
      connectivity[0].faces[2]={1,{},rotated,{},{},false};
      connectivity[1].faces[2]={0,{},rotated,{},{},false};
   }
   auto Mesh() const
   {return QuadMesh<1>{MakeFIFOView(nodes.data(),GlobalIndex(2),GlobalIndex(6)),
      MakeFIFOView(pointer,GlobalIndex(2),GlobalIndex(2),GlobalIndex(2)),connectivity,2};}
};
bool OrientedPrescribedTrace()
{
   const OrientedMeshFixture fixture;
   const auto mesh=fixture.Mesh();
   const auto faces=MakeUnstructuredInteriorFaceConnectivity(mesh);
   const auto partition=MakePartition(MakeCellPart(mesh),MakeInteriorFacePart<0,0>(faces));
   const auto fe=MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto local_space=MakeFiniteElementSpace(mesh,fe);
   const auto global_space=MakeMixedFiniteElementSpace(partition,std::tuple{fe},DGDirectSumNumbering{});
   Vector values(8);auto* data=values.WriteHostData();
   const Real prescribed[8]={2,2,5,5,5,5,7,7};
   std::copy_n(prescribed,8,data);
   const auto local=MakeWeakFormContext(MakeTrialField<"u">(local_space),MakeTestField<"v">(local_space),
      MakeFiniteElementField<"w">(local_space,values),MakeIntegrationDomain<"m">(mesh));
   const auto global=MakeWeakFormContext(MakeTrialField<"u">(global_space),MakeTestField<"v">(global_space),
      MakeFiniteElementField<"w">(global_space,values),MakeIntegrationDomain<"m">(partition));
   const auto c=MakeCoefficient<"trace",FieldValue<"w">>([] GENDIL_HOST_DEVICE (Real value){return value;});
   const auto form=integrate(InteriorFacets<"m">{},minus(c)*minus(TrialSpace<"u">{})*plus(TestSpace<"v">{}));
   Dense matrix(64,0);
   const Real points[3]={.5-std::sqrt(.15),.5,.5+std::sqrt(.15)};
   const Real weights[3]={5./18,4./9,5./18};
   for(Integer q=0;q<3;++q)
   {
      const Real t=points[q];
      const Real trial[2]={1-t,t},test[2]={t,1-t};
      for(Integer i=0;i<2;++i)for(Integer j=0;j<2;++j)
         matrix[(5+2*i)*8+(1+2*j)]+=weights[q]*(2+3*t)*test[i]*trial[j];
   }
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3,3>{});
   bool ok=CheckForm(form,local,rule,matrix,8,8,"oriented local one-sided varying coefficient");
   ok=CheckForm(form,global,rule,matrix,8,8,"oriented global one-sided varying coefficient") && ok;
   // The prescribed field has a context-resolved shape. Keeping its product
   // inside the trace exercises the nested fallback multiplication after the
   // original trial occurrence becomes test-linear during transposition.
   const auto raw_field=integrate(InteriorFacets<"m">{},
      minus(FiniteElementField<"w">{}*TrialSpace<"u">{})*plus(TestSpace<"v">{}));
   ok=CheckForm(raw_field,local,rule,matrix,8,8,"oriented local nested prescribed field") && ok;
   return CheckForm(raw_field,global,rule,matrix,8,8,"oriented global nested prescribed field") && ok;
}

// Rotate two unequal rectangles by an orthogonal map with columns n and t.
// Their common face has oblique normal n=(.6,.8), length .7, and reference
// coordinates xi=1 in the width-.5 cell and xi=0 in the width-1.2 cell.
struct ObliqueMeshFixture
{
   std::array<Real,12> nodes{0,0, .3,.4, -.56,.42, -.26,.82, 1.02,1.36, .46,1.78};
   std::array<int,8> restriction{0,1,2,3, 1,4,3,5};
   HostDevicePointer<const int> pointer{};
   UnstructuredConformingConnectivity<HyperCube<2>> connectivity{2};
   ObliqueMeshFixture()
   {
      pointer.host_pointer=restriction.data();
      for(Integer cell=0;cell<2;++cell) for(Integer face=0;face<4;++face)
         connectivity[cell].faces[face]={0,{},MakeReferencePermutation<2>(),{},{},true};
      connectivity[0].faces[2]={1,{},MakeReferencePermutation<2>(),{},{},false};
      connectivity[1].faces[0]={0,{},MakeReferencePermutation<2>(),{},{},false};
   }
   auto Mesh() const
   {
      return QuadMesh<1>{MakeFIFOView(nodes.data(),GlobalIndex(2),GlobalIndex(6)),
         MakeFIFOView(pointer,GlobalIndex(2),GlobalIndex(2),GlobalIndex(2)),connectivity,2};
   }
};

constexpr Real oblique_height=.7;
constexpr std::array<Real,2> oblique_width{.5,1.2};
constexpr Real oblique_normal[2]={.6,.8};
constexpr Real oblique_tangent[2]={-.8,.6};
constexpr Real oblique_matrix[2][2]={{2,1},{-.5,3}};

template<class Function>
void IndependentFaceQuadrature(Function&& function)
{
   const Real points[3]={.5-std::sqrt(.15),.5,.5+std::sqrt(.15)};
   const Real weights[3]={5./18,4./9,5./18};
   for(Integer q=0;q<3;++q) function(points[q],weights[q]);
}

// Explicit Q1 values and physical gradients in each cell's original numbering.
// This oracle does not use GenDiL basis, geometry, normals or projection helpers.
void ObliqueBasis(Integer cell,Real xi,Real eta,Real (&value)[4],Real (&gradient)[4][2])
{
   for(Integer j=0;j<2;++j) for(Integer i=0;i<2;++i)
   {
      const Integer node=i+2*j;
      const Real lx=i==0?1-xi:xi, ly=j==0?1-eta:eta;
      const Real dx=(i==0?-1.:1.)*ly/oblique_width[cell];
      const Real dy=lx*(j==0?-1.:1.)/oblique_height;
      value[node]=lx*ly;
      for(Integer d=0;d<2;++d)
         gradient[node][d]=oblique_normal[d]*dx+oblique_tangent[d]*dy;
   }
}

enum class ObliqueVectorTerm { NormalProjection, TangentialProjection, Normal, NormalDerivative };
Dense ExpectedObliqueVector(ObliqueVectorTerm term)
{
   Dense matrix(16*16,0);
   IndependentFaceQuadrature([&](Real eta,Real weight) {
      Real value[8]{},derivative[8]{};
      for(Integer cell=0;cell<2;++cell)
      {
         Real phi[4]{},gradient[4][2]{};
         ObliqueBasis(cell,cell==0?1.:0.,eta,phi,gradient);
         for(Integer node=0;node<4;++node)
         {
            value[4*cell+node]=phi[node];
            for(Integer d=0;d<2;++d)
               derivative[4*cell+node]+=oblique_normal[d]*gradient[node][d];
         }
      }
      for(Integer row=0;row<8;++row) for(Integer col=0;col<8;++col)
      {
         const Real row_sign=row<4?1.:-1.,col_sign=col<4?1.:-1.;
         for(Integer i=0;i<2;++i) for(Integer j=0;j<2;++j)
         {
            Real entry=0;
            if(term==ObliqueVectorTerm::NormalDerivative)
            {
               entry=.5*oblique_matrix[i][j]*derivative[col]*row_sign*value[row];
               if(col<4 && row>=4 && i==j) entry+=.2*value[col]*derivative[row];
            }
            else
            {
               Real coupling=0;
               for(Integer k=0;k<2;++k)
               {
                  if(term==ObliqueVectorTerm::NormalProjection)
                     coupling+=oblique_matrix[i][k]*oblique_normal[k]*oblique_normal[j];
                  else if(term==ObliqueVectorTerm::TangentialProjection)
                     coupling+=oblique_matrix[k][i]*((k==j?1.:0.)-oblique_normal[k]*oblique_normal[j]);
                  else coupling+=oblique_normal[k]*oblique_matrix[k][i]*oblique_normal[j];
               }
               const Real sign=term==ObliqueVectorTerm::TangentialProjection?col_sign:row_sign;
               entry=.5*coupling*sign*value[col]*value[row];
            }
            matrix[(8*i+row)*16+8*j+col]+=weight*oblique_height*entry;
         }
      }
   });
   return matrix;
}

Dense ExpectedObliqueScalarDerivative()
{
   Dense matrix(8*8,0);
   IndependentFaceQuadrature([&](Real eta,Real weight) {
      Real value[8]{},derivative[8]{};
      for(Integer cell=0;cell<2;++cell)
      {
         Real phi[4]{},gradient[4][2]{};
         ObliqueBasis(cell,cell==0?1.:0.,eta,phi,gradient);
         for(Integer node=0;node<4;++node)
         {
            value[4*cell+node]=phi[node];
            for(Integer d=0;d<2;++d)
               derivative[4*cell+node]+=oblique_normal[d]*gradient[node][d];
         }
      }
      for(Integer row=0;row<8;++row) for(Integer col=0;col<8;++col)
      {
         Real entry=.5*derivative[col]*(row<4?1.:-1.)*value[row];
         if(col<4 && row>=4) entry+=.3*value[col]*derivative[row];
         matrix[row*8+col]+=weight*oblique_height*entry;
      }
   });
   return matrix;
}

Dense ExpectedObliqueBoundary(bool vector_projection)
{
   const Integer dofs=vector_projection?16:8;
   Dense matrix(dofs*dofs,0);
   for(Integer cell=0;cell<2;++cell) for(Integer axis=0;axis<2;++axis)
      for(Integer side=0;side<2;++side)
      {
         if(axis==0 && ((cell==0 && side==1) || (cell==1 && side==0))) continue;
         Real normal[2]{};
         for(Integer d=0;d<2;++d)
            normal[d]=(side==0?-1.:1.)*(axis==0?oblique_normal[d]:oblique_tangent[d]);
         const Real measure=axis==0?oblique_height:oblique_width[cell];
         IndependentFaceQuadrature([&](Real q,Real weight) {
            Real value[4]{},gradient[4][2]{},derivative[4]{};
            ObliqueBasis(cell,axis==0?Real(side):q,axis==1?Real(side):q,value,gradient);
            for(Integer node=0;node<4;++node) for(Integer d=0;d<2;++d)
               derivative[node]+=normal[d]*gradient[node][d];
            for(Integer row=0;row<4;++row) for(Integer col=0;col<4;++col)
            {
               if(vector_projection)
               {
                  for(Integer i=0;i<2;++i) for(Integer j=0;j<2;++j)
                  {
                     Real coupling=0;
                     for(Integer k=0;k<2;++k)
                        coupling+=oblique_matrix[i][k]*normal[k]*normal[j]
                           +.4*oblique_matrix[k][i]*((k==j?1.:0.)-normal[k]*normal[j]);
                     matrix[(8*i+4*cell+row)*16+8*j+4*cell+col]
                        +=weight*measure*coupling*value[row]*value[col];
                  }
               }
               else matrix[(4*cell+row)*8+4*cell+col]
                  +=weight*measure*(derivative[col]*value[row]+.3*value[col]*derivative[row]);
            }
         });
      }
   return matrix;
}

template<class Context,class Rule>
bool CheckObliqueVectorInterior(const Context& context,const Rule& rule)
{
   const VectorTrialSpace<"U"> u;
   const VectorTestSpace<"V"> v;
   const InteriorFacets<"m"> faces;
   const auto b=MakeMatrixCoefficient<"B">([] GENDIL_HOST_DEVICE () {
      return std::array<std::array<Real,2>,2>{{{{2,1}},{{-.5,3}}}};
   });
   bool ok=CheckForm(integrate(faces,dot(b*normal_projection(average(u)),jump(v))),
      context,rule,ExpectedObliqueVector(ObliqueVectorTerm::NormalProjection),16,16,
      "oblique normal projection with average and nonsymmetric coefficient");
   ok=CheckForm(integrate(faces,dot(tangential_projection(jump(u)),b*average(v))),
      context,rule,ExpectedObliqueVector(ObliqueVectorTerm::TangentialProjection),16,16,
      "oblique tangential projection with average and nonsymmetric coefficient") && ok;
   ok=CheckForm(integrate(faces,dot(average(u),Normal{})*dot(b*jump(v),Normal{})),
      context,rule,ExpectedObliqueVector(ObliqueVectorTerm::Normal),16,16,"explicit oblique normal contractions") && ok;
   const auto derivative=dot(b*average(normal_derivative(u)),jump(v))
      +.2*dot(minus(u),plus(normal_derivative(v)));
   return CheckForm(integrate(faces,derivative),context,rule,
      ExpectedObliqueVector(ObliqueVectorTerm::NormalDerivative),16,16,
      "vector normal derivatives with unequal widths") && ok;
}

bool ProjectionsNormalsAndAverages()
{
   const ObliqueMeshFixture fixture;
   const auto mesh=fixture.Mesh();
   const auto faces=MakeUnstructuredInteriorFaceConnectivity(mesh);
   const auto partition=MakePartition(MakeCellPart(mesh),MakeInteriorFacePart<0,0>(faces));
   const auto scalar_fe=MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto vector_fe=MakeVectorFiniteElement(scalar_fe,scalar_fe);
   const auto scalar=MakeFiniteElementSpace(mesh,scalar_fe);
   const auto vector=MakeFiniteElementSpace(mesh,vector_fe);
   const auto global_scalar=MakeMixedFiniteElementSpace(partition,std::tuple{scalar_fe},DGDirectSumNumbering{});
   const auto global_vector=MakeMixedFiniteElementSpace(partition,std::tuple{vector_fe},DGDirectSumNumbering{});
   const auto local_context=MakeWeakFormContext(MakeTrialField<"u">(scalar),MakeTestField<"v">(scalar),
      MakeTrialField<"U">(vector),MakeTestField<"V">(vector),MakeIntegrationDomain<"m">(mesh));
   const auto global_context=MakeWeakFormContext(MakeTrialField<"u">(global_scalar),MakeTestField<"v">(global_scalar),
      MakeTrialField<"U">(global_vector),MakeTestField<"V">(global_vector),MakeIntegrationDomain<"m">(partition));
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3,3>{});
   bool ok=CheckObliqueVectorInterior(local_context,rule);
   ok=CheckObliqueVectorInterior(global_context,rule) && ok;
   const TrialSpace<"u"> u;
   const TestSpace<"v"> v;
   const auto scalar_form=integrate(InteriorFacets<"m">{},average(normal_derivative(u))*jump(v)
      +.3*minus(u)*plus(normal_derivative(v)));
   const auto scalar_matrix=ExpectedObliqueScalarDerivative();
   ok=CheckForm(scalar_form,local_context,rule,scalar_matrix,8,8,"local scalar normal derivative and average") && ok;
   ok=CheckForm(scalar_form,global_context,rule,scalar_matrix,8,8,"global scalar normal derivative and average") && ok;
   const VectorTrialSpace<"U"> U;
   const VectorTestSpace<"V"> V;
   const auto b=MakeMatrixCoefficient<"B">([] GENDIL_HOST_DEVICE () {
      return std::array<std::array<Real,2>,2>{{{{2,1}},{{-.5,3}}}};
   });
   const BoundaryFacets<"m"> boundary;
   ok=CheckForm(integrate(boundary,dot(b*normal_projection(U),V)+.4*dot(tangential_projection(U),b*V)),
      local_context,rule,ExpectedObliqueBoundary(true),16,16,"boundary oblique normal and tangent projections") && ok;
   return CheckForm(integrate(boundary,normal_derivative(u)*v+.3*u*normal_derivative(v)),local_context,rule,
      ExpectedObliqueBoundary(false),8,8,"boundary outward normal derivatives") && ok;
}

struct WeightedRestriction {using dof_shape_type=std::index_sequence<2>;static constexpr Integer tensor_dim=1;};
GlobalIndex GetNumberOfLocalDofs(const WeightedRestriction&){return 4;}
GlobalIndex GetNumberOfGlobalDofs(const WeightedRestriction&){return 5;}
GlobalIndex GetAlgebraicDofExtent(const WeightedRestriction&){return 7;}
template<class Visitor>
void ForEachRestrictionEntry(const WeightedRestriction&,GlobalIndex cell,const std::array<GlobalIndex,1>& local,Visitor&& visitor)
{
   const GlobalIndex row=2*cell+local[0];
   visitor(row+1,RestrictionSign{row%2==0});visitor(GlobalIndex{5},Real(.5));
}
} // namespace form_transpose_facet_test
namespace gendil
{
template<> inline constexpr size_t static_restriction_entry_count_v<form_transpose_facet_test::WeightedRestriction> = 2;
}
namespace form_transpose_facet_test
{
bool WeightedFacet()
{
   const LeftMinusMesh mesh;
   const auto space=FiniteElementSpace(mesh,MakeLobattoFiniteElement(FiniteElementOrders<1>{}),WeightedRestriction{});
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(space),MakeTestField<"v">(space),MakeIntegrationDomain<"m">(mesh));
   const TrialSpace<"u"> u;const TestSpace<"v"> v;
   const auto form=integrate(InteriorFacets<"m">{},2.*minus(u)*plus(v)-3.*plus(u)*minus(v));
   // Literal restriction entries, independent of production gather/scatter.
   Real restriction[4][7]{};
   for(Integer i=0;i<4;++i){restriction[i][i+1]=(i%2==0?-1:1);restriction[i][5]=.5;}
   Real local[4][4]{};local[2][1]=2;local[1][2]=-3;
   Dense matrix(49,0);
   for(Integer i=0;i<7;++i)for(Integer j=0;j<7;++j)
      for(Integer r=0;r<4;++r)for(Integer c=0;c<4;++c)
         matrix[i*7+j]+=restriction[r][i]*local[r][c]*restriction[c][j];
   return CheckForm(form,context,MakeIntegrationRule(IntegrationRuleNumPoints<3>{}),matrix,7,7,
      "signed overlapping weighted restrictions");
}
} // namespace form_transpose_facet_test
int main()
{
   using namespace form_transpose_facet_test;
   bool ok=RectangularLocalAndGlobal();
   ok=FluxAndBoundary() && ok;
   ok=WeightedFacet() && ok;
   ok=OrientedPrescribedTrace() && ok;
   ok=ProjectionsNormalsAndAverages() && ok;
   if(ok) std::cout << "Bilinear form transpose matches literal facet matrices and fluxes.\n";
   return ok?0:1;
}
