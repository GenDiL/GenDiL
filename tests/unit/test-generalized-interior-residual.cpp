// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include "generalized-interior-test-utils.hpp"

namespace weighted_interior
{
using namespace gendil;
struct Restriction
{
   using dof_shape_type = std::index_sequence<2,2>;
   static constexpr Integer tensor_dim = 2;
};
GENDIL_HOST_DEVICE GlobalIndex GetNumberOfLocalDofs(const Restriction&) { return 8; }
GENDIL_HOST_DEVICE GlobalIndex GetNumberOfGlobalDofs(const Restriction&) { return 9; }
GENDIL_HOST_DEVICE GlobalIndex GetAlgebraicDofExtent(const Restriction&) { return 12; }
template<class Visitor>
GENDIL_HOST_DEVICE void ForEachRestrictionEntry(const Restriction&,GlobalIndex cell,
   const std::array<GlobalIndex,2>& local,Visitor&& visitor)
{
   const auto row = 4*cell+local[0]+2*local[1];
   visitor(row+1,RestrictionSign{row%2 == 0});
   visitor(GlobalIndex{9},Real{.5});
}
}
namespace gendil
{
template<> inline constexpr size_t static_restriction_entry_count_v<weighted_interior::Restriction> = 2;
}

namespace
{
using namespace gendil;
using namespace interior_test;

template<class Policy>
bool CheckCoupledMesh()
{
   constexpr Real hx = .5, hy = .75, ox = .25, oy = -.5;
   constexpr size_t input_extent = 64, output_extent = 32;
   CartesianMesh<2> mesh({2, 2}, {hx, hy}, {ox, oy}, false);
   const auto p1 = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   const auto p2 = MakeLobattoFiniteElement(FiniteElementOrders<2, 2>{});
   const auto p0 = MakeLegendreFiniteElement(FiniteElementOrders<0, 0>{});
   auto u_space = MakeFiniteElementSpace(mesh, p1, ContiguousL2RestrictionSpecification{2, input_extent});
   auto p_space = MakeFiniteElementSpace(mesh, p2, ContiguousL2RestrictionSpecification{22, input_extent});
   auto v_space = MakeFiniteElementSpace(mesh, p1, ContiguousL2RestrictionSpecification{3, output_extent});
   auto q_space = MakeFiniteElementSpace(mesh, p0, ContiguousL2RestrictionSpecification{24, output_extent});
   auto f_space = MakeFiniteElementSpace(mesh, p1);
   Vector x(input_extent), bound(16), y(output_extent+3);
   x = 101;
   auto* input = x.ReadWriteHostData();
   auto* data = bound.WriteHostData();
   for (size_t cell = 0; cell < 4; ++cell)
   {
      for (size_t j = 0; j < 2; ++j)
      for (size_t i = 0; i < 2; ++i)
      {
         const Real xx = ox+hx*(cell%2+i), yy = oy+hy*(cell/2+j);
         input[2+4*cell+i+2*j] = U(cell, xx, yy);
         data[4*cell+i+2*j] = F(cell, xx, yy);
      }
      for (size_t j = 0; j < 3; ++j)
      for (size_t i = 0; i < 3; ++i)
      {
         const Real xx = ox+hx*(cell%2+Real(i)/2), yy = oy+hy*(cell/2+Real(j)/2);
         input[22+9*cell+i+3*j] = P(cell, xx, yy);
      }
   }
   const auto context = MakeWeakFormContext(MakeTrialField<"u">(u_space),
      MakeTrialField<"p">(p_space), MakeTestField<"v">(v_space), MakeTestField<"q">(q_space),
      MakeFiniteElementField<"f">(f_space, MakeReadOnlyElementTensorView<Policy>(f_space, bound)),
      MakeIntegrationDomain<"mesh">(mesh));
   const TrialSpace<"u"> u;
   const TrialSpace<"p"> p;
   const TestSpace<"v"> v;
   const TestSpace<"q"> q;
   const FiniteElementField<"f"> f;
   const InteriorFacets<"mesh"> interior;
   auto read_u = MakeCoefficient<"read_u", FieldValue<"u">>(
      [] GENDIL_HOST_DEVICE (Real value) { return value; });
   // The callback occurs before the cell declaration that makes u active.
   const auto form = integrate(interior, average(read_u)*jump(v))
      + integrate(Cells<"mesh">{}, u*u*v)
      + integrate(interior, jump(u)*average(p)*jump(v))
      + integrate(interior, jump(p)*jump(q))
      + integrate(BoundaryFacets<"mesh">{}, f*v)
      + integrate(interior, jump(f)*jump(v));
   auto op = MakeWeakFormResidualOperator<Policy>(form, context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{}));
   std::array<Real, 16> rows{};
   std::array<Real, output_extent> expected{};
   boundary_test::CellPoints(2, 2, hx, hy, ox, oy, [&](const PointData& point)
   {
      const Real uq = U(point.cell, point.x, point.y);
      AddP1(rows, point, hx, hy, uq*uq);
   });
   boundary_test::BoundaryPoints(2, 2, hx, hy, ox, oy, [&](const PointData& point)
   {
      AddP1(rows, point, hx, hy, F(point.cell, point.x, point.y));
   });
   InteriorPoints(2, 2, hx, hy, ox, oy, [&](const PointData& point, size_t neighbor)
   {
      const Real uc = U(point.cell, point.x, point.y), un = U(neighbor, point.x, point.y);
      const Real pc = P(point.cell, point.x, point.y), pn = P(neighbor, point.x, point.y);
      const Real fc = F(point.cell, point.x, point.y), fn = F(neighbor, point.x, point.y);
      AddP1(rows, point, hx, hy, (uc+un)/2 + (uc-un)*(pc+pn)/2 + fc-fn);
      expected[24+point.cell] += point.weight*(pc-pn);
   });
   for (size_t i = 0; i < rows.size(); ++i) { expected[3+i] = rows[i]; }
   bool ok = op.InputSize() == input_extent && op.OutputSize() == output_extent;
   observed_facet_vectors::VectorRef observed_x{x}, observed_y{y};
   y = 9; op(observed_x,observed_y);
   ok = observed_x.reads == 1 && observed_x.writes == 0 && observed_x.read_writes == 0 &&
      observed_y.reads == 0 && observed_y.writes == 1 && observed_y.read_writes == 0 && ok;
   ok = Check(y, expected, "coupled mixed interior residual") && ok;
   op(x,y); ok = Check(y, expected, "interior replacement") && ok;
   y = -3; op.ApplyAdd(x,y); ok = Check(y, expected, "interior addition", 1, -3) && ok;
   op.ApplyAdd(x,y); ok = Check(y, expected, "repeated interior addition", 2, -3) && ok;
   Vector other_x(input_extent), other_y(output_extent+3);
   CopyVector<false>(x, other_x);
   op(other_x,other_y); ok = Check(other_y, expected, "rebound interior buffers") && ok;
   return ok;
}

template<class Policy>
bool CheckCachedCurrentAndNeighbor()
{
   constexpr Real hx = .5, hy = .75;
   CartesianMesh<2> mesh({2, 2}, {hx, hy}, {0., 0.}, false);
   auto space = MakeFiniteElementSpace(mesh, MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{}));
   auto context = MakeWeakFormContext(MakeTrialField<"u">(space), MakeTestField<"u">(space),
      MakeIntegrationDomain<"mesh">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   auto op = MakeWeakFormResidualOperator<Policy>(
      integrate(InteriorFacets<"mesh">{}, jump(u)*jump(v)), context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4, 4>{}));
   Vector x(16), y(19);
   auto* data = x.WriteHostData();
   for (size_t cell = 0; cell < 4; ++cell)
   for (size_t j = 0; j < 2; ++j)
   for (size_t i = 0; i < 2; ++i)
      { data[4*cell+i+2*j] = U(cell,hx*(cell%2+i),hy*(cell/2+j)); }
   std::array<Real, 16> expected{};
   InteriorPoints(2,2,hx,hy,0,0,[&](const PointData& point,size_t neighbor)
   {
      AddP1(expected,point,hx,hy,U(point.cell,point.x,point.y)-U(neighbor,point.x,point.y));
   });
   y = 7; op(x,y);
   bool ok = Check(y,expected,"cached current versus neighbor under same field name");
   y = -2; op.ApplyAdd(x,y);
   return Check(y,expected,"cached interior addition",1,-2) && ok;
}

template<class Policy, bool Compound = false>
bool CheckActiveUpwindSelector()
{
   constexpr Real hx = .5, hy = .75;
   CartesianMesh<2> mesh({2,2},{hx,hy},{0.,0.},false);
   auto u_space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1,1>{}),
      ContiguousL2RestrictionSpecification{0,20});
   auto p_space = MakeFiniteElementSpace(mesh,
      MakeLegendreFiniteElement(FiniteElementOrders<0,0>{}),
      ContiguousL2RestrictionSpecification{16,20});
   auto v_space = MakeFiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1,1>{}));
   auto context = MakeWeakFormContext(MakeTrialField<"u">(u_space),
      MakeTrialField<"p">(p_space),MakeTestField<"v">(v_space),
      MakeIntegrationDomain<"mesh">(mesh));
   TrialSpace<"u"> u;
   TrialSpace<"p"> p;
   TestSpace<"v"> v;
   auto velocity = MakeVectorCoefficient<"velocity",FieldValue<"p">>(
      [] GENDIL_HOST_DEVICE (Real value) {
         return std::array<Real,2>{value,1-2*value};
      });
   // The selector's P0 field and the P1 field use distinct spaces. The
   // non-compound case resolves p through its later cell declaration.
   auto form = [&]
   {
      if constexpr (Compound)
      {
         return integrate(InteriorFacets<"mesh">{},upwind(average(velocity),u+p)*jump(v))
            + integrate(Cells<"mesh">{},p*v);
      }
      else
      {
         return integrate(InteriorFacets<"mesh">{},upwind(average(velocity),u)*jump(v))
            + integrate(Cells<"mesh">{},p*v);
      }
   }();
   auto op = MakeWeakFormResidualOperator<Policy>(form,context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4,4>{}));
   Vector x(20),y(19);
   auto* input = x.WriteHostData();
   for(size_t cell=0;cell<4;++cell)
   for(size_t j=0;j<2;++j)
   for(size_t i=0;i<2;++i)
      { input[4*cell+i+2*j]=U(cell,hx*(cell%2+i),hy*(cell/2+j)); }
   const std::array<Real,4> speeds{-1.5,.5,2,-.25};
   bool ok = true;
   for(const Real sign : {Real{1},Real{-1}})
   {
      input=x.ReadWriteHostData();
      for(size_t cell=0;cell<4;++cell) { input[16+cell]=sign*speeds[cell]; }
      std::array<Real,16> expected{};
      boundary_test::CellPoints(2,2,hx,hy,0,0,[&](const PointData& point) {
         AddP1(expected,point,hx,hy,sign*speeds[point.cell]);
      });
      InteriorPoints(2,2,hx,hy,0,0,[&](const PointData& point,size_t neighbor) {
         const Real speed=sign*(speeds[point.cell]+speeds[neighbor])/2;
         const Real normal_speed=speed*point.nx+(1-2*speed)*point.ny;
         const size_t upwind_cell=normal_speed>=0 ? point.cell : neighbor;
         Real transported=U(upwind_cell,point.x,point.y);
         if constexpr (Compound) { transported+=sign*speeds[upwind_cell]; }
         AddP1(expected,point,hx,hy,normal_speed*transported);
      });
      y=9;op(x,y);
      ok=Check(y,expected,Compound ? "heterogeneous transported upwind fields" :
         "distinct active upwind selector") && ok;
      y=-2;op.ApplyAdd(x,y);
      ok=Check(y,expected,Compound ? "heterogeneous upwind addition" :
         "active upwind selector addition",1,-2) && ok;
   }
   return ok;
}

template<class Policy>
bool CheckWeightedInteriorEndpoints()
{
   constexpr Real hx = .5, hy = .75;
   CartesianMesh<2> mesh({2,1},{hx,hy},{0.,0.},false);
   auto space = FiniteElementSpace(mesh,
      MakeLobattoFiniteElement(FiniteElementOrders<1,1>{}),weighted_interior::Restriction{});
   auto context = MakeWeakFormContext(MakeTrialField<"u">(space),MakeTestField<"v">(space),
      MakeIntegrationDomain<"mesh">(mesh));
   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   auto form = integrate(InteriorFacets<"mesh">{},jump(u)*jump(v))
      + integrate(InteriorFacets<"mesh">{},2.0*jump(v));
   auto op = MakeWeakFormResidualOperator<Policy>(form,context,
      MakeIntegrationRule(IntegrationRuleNumPoints<4,4>{}));
   Vector x(12),y(15); x=0;
   auto* data=x.ReadWriteHostData();
   std::array<Real,8> local{};
   for(size_t row=0;row<8;++row)
   {
      data[row+1]=Real(row+1);
      local[row]=(row%2 == 0 ? -1 : 1)*Real(row+1)+1;
   }
   data[9]=2;
   auto value=[&](size_t cell,Real xi,Real eta) {
      return local[4*cell]*(1-xi)*(1-eta)+local[4*cell+1]*xi*(1-eta)
         +local[4*cell+2]*(1-xi)*eta+local[4*cell+3]*xi*eta;
   };
   std::array<Real,8> rows{};
   InteriorPoints(2,1,hx,hy,0,0,[&](const PointData& point,size_t neighbor) {
      const Real current=value(point.cell,point.xi,point.eta);
      const Real adjacent=value(neighbor,1-point.xi,point.eta);
      AddP1(rows,point,hx,hy,current-adjacent+2);
   });
   std::array<Real,12> expected{};
   for(size_t row=0;row<8;++row)
   {
      expected[row+1]=(row%2 == 0 ? -1 : 1)*rows[row];
      expected[9]+=rows[row]/2;
   }
   y=9;op(x,y);
   bool ok=Check(y,expected,"weighted signed interior restrictions");
   y=-2;op.ApplyAdd(x,y);
   return Check(y,expected,"overlapping additive interior restriction",1,-2) && ok;
}
}
int main()
{
   bool ok = CheckCoupledMesh<SerialKernelConfiguration>() &&
             CheckCachedCurrentAndNeighbor<SerialKernelConfiguration>() &&
             CheckActiveUpwindSelector<SerialKernelConfiguration>() &&
             CheckActiveUpwindSelector<SerialKernelConfiguration,true>() &&
             CheckWeightedInteriorEndpoints<SerialKernelConfiguration>();
#ifdef GENDIL_USE_DEVICE
   // Local facet traversal requires one cell per device block.
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<4,4>,2,1>;
   ok = CheckCoupledMesh<Policy>() && CheckCachedCurrentAndNeighbor<Policy>() &&
      CheckActiveUpwindSelector<Policy>() &&
      CheckActiveUpwindSelector<Policy,true>() &&
      CheckWeightedInteriorEndpoints<Policy>() && ok;
#endif
   return ok ? 0 : 1;
}
