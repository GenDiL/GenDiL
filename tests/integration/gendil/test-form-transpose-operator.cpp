// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace form_transpose_operator_test
{
using namespace gendil;
using Policy = SerialKernelConfiguration;
using Dense = std::vector<Real>;

bool Near(Real actual, Real expected, const char* label, Real tolerance = 2.e-12)
{
   if (std::isfinite(actual) && std::isfinite(expected) &&
       std::abs(actual-expected) <= tolerance*(1+std::abs(expected))) return true;
   std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
   return false;
}
Dense Transposed(const Dense& matrix, Integer rows, Integer cols)
{
   Dense result(matrix.size());
   for (Integer i=0;i<rows;++i) for (Integer j=0;j<cols;++j)
      result[j*rows+i]=matrix[i*cols+j];
   return result;
}
template<class Operator>
bool CheckOperator(const Operator& op, const Dense& matrix, Integer rows, Integer cols, const char* label)
{
   if (op.InputSize()!=cols || op.OutputSize()!=rows)
   { std::cerr << label << ": incorrect extents\n"; return false; }
   bool ok=true;
   Vector x(cols),y(rows);
   for (Integer j=0;j<cols;++j)
   {
      auto* data=x.WriteHostData(); std::fill_n(data,cols,Real(0)); data[j]=1;
      y=19; op(x,y);
      for (Integer i=0;i<rows;++i) ok=Near(y.ReadHostData()[i],matrix[i*cols+j],label) && ok;
   }
   auto* data=x.WriteHostData();
   for(Integer j=0;j<cols;++j) data[j]=.4+.2*j-.07*j*j;
   Dense expected(rows,0);
   for(Integer i=0;i<rows;++i) for(Integer j=0;j<cols;++j)
      expected[i]+=matrix[i*cols+j]*data[j];
   y=91; op(x,y);
   for(Integer i=0;i<rows;++i) ok=Near(y.ReadHostData()[i],expected[i],label) && ok;
   auto* out=y.WriteHostData(); for(Integer i=0;i<rows;++i) out[i]=.7-.3*i;
   for(Integer repetition=1;repetition<=2;++repetition)
   {
      op.ApplyAdd(x,y);
      for(Integer i=0;i<rows;++i)
         ok=Near(y.ReadHostData()[i],.7-.3*i+repetition*expected[i],label) && ok;
   }
   return ok;
}
template<class Form,class Context,class Rule>
bool CheckForm(const Form& form,const Context& context,const Rule& rule,
               const Dense& expected,Integer rows,Integer cols,const char* label)
{
   const auto forward=MakeGenericOperator<Policy>(form,context,rule);
   const auto reverse=MakeGenericOperator<Policy>(transpose(form),context,rule);
   const auto twice=MakeGenericOperator<Policy>(transpose(transpose(form)),context,rule);
   bool ok=CheckOperator(forward,expected,rows,cols,label);
   ok=CheckOperator(reverse,Transposed(expected,rows,cols),cols,rows,label) && ok;
   ok=CheckOperator(twice,expected,rows,cols,label) && ok;
   Vector x(cols),y(rows),ax(rows),aty(cols);
   auto* xv=x.WriteHostData(); auto* yv=y.WriteHostData();
   for(Integer j=0;j<cols;++j) xv[j]=.3-.7*j;
   for(Integer i=0;i<rows;++i) yv[i]=.8+.4*i-.2*i*i;
   forward(x,ax); reverse(y,aty);
   Real left=0,right=0,scale=1;
   for(Integer i=0;i<rows;++i) { const Real term=yv[i]*ax.ReadHostData()[i];left+=term;scale+=std::abs(term); }
   for(Integer j=0;j<cols;++j) { const Real term=xv[j]*aty.ReadHostData()[j];right+=term;scale+=std::abs(term); }
   return Near(left/scale,right/scale,label) && ok;
}

struct StoredMatrix
{
   const Real* scale;
   StoredMatrix()=delete;
   explicit StoredMatrix(const Real& value):scale(&value) {}
   auto operator()() const
   { return std::array<std::array<Real,2>,2>{{{{2*(*scale),*scale}},{{-.5*(*scale),3*(*scale)}}}}; }
};
bool GradientAndCapturedData()
{
   constexpr Real hx=.4,hy=.7;
   CartesianMesh<2> mesh({1,1},{hx,hy},{.2,-.3},false);
   const auto fe=MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto space=MakeFiniteElementSpace(mesh,fe);
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(space),MakeTestField<"v">(space),
      MakeIntegrationDomain<"m">(mesh));
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3,3>{});
   Real factor=1;
   const auto B=MakeMatrixCoefficient<"B">(StoredMatrix{factor});
   const auto form=integrate(Cells<"m">{},dot(B*grad(TrialSpace<"u">{}),grad(TestSpace<"v">{})));
   const auto forward=MakeGenericOperator<Policy>(form,context,rule);
   const auto reversed=transpose(form);
   const auto reverse=MakeGenericOperator<Policy>(reversed,context,rule);
   const auto twice=MakeGenericOperator<Policy>(transpose(reversed),context,rule);
   const auto sum=MakeGenericOperator<Policy>(reversed+reversed,context,rule);
   const auto independent=integrate(Cells<"m">{},
      dot(grad(TrialSpace<"v">{}),B*grad(TestSpace<"u">{})));
   const auto reference=MakeGenericOperator<Policy>(independent,context,rule);
   bool ok=true;
   for(Real value: {Real(1),Real(-.75)})
   {
      factor=value;
      Dense matrix(16,0);
      // Explicit Q1 derivatives and scalar matrix entries, independent of DSL
      // pullbacks, matrix multiplication and finite-element interpolation.
      const std::array<Real,2> q{.5-std::sqrt(1./12),.5+std::sqrt(1./12)};
      const Real entries[2][2]={{2,1},{-.5,3}};
      for(Real eta:q) for(Real xi:q)
      {
         Real derivative[4][2]{};
         for(Integer j=0;j<2;++j) for(Integer i=0;i<2;++i)
         {
            derivative[i+2*j][0]=(i==0?-1.:1.)*(j==0?1-eta:eta)/hx;
            derivative[i+2*j][1]=(i==0?1-xi:xi)*(j==0?-1.:1.)/hy;
         }
         for(Integer row=0;row<4;++row) for(Integer col=0;col<4;++col)
            for(Integer d=0;d<2;++d) for(Integer e=0;e<2;++e)
               matrix[row*4+col]+=.25*hx*hy*factor*derivative[row][d]*entries[d][e]*derivative[col][e];
      }
      auto transposed=Transposed(matrix,4,4);
      ok=CheckOperator(forward,matrix,4,4,"nonsymmetric gradient") && ok;
      ok=CheckOperator(reverse,transposed,4,4,"transposed gradient with updated capture") && ok;
      ok=CheckOperator(twice,matrix,4,4,"double transpose") && ok;
      ok=CheckOperator(reference,transposed,4,4,"independent transposed contraction") && ok;
      for(auto& entry:transposed) entry*=2;
      ok=CheckOperator(sum,transposed,4,4,"sum containing transpose") && ok;
   }
   return ok;
}

bool RectangularComponentsAndOuter()
{
   CartesianMesh<1> mesh({1},{1.5},{0.},false);
   const auto p0=MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto u_space=MakeFiniteElementSpace(mesh,MakeVectorFiniteElement(p0,p0));
   const auto v_space=MakeFiniteElementSpace(mesh,MakeVectorFiniteElement(p0,p0,p0));
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(u_space),MakeTestField<"v">(v_space),
      MakeIntegrationDomain<"m">(mesh));
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<2>{});
   const auto C=MakeMatrixCoefficient<"C">([] {
      return std::array<std::array<Real,2>,3>{{{{2,-1}},{{.5,3}},{{-2,.75}}}};
   });
   bool ok=CheckForm(integrate(Cells<"m">{},dot(C*VectorTrialSpace<"u">{},VectorTestSpace<"v">{})),
      context,rule,Dense{3,-1.5,.75,4.5,-3,1.125},3,2,"rectangular components");
   const auto square_context=MakeWeakFormContext(MakeTrialField<"u">(u_space),MakeTestField<"v">(u_space),
      MakeIntegrationDomain<"m">(mesh));
   const auto a=MakeVectorCoefficient<"a">([] {return std::array<Real,2>{2,-.5};});
   const auto b=MakeVectorCoefficient<"b">([] {return std::array<Real,2>{.75,-3};});
   // inner(a outer u, v outer b) = sum_ij a_i u_j v_i b_j.
   const auto outer_form=integrate(Cells<"m">{},inner(outer(a,VectorTrialSpace<"u">{}),
      outer(VectorTestSpace<"v">{},b)));
   ok=CheckForm(outer_form,square_context,rule,Dense{2.25,-9,-.5625,2.25},2,2,
      "trial in second outer operand") && ok;
   return ok;
}

struct Q1Values
{
   Real value[4]{};
   Real gradient[4][2]{};
};
Q1Values Q1At(Real xi,Real eta,Real hx,Real hy)
{
   Q1Values result;
   for(Integer j=0;j<2;++j) for(Integer i=0;i<2;++i)
   {
      const Integer node=i+2*j;
      result.value[node]=(i==0?1-xi:xi)*(j==0?1-eta:eta);
      result.gradient[node][0]=(i==0?-1.:1.)*(j==0?1-eta:eta)/hx;
      result.gradient[node][1]=(i==0?1-xi:xi)*(j==0?-1.:1.)/hy;
   }
   return result;
}

bool DivergenceAndTensorNodes()
{
   constexpr Real hx=.6,hy=1.3;
   CartesianMesh<2> mesh({1,1},{hx,hy},{-.2,.4},false);
   const auto q1=MakeLobattoFiniteElement(FiniteElementOrders<1,1>{});
   const auto p0=MakeLegendreFiniteElement(FiniteElementOrders<0,0>{});
   const auto vector_q1=MakeFiniteElementSpace(mesh,MakeVectorFiniteElement(q1,q1));
   const auto scalar_q1=MakeFiniteElementSpace(mesh,q1);
   const auto vector_p0=MakeFiniteElementSpace(mesh,MakeVectorFiniteElement(p0,p0));
   const auto divergence_context=MakeWeakFormContext(MakeTrialField<"u">(vector_q1),
      MakeTestField<"v">(scalar_q1),MakeIntegrationDomain<"m">(mesh));
   const auto tensor_context=MakeWeakFormContext(MakeTrialField<"u">(vector_q1),
      MakeTestField<"v">(vector_p0),MakeIntegrationDomain<"m">(mesh));
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3,3>{});
   const VectorTrialSpace<"u"> u;
   const VectorTestSpace<"v"> v;
   const auto C=MakeMatrixCoefficient<"C">([] {
      return std::array<std::array<Real,2>,2>{{{{2,-1}},{{.5,3}}}};
   });
   const auto b=MakeVectorCoefficient<"b">([] {return std::array<Real,2>{.75,-1.25};});
   Dense divergence(4*8,0),symmetric(2*8,0),skew_part(2*8,0),transposed(2*8,0);
   const Real entries[2][2]={{2,-1},{.5,3}};
   const Real direction[2]={.75,-1.25};
   const std::array<Real,2> points{.5-std::sqrt(1./12),.5+std::sqrt(1./12)};
   // Explicit polynomial basis integration: grad(e_c N)(a,b)=delta_ac*d_b N.
   // Separate scalar contractions distinguish sym, skew, and matrix transpose,
   // including their off-diagonal signs. C acts on the left and cannot commute.
   for(Real eta:points) for(Real xi:points)
   {
      const auto basis=Q1At(xi,eta,hx,hy);
      const Real weight=hx*hy/4;
      for(Integer component=0;component<2;++component) for(Integer node=0;node<4;++node)
      {
         const Integer col=4*component+node;
         for(Integer row=0;row<4;++row)
            divergence[row*8+col]+=weight*basis.value[row]*basis.gradient[node][component];
         for(Integer row=0;row<2;++row) for(Integer a=0;a<2;++a) for(Integer b=0;b<2;++b)
         {
            const Real ab=a==component?basis.gradient[node][b]:0;
            const Real ba=b==component?basis.gradient[node][a]:0;
            const Real factor=weight*entries[row][a]*direction[b];
            symmetric[row*8+col]+=factor*(ab+ba)/2;
            skew_part[row*8+col]+=factor*(ab-ba)/2;
            transposed[row*8+col]+=factor*ba;
         }
      }
   }
   bool ok=CheckForm(integrate(Cells<"m">{},div(u)*TestSpace<"v">{}),
      divergence_context,rule,divergence,4,8,"rectangular divergence");
   ok=CheckForm(integrate(Cells<"m">{},inner(C*sym(grad(u)),outer(v,b))),
      tensor_context,rule,symmetric,2,8,"symmetric gradient with matrix product") && ok;
   ok=CheckForm(integrate(Cells<"m">{},inner(C*skew(grad(u)),outer(v,b))),
      tensor_context,rule,skew_part,2,8,"skew gradient with matrix product") && ok;
   ok=CheckForm(integrate(Cells<"m">{},inner(C*transpose(grad(u)),outer(v,b))),
      tensor_context,rule,transposed,2,8,"pointwise transpose inside form transpose") && ok;
   return ok;
}

bool CurlNode()
{
   constexpr Real hx=.7,hy=1.1,hz=1.4;
   CartesianMesh<3> mesh({1,1,1},{hx,hy,hz},{.2,-.1,.3},false);
   const auto q1=MakeLobattoFiniteElement(FiniteElementOrders<1,1,1>{});
   const auto p0=MakeLegendreFiniteElement(FiniteElementOrders<0,0,0>{});
   const auto us=MakeFiniteElementSpace(mesh,MakeVectorFiniteElement(q1,q1,q1));
   const auto vs=MakeFiniteElementSpace(mesh,MakeVectorFiniteElement(p0,p0,p0));
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(us),MakeTestField<"v">(vs),
      MakeIntegrationDomain<"m">(mesh));
   const auto B=MakeMatrixCoefficient<"B">([] {
      return std::array<std::array<Real,3>,3>{{{{2,-1,.5}},{{.25,3,-2}},{{-1.5,.75,1}}}};
   });
   Dense matrix(3*24,0);
   const Real entries[3][3]={{2,-1,.5},{.25,3,-2},{-1.5,.75,1}};
   for(Integer k=0;k<2;++k) for(Integer j=0;j<2;++j) for(Integer i=0;i<2;++i)
   {
      // Exact cell integrals of the three derivatives of a trilinear nodal
      // basis. curl(e_c N)=grad(N) cross e_c, written out without DSL helpers.
      const Real gx=(i==0?-1.:1.)*hy*hz/4;
      const Real gy=(j==0?-1.:1.)*hx*hz/4;
      const Real gz=(k==0?-1.:1.)*hx*hy/4;
      const Real curls[3][3]={{0,gz,-gy},{-gz,0,gx},{gy,-gx,0}};
      for(Integer component=0;component<3;++component) for(Integer row=0;row<3;++row)
         for(Integer d=0;d<3;++d)
            matrix[row*24+component*8+i+2*j+4*k]+=entries[row][d]*curls[component][d];
   }
   return CheckForm(integrate(Cells<"m">{},dot(B*curl(VectorTrialSpace<"u">{}),VectorTestSpace<"v">{})),
      context,MakeIntegrationRule(IntegrationRuleNumPoints<2,2,2>{}),matrix,3,24,"rectangular curl");
}

bool LivePrescribedFieldAndFallbackProduct()
{
   constexpr Real length=1.5;
   CartesianMesh<1> mesh({1},{length},{-.2},false);
   const auto us=MakeFiniteElementSpace(mesh,MakeLobattoFiniteElement(FiniteElementOrders<1>{}));
   const auto vs=MakeFiniteElementSpace(mesh,MakeLegendreFiniteElement(FiniteElementOrders<0>{}));
   Vector values(2);
   auto* data=values.WriteHostData();data[0]=1.25;data[1]=-.75;
   const auto bound=MakeReadOnlyElementTensorView<Policy>(us,values);
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(us),MakeTestField<"v">(vs),
      MakeFiniteElementField<"f">(us,bound),MakeIntegrationDomain<"m">(mesh));
   const FiniteElementField<"f"> f;
   const TrialSpace<"u"> u;
   // f has no declared expression shape, so this uses the fallback product node.
   static_assert(std::is_same_v<decltype(f*u),MultFieldExpr<FiniteElementField<"f">,TrialSpace<"u">>>);
   const auto form=integrate(Cells<"m">{},-(2.5*(f*u))*TestSpace<"v">{});
   // Integrate the prescribed linear polynomial against each Q1 basis exactly.
   const Dense matrix{-2.5*length*(2*1.25-.75)/6,-2.5*length*(1.25-2*.75)/6};
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<3>{});
   bool ok=CheckForm(form,context,rule,matrix,1,2,"left prescribed multiplier");
   const auto right=integrate(Cells<"m">{},-(2.5*(u*f))*TestSpace<"v">{});
   return CheckForm(right,context,rule,matrix,1,2,"right prescribed multiplier") && ok;
}

template<FormExpr Form,DomainExpr Domain>
auto ZeroWithContract(const Form&,const Domain& domain)
{
   using Signature=form_signature_t<Form>;
   using Original=typename form_symbols_t<Form>::plain_declarations;
   using Declarations=weak_form_traits::symbol_summary<Original::supported,typename Original::active,
      typename Original::tests,typename Original::bound,typename Signature::callable_input_relations>;
   using Zero=ZeroExpr<Original,typename Signature::logical_output_sides>;
   auto term=integrate(domain,Zero{});
   return SumFormExpr<std::tuple<decltype(term)>,Declarations>{std::tuple{term}};
}

bool LiveAndZeroBlocks()
{
   CartesianMesh<1> mesh({2},{1.},{0.},false);
   const auto p0=MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto us=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{1,5});
   const auto vs=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{2,6});
   const auto qs=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{12,15});
   const auto rs=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{8,11});
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(us),MakeTestField<"v">(vs),
      MakeTrialField<"q">(qs),MakeTestField<"r">(rs),MakeIntegrationDomain<"m">(mesh));
   const auto discarded=MakeCoefficient<"discarded",FieldValue<"missing">>([](Real){return Real(99);});
   const auto zero=ZeroWithContract(integrate(Cells<"m">{},discarded*TrialSpace<"q">{}*TestSpace<"r">{}),Cells<"m">{});
   const auto form=integrate(Cells<"m">{},TrialSpace<"u">{}*TestSpace<"v">{})+zero;
   Dense matrix(11*15,0);matrix[2*15+1]=1;matrix[3*15+2]=1;
   return CheckForm(form,context,MakeIntegrationRule(IntegrationRuleNumPoints<2>{}),matrix,11,15,
      "live plus typed zero block retains dimensions without discarded reads");
}

bool CoupledOffsetsAndSameName()
{
   CartesianMesh<1> mesh({2},{1.},{0.},false);
   const auto p0=MakeLegendreFiniteElement(FiniteElementOrders<0>{});
   const auto us=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{1,9});
   const auto ps=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{5,9});
   const auto vs=MakeFiniteElementSpace(mesh,p0,ContiguousL2RestrictionSpecification{2,6});
   const auto context=MakeWeakFormContext(MakeTrialField<"u">(us),MakeTrialField<"p">(ps),
      MakeTestField<"v">(vs),MakeIntegrationDomain<"m">(mesh));
   const auto rule=MakeIntegrationRule(IntegrationRuleNumPoints<2>{});
   const TrialSpace<"u"> u; const TrialSpace<"p"> p; const TestSpace<"v"> v;
   Dense matrix(6*9,0);
   matrix[2*9+1]=1;matrix[2*9+5]=1;matrix[3*9+2]=1;matrix[3*9+6]=1;
   bool ok=CheckForm(integrate(Cells<"m">{},(u+p)*v),context,rule,matrix,6,9,"coupled offsets");
   const auto same=MakeWeakFormContext(MakeTrialField<"w">(us),MakeIntegrationDomain<"m">(mesh));
   Dense identity(9*9,0);identity[1*9+1]=1;identity[2*9+2]=1;
   ok=CheckForm(integrate(Cells<"m">{},TrialSpace<"w">{}*TestSpace<"w">{}),same,rule,
      identity,9,9,"same-name trial/test occurrences") && ok;
   return ok;
}
} // namespace form_transpose_operator_test

int main()
{
   using namespace form_transpose_operator_test;
   bool ok=GradientAndCapturedData();
   ok=RectangularComponentsAndOuter() && ok;
   ok=DivergenceAndTensorNodes() && ok;
   ok=CurlNode() && ok;
   ok=LivePrescribedFieldAndFallbackProduct() && ok;
   ok=CoupledOffsetsAndSameName() && ok;
   ok=LiveAndZeroBlocks() && ok;
   if(ok) std::cout << "Bilinear form transpose matches independent cell contractions.\n";
   return ok?0:1;
}
