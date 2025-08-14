#pragma once

#include <cassert>
#include <array>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <complex>

#include <unsupported/Eigen/MatrixFunctions>

using Complex = std::complex<double>;
using Idx = std::size_t;

using G = std::complex<double>; // U(1); abs(u)=1, u \in G
using GaugeField = std::vector<std::array<G, 5>>; // U[x][mu], U[x][0] = 0.0


struct DW2{
  static constexpr int NS=2;
  static constexpr int DIM=3;
  static constexpr int DIMEXT=4;

  using MS=Eigen::Matrix2cd; // weyl
  using VD=Eigen::Vector2d; // weyl
  // using MD=Eigen::Matrix4cd; // dirac
  // using VD=Eigen::Vector4d; // dirac
  // using VE=Eigen::Vector5d; // embed
  // using VC=Eigen::VectorXcd;

  static constexpr Complex I = Complex(0.0, 1.0);

  const std::vector<int> L;
  const Idx vol;
  const int Ls;
  const double M5;

  std::array<MS, DIM+1> sigma;
  std::vector<int> bdy;
  // std::array<MD, 6> gamma;

  DomainWall()=delete;

  DomainWall(const std::vector<int> L_,
             const std::vector<int> bdy_=std::vector<int>{3,1,1,-1},
             const double M5_ = 1.0)
    : L(L_)
    , vol(L[1]*L[2]*L[3])
    , Ls(L[DIMEXT])
    , bdy(bdy_)
    , M5(M5_)
  {
    assert(L[0]==DIMEXT);
    assert(bdy_[0]==DIM);
    set_sigma();
  }

  DomainWall & operator=(const DomainWall&) = delete;

  Idx idxDIM(const std::vector<int>& x ) const { // 3d
    assert(x[0]==DIM);

    Idx i = (x[1]+L[1])%L[1];
    for(int mu=1; mu<DIM; mu++){
      const int x_mup1 = (x[mu+1]+L[mu+1])%L[mu+1];
      i = x_mup1 + L[mu+1]*i;
    }
    return i;
  }

  Idx idxEXT(const std::vector<int>& x ) const { // 4d
    assert(x[0]==DIMEXT);

    Idx iDIM = idxDIM( std::vector<int>{DIM, x[1],x[2],x[3]} );

    // std::cout << "debug. x5 = " << x[5] << std::endl;
    // std::cout << "debug. mo = " << (x[5]+L[5])%L[5] << std::endl;
    return iDIM + vol* ((x[DIMEXT]+Ls)%Ls);
  }

  void set_sigma(){
    sigma[0] << 1,0,0,1;
    sigma[1] << 0,1,1,0;
    sigma[2] << 0,-I,I,0;
    sigma[3] << 1,0,0,-1;
    // sigma[4] << I,0,0,I;
  }

  Idx size() const { return NS*Ls*vol; }


  Eigen::MatrixXcd get_Dw( const double M ) const {
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(NS*vol, NS*vol);

    std::vector<int> x(DIM+1);
    x[0] = DIM;
    for(x[1]=0; x[1]<L[1]; x[1]++){
      for(x[2]=0; x[2]<L[2]; x[2]++){
        for(x[3]=0; x[3]<L[3]; x[3]++){
          const Idx i=idxDIM(x);

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(int mu=1; mu<=DIM; mu++){
            std::vector<int> xpmu = x;
            std::vector<int> xmmu = x;
            xpmu[mu] += 1;
            xmmu[mu] -= 1;
            const Idx ipmu=idxDIM(xpmu);
            const Idx immu=idxDIM(xmmu);

            // wilson
            res.block(NS*i, NS*ipmu, NS,NS) -= 0.5*(sigma[0]-sigma[mu]);
            res.block(NS*i, NS*immu, NS,NS) -= 0.5*(sigma[0]+sigma[mu]);
            res.block(NS*i, NS*i,    NS,NS) += 1.0*sigma[0];

            // bc
            if(x[mu]==L[mu]-1) res.block(NS*i, NS*ipmu, NS,NS) *= bdy[mu];
            else if(x[mu]==0) res.block(NS*i, NS*immu, NS,NS) *= bdy[mu];
          }
          // mass
          res.block(NS*i, NS*i, NS,NS) += M * sigma[0];
        }}}

    return res;
  }



  Eigen::MatrixXcd get_G3() const {
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(NS*vol, NS*vol);

// #ifdef _OPENMP
// #pragma omp parallel for collapse(4)
// #endif
    std::vector<int> x(DIM+1);
    x[0] = DIM;
    for(x[1]=0; x[1]<L[1]; x[1]++){
      for(x[2]=0; x[2]<L[2]; x[2]++){
        for(x[3]=0; x[3]<L[3]; x[3]++){
          std::vector<int> xT{ DIM, x[1], x[2], L[3]-x[3]-1 };
          const Idx iT=idxDIM(xT);

          res.block(NS*iT, NS*iT, NS,NS) = sigma[3];
        }}}

    return res;
  }


  Eigen::MatrixXcd get_Hw( const double M ) const {
    return get_G3() * get_Dw(M);
  }


//   Eigen::MatrixXcd matrix_form( const double m ) const {
//     const Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(4*vol, 4*vol);
//     const Eigen::MatrixXcd g5 = get_g5();

//     const Eigen::MatrixXcd Pp = 0.5*(id + g5);
//     const Eigen::MatrixXcd Pm = 0.5*(id - g5);

//     // Eigen::MatrixXcd Hw = get_Hw(-M5);
//     Eigen::MatrixXcd Hw = get_Hw(-M5);
//     Hw *= -1.0;
//     std::cout << "exp" << std::endl;
//     const Eigen::MatrixXcd Tinv = Hw.exp();
//     std::cout << "exp done" << std::endl;

//     // main matrix
//     Eigen::MatrixXcd Dchi = Eigen::MatrixXcd::Zero(4*Ls*vol, 4*Ls*vol);
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//     for(int s=0; s<L[5]; s++){
//       const Idx icurr = idx5(std::vector<int>{5, 0,0,0,0, s});
//       const Idx inext = idx5(std::vector<int>{5, 0,0,0,0, s+1});

//       // block diag
//       if(s==0) Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Pm-m*Pp;
//       else Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Eigen::MatrixXcd::Identity(4*vol, 4*vol);

//       // s block hop
//       if(s!=Ls-1) Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv;
//       else Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv * (Pp-m*Pm);
//     }

//     return Dchi;
//   } // end matrix_form


//   Eigen::MatrixXcd matrix_form( const double m, const GaugeField& U ) const {
//     const Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(4*vol, 4*vol);
//     const Eigen::MatrixXcd g5 = get_g5();

//     const Eigen::MatrixXcd Pp = 0.5*(id + g5);
//     const Eigen::MatrixXcd Pm = 0.5*(id - g5);

//     // Eigen::MatrixXcd Hw = get_Hw(-M5);
//     Eigen::MatrixXcd Hw = get_Hw(-M5, U);
//     Hw *= -1.0;
//     const Eigen::MatrixXcd Tinv = Hw.exp();

//     // main matrix
//     Eigen::MatrixXcd Dchi = Eigen::MatrixXcd::Zero(4*Ls*vol, 4*Ls*vol);
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//     for(int s=0; s<L[5]; s++){
//       const Idx icurr = idx5(std::vector<int>{5, 0,0,0,0, s});
//       const Idx inext = idx5(std::vector<int>{5, 0,0,0,0, s+1});

//       // block diag
//       if(s==0) Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Pm-m*Pp;
//       else Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Eigen::MatrixXcd::Identity(4*vol, 4*vol);

//       // s block hop
//       if(s!=Ls-1) Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv;
//       else Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv * (Pp-m*Pm);
//     }

//     return Dchi;
//   } // end matrix_form

};


