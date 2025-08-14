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


struct DomainWall{
  using MS=Eigen::Matrix2cd; // weyl
  using MD=Eigen::Matrix4cd; // dirac
  using VD=Eigen::Vector4d; // dirac
  // using VE=Eigen::Vector5d; // embed
  // using VC=Eigen::VectorXcd;

  static constexpr Complex I = Complex(0.0, 1.0);

  const std::vector<int> L;
  const Idx vol;
  const int Ls;
  const double M5;

  std::array<MS, 5> sigma;
  std::array<MD, 6> gamma;

  DomainWall()=delete;

  DomainWall(const std::vector<int> L_,
             const double M5_ = 1.0)
    : L(L_)
    , vol(L[1]*L[2]*L[3]*L[4])
    , Ls(L[5])
    , M5(M5_)
  {
    assert(L[0]==5);
    set_sigma();
    set_gamma();
  }

  DomainWall & operator=(const DomainWall&) = delete;

  Idx idx4(const std::vector<int>& x ) const { // 4d
    assert(x[0]==4);

    Idx i = (x[1]+L[1])%L[1];
    for(int mu=1; mu<=3; mu++){
      const int x_mup1 = (x[mu+1]+L[mu+1])%L[mu+1];
      i = x_mup1 + L[mu+1]*i;
    }
    return i;
  }

  Idx idx5(const std::vector<int>& x ) const { // 5d
    assert(x[0]==5);

    Idx i4 = idx4( std::vector<int>{4, x[1],x[2],x[3],x[4]} );

    // std::cout << "debug. x5 = " << x[5] << std::endl;
    // std::cout << "debug. mo = " << (x[5]+L[5])%L[5] << std::endl;
    return i4 + vol* ((x[5]+L[5])%L[5]);
  }

  void set_sigma(){
    sigma[0] << 1,0,0,1;
    sigma[1] << 0,1,1,0;
    sigma[2] << 0,-I,I,0;
    sigma[3] << 1,0,0,-1;
    sigma[4] << I,0,0,I;
  }

  void set_gamma(){
    // id
    gamma[0] = MD::Identity();

    // chiral
    for(int mu=1; mu<=3; mu++){
      gamma[mu] = MD::Zero();
      gamma[mu].block(0,2,2,2) = sigma[mu];
      gamma[mu].block(2,0,2,2) = sigma[mu];
    }

    // g4
    gamma[4] = MD::Zero();
    gamma[4].block(0,2,2,2) = sigma[4];
    gamma[4].block(2,0,2,2) = -sigma[4];

    // g5
    gamma[5] = MD::Zero();
    gamma[5].block(0,0,2,2) = sigma[0];
    gamma[5].block(2,2,2,2) = -sigma[0];
  }

  Idx size() const { return 4*Ls*vol; }

  Eigen::MatrixXcd get_Dw( const double M ) const {
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(4*vol, 4*vol);
// #ifdef _OPENMP
// #pragma omp parallel for collapse(4)
// #endif
    for(int ix=0; ix<L[1]; ix++){
      for(int iy=0; iy<L[2]; iy++){
        for(int iz=0; iz<L[3]; iz++){
          for(int it=0; it<L[4]; it++){

            const std::vector<int> x{4, ix, iy, iz, it};
            const Idx i=idx4(x);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int mu=1; mu<=4; mu++){
              std::vector<int> xpmu = x;
              std::vector<int> xmmu = x;
              xpmu[mu] += 1;
              xmmu[mu] -= 1;
              const Idx ipmu=idx4(xpmu);
              const Idx immu=idx4(xmmu);

              // wilson
              res.block(4*i, 4*ipmu, 4,4) -= 0.5*(gamma[0]-gamma[mu]);
              res.block(4*i, 4*immu, 4,4) -= 0.5*(gamma[0]+gamma[mu]);
              res.block(4*i, 4*i,    4,4) += 1.0*gamma[0];

              // bc
              if(it==L[4]-1 && mu==4) res.block(4*i, 4*ipmu, 4,4) *= -1.0;
              else if(it==0 && mu==4) res.block(4*i, 4*immu, 4,4) *= -1.0;
            }
            // mass
            res.block(4*i, 4*i, 4,4) += M * gamma[0];
          }}}}

    return res;
  }

  Eigen::MatrixXcd get_Dw( const double M, const GaugeField& U ) const {
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(4*vol, 4*vol);
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for(int ix=0; ix<L[1]; ix++){
      for(int iy=0; iy<L[2]; iy++){
        for(int iz=0; iz<L[3]; iz++){
          for(int it=0; it<L[4]; it++){

            const std::vector<int> x{4, ix, iy, iz, it};
            const Idx i=idx4(x);

// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
            for(int mu=1; mu<=4; mu++){
              std::vector<int> xpmu = x;
              std::vector<int> xmmu = x;
              xpmu[mu] += 1;
              xmmu[mu] -= 1;
              const Idx ipmu=idx4(xpmu);
              const Idx immu=idx4(xmmu);

              // wilson
              res.block(4*i, 4*ipmu, 4,4) -= 0.5*(gamma[0]-gamma[mu])*U[i][mu];
              res.block(4*i, 4*immu, 4,4) -= 0.5*(gamma[0]+gamma[mu])*std::conj(U[immu][mu]);
              res.block(4*i, 4*i,    4,4) += 1.0*gamma[0];

              // bc
              if(it==L[4]-1 && mu==4) res.block(4*i, 4*ipmu, 4,4) *= -1.0;
              else if(it==0 && mu==4) res.block(4*i, 4*immu, 4,4) *= -1.0;
            }
            // mass
            res.block(4*i, 4*i, 4,4) += M * gamma[0];
          }}}}

    return res;
  }


  Eigen::MatrixXcd get_X( const double M, const GaugeField& U ) const {
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(2*vol, 2*vol);
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for(int ix=0; ix<L[1]; ix++){
      for(int iy=0; iy<L[2]; iy++){
        for(int iz=0; iz<L[3]; iz++){
          for(int it=0; it<L[4]; it++){

            const std::vector<int> x{4, ix, iy, iz, it};
            const Idx i=idx4(x);

// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
            for(int mu=1; mu<=4; mu++){
              std::vector<int> xpmu = x;
              std::vector<int> xmmu = x;
              xpmu[mu] += 1;
              xmmu[mu] -= 1;
              const Idx ipmu=idx4(xpmu);
              const Idx immu=idx4(xmmu);

              // wilson
              res.block(2*i, 2*ipmu, 2,2) -= 0.5*(sigma[0]-sigma[mu])*U[i][mu];
              res.block(2*i, 2*immu, 2,2) -= 0.5*(sigma[0]+sigma[mu])*std::conj(U[immu][mu]);
              res.block(2*i, 2*i,    2,2) += 1.0*sigma[0];

              // bc
              if(it==L[4]-1 && mu==4) res.block(2*i, 2*ipmu, 2,2) *= -1.0;
              else if(it==0 && mu==4) res.block(2*i, 2*immu, 2,2) *= -1.0;
            }
            // mass
            res.block(2*i, 2*i, 2,2) += M * sigma[0];
          }}}}

    return res;
  }

  Eigen::MatrixXcd get_g5() const {
    Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(4*vol, 4*vol);

#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for(int ix=0; ix<L[1]; ix++){
      for(int iy=0; iy<L[2]; iy++){
        for(int iz=0; iz<L[3]; iz++){
          for(int it=0; it<L[4]; it++){
            const std::vector<int> x{4, ix, iy, iz, it};
            const Idx i=idx4(x);
            res.block(4*i, 4*i, 4,4) = gamma[5];
          }}}}

    return res;
  }


  Eigen::MatrixXcd get_Hw( const double M ) const {
    return get_g5() * get_Dw(M);
  }

  Eigen::MatrixXcd get_Hw( const double M, const GaugeField& U ) const {
    return get_g5() * get_Dw(M, U);
  }

  // Eigen::MatrixXcd get_Hw( const double M ) const {
  //   Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(4*vol, 4*vol);

  //   for(int ix=0; ix<L[1]; ix++){
  //     for(int iy=0; iy<L[2]; iy++){
  //       for(int iz=0; iz<L[3]; iz++){
  //         for(int it=0; it<L[4]; it++){

  //           const std::vector<int> x{4, ix, iy, iz, it};
  //           const Idx i=idx4(x);

  //           for(int mu=1; mu<=4; mu++){
  //             std::vector<int> xpmu = x;
  //             std::vector<int> xmmu = x;
  //             xpmu[mu] += 1;
  //             xmmu[mu] -= 1;

  //             Idx ipmu=idx4(xpmu);
  //             Idx immu=idx4(xmmu);

  //             // +C
  //             res.block(4*i, 4*ipmu+2, 2,2) += 0.5*sigma[mu];
  //             res.block(4*i, 4*immu+2, 2,2) -= 0.5*sigma[mu];
  //             // -C
  //             res.block(4*i+2, 4*ipmu, 2,2) -= 0.5*sigma[mu];
  //             res.block(4*i+2, 4*immu, 2,2) += 0.5*sigma[mu];

  //             // +B
  //             res.block(4*i, 4*ipmu, 2,2) -= 0.5*sigma[0];
  //             res.block(4*i, 4*immu, 2,2) -= 0.5*sigma[0];
  //             res.block(4*i, 4*i, 2,2) += 1.0*sigma[0];
  //             // -B
  //             res.block(4*i+2, 4*ipmu+2, 2,2) += 0.5*sigma[0];
  //             res.block(4*i+2, 4*immu+2, 2,2) += 0.5*sigma[0];
  //             res.block(4*i+2, 4*i+2, 2,2) -= 1.0*sigma[0];
  //           }
  //           // mass
  //           res.block(4*i, 4*i, 4,4) += M * gamma[5];
  //         }}}}

  //   return res;
  // }


  // void projectors( Eigen::MatrixXcd& Pp, Eigen::MatrixXcd& Pm ) const {
  //   for(int ix=0; ix<L[1]; ix++){
  //     for(int iy=0; iy<L[2]; iy++){
  //       for(int iz=0; iz<L[3]; iz++){
  //         for(int it=0; it<L[4]; it++){

  //           const std::vector<int> x{4, ix, iy, iz, it};
  //           const Idx i=idx4(x);

  //           Pm.block(4*i, 4*i, 4,4) = 0.5*(gamma[0]-gamma[5]);
  //           Pp.block(4*i, 4*i, 4,4) = 0.5*(gamma[0]+gamma[5]);
  //         }}}}
  // }


  Eigen::MatrixXcd matrix_form( const double m ) const {
    const Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(4*vol, 4*vol);
    const Eigen::MatrixXcd g5 = get_g5();

    const Eigen::MatrixXcd Pp = 0.5*(id + g5);
    const Eigen::MatrixXcd Pm = 0.5*(id - g5);

    // Eigen::MatrixXcd Hw = get_Hw(-M5);
    Eigen::MatrixXcd Hw = get_Hw(-M5);
    Hw *= -1.0;
    std::cout << "exp" << std::endl;
    const Eigen::MatrixXcd Tinv = Hw.exp();
    std::cout << "exp done" << std::endl;

    // main matrix
    Eigen::MatrixXcd Dchi = Eigen::MatrixXcd::Zero(4*Ls*vol, 4*Ls*vol);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int s=0; s<L[5]; s++){
      const Idx icurr = idx5(std::vector<int>{5, 0,0,0,0, s});
      const Idx inext = idx5(std::vector<int>{5, 0,0,0,0, s+1});

      // block diag
      if(s==0) Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Pm-m*Pp;
      else Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Eigen::MatrixXcd::Identity(4*vol, 4*vol);

      // s block hop
      if(s!=Ls-1) Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv;
      else Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv * (Pp-m*Pm);
    }

    return Dchi;
  } // end matrix_form


  Eigen::MatrixXcd matrix_form( const double m, const GaugeField& U ) const {
    const Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(4*vol, 4*vol);
    const Eigen::MatrixXcd g5 = get_g5();

    const Eigen::MatrixXcd Pp = 0.5*(id + g5);
    const Eigen::MatrixXcd Pm = 0.5*(id - g5);

    // Eigen::MatrixXcd Hw = get_Hw(-M5);
    Eigen::MatrixXcd Hw = get_Hw(-M5, U);
    Hw *= -1.0;
    const Eigen::MatrixXcd Tinv = Hw.exp();

    // main matrix
    Eigen::MatrixXcd Dchi = Eigen::MatrixXcd::Zero(4*Ls*vol, 4*Ls*vol);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int s=0; s<L[5]; s++){
      const Idx icurr = idx5(std::vector<int>{5, 0,0,0,0, s});
      const Idx inext = idx5(std::vector<int>{5, 0,0,0,0, s+1});

      // block diag
      if(s==0) Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Pm-m*Pp;
      else Dchi.block(4*icurr, 4*icurr, 4*vol,4*vol) = Eigen::MatrixXcd::Identity(4*vol, 4*vol);

      // s block hop
      if(s!=Ls-1) Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv;
      else Dchi.block(4*icurr, 4*inext, 4*vol,4*vol) = -Tinv * (Pp-m*Pm);
    }

    return Dchi;
  } // end matrix_form

};


