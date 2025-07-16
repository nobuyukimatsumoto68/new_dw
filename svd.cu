#include <iostream>
#include <iomanip>
#include <fstream>
// #include <cstdlib>
#include <cassert>
#include <random>
// #include <algorithm>
// #include <cstdint>
// #include <complex>
// #include <array>
// #include <vector>
// #include <map>

#include <Eigen/Eigenvalues>



#include "gpu_header.h"
#include "dirac.h"



// gauge
// using G = std::complex<double>; // U(1); abs(u)=1, u \in G
// using GaugeField = std::vector<std::array<G, 5>>; // U[x][mu], U[x][0] = 0.0


int main(int argc, char* argv[]){
  std::cout << std::scientific << std::setprecision(15);
  std::clog << std::scientific << std::setprecision(15);

  const Complex I = Complex(0.0, 1.0);

#ifdef _OPENMP
  omp_set_num_threads(12);
#endif

  // int device;
  // CUDA_CHECK(cudaGetDeviceCount(&device));
  // cudaDeviceProp device_prop[device];
  // cudaGetDeviceProperties(&device_prop[0], 0);
  // std::cout << "# dev = " << device_prop[0].name << std::endl;
  // CUDA_CHECK(cudaSetDevice(0));// "TITAN V"
  // std::cout << "# (GPU device is set.)" << std::endl;

  // ---------------------------------------

  int Ls = 4; // 8
  int seed = 1;
  double width = 0.0;

  if(argc==4){
    Ls = atoi(argv[1]);
    seed = atoi(argv[2]);
    width = atof(argv[3]);
  }
  std::cout << "Ls = " << Ls << std::endl;
  std::cout << "seed = " << seed << std::endl;
  std::cout << "width = " << width << std::endl;

  // std::vector<int> L5{ 5, 3,3,3,16, 2};
  // std::vector<int> L5{ 5, 4,4,4,4, 1};
  // std::vector<int> L5{ 5, 6,6,6,6, 1};
  std::vector<int> L5{ 5, 4,4,4,4, Ls};
  // std::vector<int> L5{ 5, 2,2,2,2, 4};
  // std::vector<int> L5{ 5, 2,2,2,2, 2};
  // for(auto elem : L5) std::cout << elem << std::endl;

  DomainWall Ddw(L5);

  std::mt19937 gen(seed);
  std::normal_distribution d{0.0, width};

  GaugeField u(Ddw.vol);
  for(Idx i=0; i<Ddw.vol; i++){
    for(int mu=1; mu<=4; mu++){
      // u[i][mu] = 1.0;
      u[i][mu] = std::exp( I*d(gen) );
    }
  }

  // Eigen::MatrixXcd HW = Ddw.get_Hw( -Ddw.M5, u );
  Eigen::MatrixXcd mat = Ddw.get_X( -Ddw.M5, u );

  // {
  //   std::ofstream file("real.dat", std::ios::trunc);
  //   file << "# real:" << std::endl;
  //   file << mat.real() << std::endl;
  // }
  // {
  //   std::ofstream file("imag.dat", std::ios::trunc);
  //   file << "# imag:" << std::endl;
  //   file << mat.imag() << std::endl;
  // }


  // =========================================

  const int n = mat.cols(); // Number of rows (or columns) of matrix A.

  // std::vector<std::complex<double>> U(n*n);
  // std::vector<std::complex<double>> VH(n*n);

  // {
  //   // cusolver
  //   cusolverDnHandle_t handle = NULL;
  //   cudaStream_t stream = NULL;
  //   cusolverDnParams_t params = NULL;

  //   const int lda = n;

  //   CuC *A;
  //   double *S;
  //   A = (CuC*)malloc(n*n*CD);
  //   S = (double*)malloc(n*DB);
  //   for(int j=0; j<n; j++) for(int i=0; i<n; i++) A[n*j+i] = cplx(mat(i,j));
  //   for(int i=0; i<n; i++) S[i] = 0.;

  //   CuC *d_A, *d_U, *d_VT;
  //   double *d_S;

  //   signed char jobu = 'A';
  //   signed char jobvt = 'A';
  //   int ldu = n;
  //   int ldvt = n;
  //   //
  //   int info = 0;
  //   int *d_info = nullptr;

  //   size_t workspaceInBytesOnDevice = 0; /* size of workspace */
  //   void *d_work = nullptr;              /* device workspace */
  //   size_t workspaceInBytesOnHost = 0;   /* size of workspace */
  //   void *h_work = nullptr;              /* host workspace for */

  //   /* step 1: create cusolver handle, bind a stream */
  //   CUSOLVER_CHECK(cusolverDnCreate(&handle));
  //   CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  //   CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  //   CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  //   CUDA_CHECK(cudaMalloc( &d_A, CD * n*n ));
  //   CUDA_CHECK(cudaMalloc( &d_S, DB * n ));
  //   CUDA_CHECK(cudaMalloc( &d_U, CD * n*n ));
  //   CUDA_CHECK(cudaMalloc( &d_VT, CD * n*n ));
  //   CUDA_CHECK(cudaMalloc( &d_info, sizeof(int)));

  //   CUDA_CHECK( cudaMemcpy(d_A, A, CD*n*n, H2D) );
  //   CUDA_CHECK( cudaMemset(d_S, 0, DB * n) );
  //   CUDA_CHECK( cudaMemset(d_U, 0, CD * n*n) );
  //   CUDA_CHECK( cudaMemset(d_VT, 0, CD * n*n) );

  //   CUSOLVER_CHECK( cusolverDnXgesvd_bufferSize( handle,
  //                                                params,
  //                                                jobu,
  //                                                jobvt,
  //                                                n,
  //                                                n,
  //                                                CUDA_C_64F,
  //                                                d_A, // device
  //                                                lda,
  //                                                CUDA_R_64F,
  //                                                d_S, // Array holding the computed eigenvalues of A
  //                                                CUDA_C_64F,
  //                                                d_U,
  //                                                ldu,
  //                                                CUDA_C_64F,
  //                                                d_VT,
  //                                                ldvt,
  //                                                CUDA_C_64F,
  //                                                &workspaceInBytesOnDevice,
  //                                                &workspaceInBytesOnHost)
  //                   );

  //   CUDA_CHECK(cudaMalloc( &d_work, workspaceInBytesOnDevice ) );
  //   h_work = malloc(workspaceInBytesOnHost);

  //   // step 4: compute spectrum
  //   CUSOLVER_CHECK( cusolverDnXgesvd( handle,
  //                                     params,
  //                                     jobu,
  //                                     jobvt,
  //                                     n,
  //                                     n,
  //                                     CUDA_C_64F,
  //                                     d_A,
  //                                     lda,
  //                                     CUDA_R_64F,
  //                                     d_S,
  //                                     CUDA_C_64F,
  //                                     d_U,
  //                                     ldu,
  //                                     CUDA_C_64F,
  //                                     d_VT,
  //                                     ldvt,
  //                                     CUDA_C_64F,
  //                                     d_work, // void *bufferOnDevice,
  //                                     workspaceInBytesOnDevice,
  //                                     h_work, // void *bufferOnHost,
  //                                     workspaceInBytesOnHost,
  //                                     d_info)
  //                   );

  //   // ---------------------------------------------

  //   CUDA_CHECK(cudaMemcpy( S, d_S, DB*n, D2H) );
  //   CUDA_CHECK(cudaMemcpy( &info, d_info, sizeof(int), D2H ));

  //   CUDA_CHECK(cudaMemcpy( reinterpret_cast<CuC*>(U.data()), d_U, CD * n*n, D2H ));
  //   CUDA_CHECK(cudaMemcpy( reinterpret_cast<CuC*>(VH.data()), d_VT, CD * n*n, D2H ));

  //   for(Idx i=0; i<n; i++) std::cout << S[i] << std::endl;

  //   std::cout << "# info (0=success) = " << info << std::endl;
  //   assert( info==0 );

  //   /* free resources */
  //   free(A);
  //   free(S);
  //   free(h_work);

  //   CUDA_CHECK(cudaFree(d_A));
  //   CUDA_CHECK(cudaFree(d_S));
  //   CUDA_CHECK(cudaFree(d_U));
  //   CUDA_CHECK(cudaFree(d_VT));
  //   CUDA_CHECK(cudaFree(d_info));
  //   CUDA_CHECK(cudaFree(d_work));

  //   CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  //   CUSOLVER_CHECK(cusolverDnDestroy(handle));
  //   CUDA_CHECK(cudaStreamDestroy(stream));
  // }

  // eigen, cusolver: both column major
  // Eigen::MatrixXcd eigenU = Eigen::Map<Eigen::MatrixXcd>(U.data(), n, n);
  // Eigen::MatrixXcd eigenVH = Eigen::Map<Eigen::MatrixXcd>(VH.data(), n, n);

  Eigen::BDCSVD<Eigen::MatrixXcd, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(n,n);
  svd.compute(mat);
  Eigen::MatrixXcd eigenU = svd.matrixU();
  Eigen::MatrixXcd eigenVH = svd.matrixV().adjoint();
  Eigen::MatrixXcd Dov = Eigen::MatrixXcd::Identity(n,n) + eigenU*eigenVH;

  std::cout << "eigenU check: " << ( eigenU * eigenU.adjoint() - Eigen::MatrixXcd::Identity(n,n) ).norm() << std::endl;
  std::cout << "eigenVH check: " << ( eigenVH * eigenVH.adjoint() - Eigen::MatrixXcd::Identity(n,n) ).norm() << std::endl;

  {
    // cusolver
    cusolverDnHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    // const int n = mat.cols(); // Number of rows (or columns) of matrix A.
    const int lda = n;

    CuC *A, *W;
    A = (CuC*)malloc(n*n*CD);
    W = (CuC*)malloc(n*CD);
    for(int j=0; j<n; j++) for(int i=0; i<n; i++) A[n*j+i] = cplx(Dov(i,j));
    for(int i=0; i<n; i++) W[i] = cplx(0.);

    CuC *d_A, *d_W, *d_VL, *d_VR;

    cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_NOVECTOR;
    cusolverEigMode_t jobvr = CUSOLVER_EIG_MODE_NOVECTOR;
    int ldvl = n;
    int ldvr = n;
    //
    int info = 0;
    int *d_info = nullptr;

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for */

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    CUDA_CHECK(cudaMalloc( &d_A, CD * n*n ));
    CUDA_CHECK(cudaMalloc( &d_W, CD * n ));
    CUDA_CHECK(cudaMalloc( &d_VL, CD * n*n ));
    CUDA_CHECK(cudaMalloc( &d_VR, CD * n*n ));
    CUDA_CHECK(cudaMalloc( &d_info, sizeof(int)));

    CUDA_CHECK( cudaMemcpy(d_A, A, CD*n*n, H2D) );
    CUDA_CHECK( cudaMemset(d_W, 0, CD * n) );
    CUDA_CHECK( cudaMemset(d_VL, 0, CD * n*n) );
    CUDA_CHECK( cudaMemset(d_VR, 0, CD * n*n) );


    CUSOLVER_CHECK( cusolverDnXgeev_bufferSize( handle,
                                                params,
                                                jobvl,
                                                jobvr,
                                                n,
                                                CUDA_C_64F,
                                                d_A, // device
                                                lda,
                                                CUDA_C_64F,
                                                d_W, // Array holding the computed eigenvalues of A
                                                CUDA_C_64F,
                                                d_VL,
                                                ldvl,
                                                CUDA_C_64F,
                                                d_VR,
                                                ldvr,
                                                CUDA_C_64F,
                                                &workspaceInBytesOnDevice,
                                                &workspaceInBytesOnHost)
                    );

    CUDA_CHECK(cudaMalloc( &d_work, workspaceInBytesOnDevice ) );
    h_work = malloc(workspaceInBytesOnHost);

    // step 4: compute spectrum
    CUSOLVER_CHECK( cusolverDnXgeev( handle,
                                     params,
                                     jobvl,
                                     jobvr,
                                     n,
                                     CUDA_C_64F,
                                     d_A,
                                     lda,
                                     CUDA_C_64F,
                                     d_W,
                                     CUDA_C_64F,
                                     d_VL,
                                     ldvl,
                                     CUDA_C_64F,
                                     d_VR,
                                     ldvr,
                                     CUDA_C_64F,
                                     d_work, // void *bufferOnDevice,
                                     workspaceInBytesOnDevice,
                                     h_work, // void *bufferOnHost,
                                     workspaceInBytesOnHost,
                                     d_info)
                    );

    // ---------------------------------------------

    CUDA_CHECK(cudaMemcpy( W, d_W, CD*n, D2H) );
    CUDA_CHECK(cudaMemcpy( &info, d_info, sizeof(int), D2H ));

    // std::vector<std::complex<double>> vr(n*n);
    // for(Idx i=0; i<N; i++) gmfourth(d_VL+i*N, d_VR+i*N);
    // CUDA_CHECK(cudaMemcpy( reinterpret_cast<CuC*>(vr.data()), d_VL, CD * n*n, D2H ));

    std::cout << "# info (0=success) = " << info << std::endl;
    assert( info==0 );

    std::vector<double> re(n), im(n);
    for(int i=0; i<n; i++) {
      re[i] = real(W[i]);
      im[i] = imag(W[i]);
    }
    // std::sort(res.begin(), res.end());
    for(int i=0; i<n; i++) std::cout << i << " "
                                     << re[i] << " "
                                     << im[i] << " "
                                     << std::endl;


    // for(int i=0; i<n; i++) std::clog << real(vr[i]) << " " << imag(vr[i]) << std::endl;
    // }
    {
      std::ofstream file("ev_Dov_seed"+std::to_string(seed)+"_width"+std::to_string(width)+".dat", std::ios::trunc);
      file << std::scientific << std::setprecision(15);
      file << "# ev" << std::endl;
      for(int i=0; i<n; i++) file << i << " " << re[i] << " " << im[i] << std::endl;

      std::complex<double> log_total;
      // #ifdef _OPENMP
      // #pragma omp parallel for reduction(+:log_total)
      // #endif
      for(int i=0; i<n; i++) {
        log_total += std::log(re[i]+I*im[i]);
      }
      std::cout << "log_total = " << log_total << std::endl;
    }

    /* free resources */
    free(A);
    free(W);
    free(h_work);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_VL));
    CUDA_CHECK(cudaFree(d_VR));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }




  // std::cout << "get Hw" << std::endl;
  // const Eigen::MatrixXcd Hw = Ddw.get_Hw( -Ddw.M5 );
  // std::cout << "calc SVD" << std::endl;
  // Eigen::BDCSVD<Eigen::MatrixXcd, Eigen::ComputeThinU | Eigen::ComputeThinV> solver(mat);

  // std::cout << "calc V" << std::endl;
  // const Eigen::MatrixXcd V = solver.matrixU() * solver.matrixV().adjoint();
  // const Eigen::MatrixXcd Dov = Eigen::MatrixXcd::Identity(4*Ddw.vol, 4*Ddw.vol) + V;
  // std::cout << "calc det" << std::endl;
  // std::cout << std::log(Dov.determinant()) << std::endl;



  return 0; // EXIT_SUCCESS;
}

