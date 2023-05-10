#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <numeric>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat; // declares a row-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class MapMatrix
{
public:
  typedef std::pair<int, int> N2;

  std::map<N2, double> data;
  int nbrow;
  int nbcol;

public:
  MapMatrix(const int &nr, const int &nc) : nbrow(nr), nbcol(nc){};

  MapMatrix(const MapMatrix &m) : nbrow(m.nbrow), nbcol(m.nbcol), data(m.data){};

  MapMatrix &operator=(const MapMatrix &m)
  {
    if (this != &m)
    {
      nbrow = m.nbrow;
      nbcol = m.nbcol;
      data = m.data;
    }
    return *this;
  }

  int NbRow() const { return nbrow; }
  int NbCol() const { return nbcol; }

  double operator()(const int &j, const int &k) const
  {
    auto search = data.find(std::make_pair(j, k));
    if (search != data.end())
      return search->second;
    return 0;
  }

  double &Assign(const int &j, const int &k)
  {
    return data[std::make_pair(j, k)];
  }

  // parallel matrix-vector product with distributed vector xi
  std::vector<double> operator*(const std::vector<double> &xi) const
  {

    std::vector<double> x(NbCol());
    std::copy(xi.begin(), xi.end(), x.begin());

    std::vector<double> b(NbRow(), 0.);
    for (auto it = data.begin(); it != data.end(); ++it)
    {
      int j = (it->first).first;
      int k = (it->first).second;
      double Mjk = it->second;
      b[j] += Mjk * x[k];
    }

    return b;
  }
};

 std::vector<double> spmat_vec_mult_2D (Eigen::SparseMatrix<double> M, std::vector<double> V)  {
  // mutiply subset of matrix with subset of vector
  Eigen::VectorXd b(V.size());
  for (int i = 0; i < V.size(); i++)
    b[i] = V[i];
  Eigen::VectorXd xe = M * b;
  
  // reduce along rows
  
  if (rank % r = 0) {
    Eigen::VectorXd x_buffer(V.size());
    for (int i = rank+1; i < rank + r; i++) {
      MPI_Recv(&x_buffer.data(), xe.size, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //MPI_ANY_SOURCE
      xe = xe + x_buffer;
    }
  }

  else {
    int rankhead = int(rank/r) * r;
    MPI_Send(xe.data(), xe.size(), MPI_DOUBLE, rankhead, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank % r = 0) {
    Eigen::VectorXd x_buffer(V.size());
    for (int i = rank+1; i < rank + r; i++) {
      MPI_Send(xe.data(), x.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      MPI_Recv(&x_buffer.data(), xe.size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  else {
    int rankhead = int(rank/r) * r;
    MPI_Recv(&xe.data(), xe.size, MPI_DOUBLE, rankhead, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  std::vector<double> x(V.size());
  for (int i = 0; i < V.size(); i++)
    x[i] = xe[i];
  return x;

  // MPI gather?? 
};

// distributed conjugate gradient on 2D distributed matrix
void CG_SPM(const Eigen::SparseMatrix<double> &A,
            const std::vector<double> &b,
            std::vector<double> &x,
            int rows;
            int row_offset,
            double tol = 1e-6)
{

  assert(b.size() == A.rows());
  x.assign(b.size(), 0.);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  //Eigen::SparseMatrix<double> n_n_preconditioner = A.middleCols(row_offset, A.rows());

  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(A);

  std::vector<double> r = b, z;
  if (rank % (rows + 1) = 0) z = prec(P, r)
  else z = offdiag_prec(r);
  std::vector<double> p = z, Ap = spmat_vec_mult_2D(A, p);

  double np2 = (p, Ap), alpha = 0., beta = 0.;
  double nr = sqrt((z, r));
  double epsilon = tol * nr;

  std::vector<double> res = sm_vec_mult(A, x);

  res += (-1) * b;

  double rres = sqrt((res, res));

  int num_it = 0;
  while (rres > 1e-5)
  {
    alpha = (nr * nr) / (np2);
    x += (+alpha) * p;
    r += (-alpha) * Ap;
    z = prec(P, r);
    nr = sqrt((z, r));
    beta = (nr * nr) / (alpha * np2);
    p = z + beta * p;
    Ap = sm_vec_mult(A, p);
    np2 = (p, Ap);

    rres = sqrt((r, r));

    num_it++;
    if (rank == 0 && !(num_it % 1))
    {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << rres << "\n";
    }
  }
}

#include <cmath>

// parallel scalar product (u,v) (u and v are distributed)
double operator,(const std::vector<double> &u, const std::vector<double> &v)
{
  assert(u.size() == v.size());
  double sp = 0.;
  for (int j = 0; j < u.size(); j++)
  {
    sp += u[j] * v[j];
  }

  return sp;
}

// norm of a vector u
double Norm(const std::vector<double> &u)
{
  return sqrt((u, u));
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double> &u, const std::vector<double> &v)
{
  assert(u.size() == v.size());
  std::vector<double> w = u;
  for (int j = 0; j < u.size(); j++)
  {
    w[j] += v[j];
  }
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double &a, const std::vector<double> &u)
{
  std::vector<double> w(u.size());
  for (int j = 0; j < w.size(); j++)
  {
    w[j] = a * u[j];
  }
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double> &u, const std::vector<double> &v)
{
  assert(u.size() == v.size());
  for (int j = 0; j < u.size(); j++)
  {
    u[j] += v[j];
  }
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &P, const std::vector<double> &u)
{
  Eigen::VectorXd b(u.size());
  for (int i = 0; i < u.size(); i++)
    b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b);
  std::vector<double> x(u.size());
  for (int i = 0; i < u.size(); i++)
    x[i] = xe[i];
  return x;
}

std::vector<double> offdiag_prec(const std::vector<double> &u)
{
  std::vector<double> x(u.size());
    for (int i = 0; i < u.size(); i++)
      x[i] = 0;
  return x;
}

// distributed conjugate gradient
void CG(const MapMatrix &A,
        const std::vector<double> &b,
        std::vector<double> &x,
        double tol = 1e-6)
{

  assert(b.size() == A.NbRow());
  x.assign(b.size(), 0.);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int n = A.NbRow();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for (auto it = A.data.begin(); it != A.data.end(); ++it)
  {
    int j = (it->first).first;
    int k = (it->first).second;
    if (k >= 0 && k < n)
      coefficients.push_back(Eigen::Triplet<double>(j, k, it->second));
  }

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n, n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  std::vector<double> r = b, z = prec(P, r), p = z, Ap = A * p;
  double np2 = (p, Ap), alpha = 0., beta = 0.;
  double nr = sqrt((z, r));
  double epsilon = tol * nr;

  std::vector<double> res = A * x;
  res += (-1) * b;

  double rres = sqrt((res, res));

  int num_it = 0;
  while (rres > 1e-5)
  {
    alpha = (nr * nr) / (np2);
    x += (+alpha) * p;
    r += (-alpha) * Ap;
    z = prec(P, r);
    nr = sqrt((z, r));
    beta = (nr * nr) / (alpha * np2);
    p = z + beta * p;
    Ap = A * p;
    np2 = (p, Ap);

    rres = sqrt((r, r));

    num_it++;
    if (rank == 0 && !(num_it % 1))
    {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << rres << "\n";
    }
  }
}

// Command Line Option Processing
int find_arg_idx(int argc, char **argv, const char *option)
{
  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], option) == 0)
    {
      return i;
    }
  }
  return -1;
}

int find_int_arg(int argc, char **argv, const char *option, int default_value)
{
  int iplace = find_arg_idx(argc, argv, option);

  if (iplace >= 0 && iplace < argc - 1)
  {
    return std::stoi(argv[iplace + 1]);
  }

  return default_value;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv); // Initialize the MPI environment

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  if (rank == 0)
    std::cout << "Number of procs " << size << std::endl;


  //   //testing for MPI Scatter
  // int buff = 6;
  // double *sendbuf, rbuf[buff]; 
  // int *displs, *scounts;
  // int testsize = 6*size-2;
  // sendbuf = (double *)malloc(testsize*sizeof(double));
  // displs = (int *)malloc(size*sizeof(int)); 
  // scounts = (int *)malloc(size*sizeof(int)); 
  // for (int i=0; i<size; ++i) { 
  //       displs[i] = (testsize*i)/size; 
  // } 
  // for (int i=0; i<size; ++i) { 
  //       scounts[i] = (i<size-1) ? displs[i+1] - displs[i] : testsize - displs[i];
  // } 
  // if (rank == 0) {
  // std::cout << "Sendbuffer: ";
  //   for (int i=0; i<testsize; i++)   {
  //     sendbuf[i] = i;
  //     std::cout << sendbuf[i] << " ";
  //   }
  //   std::cout << std::endl;
  // std::cout << "Displacements: ";
  // for (int i=0; i<size; i++)  std::cout << displs[i] << " ";
  // std::cout << std::endl;
  // std::cout << "SCounts: ";
  // for (int i=0; i<size; i++)  std::cout << scounts[i] << " ";
  // std::cout << std::endl;
  // }
  // MPI_Scatterv( sendbuf, scounts, displs, MPI_DOUBLE, rbuf, scounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // std::cout << "At proc" << rank << " Receive Buffer: ";
  // for (int i=0; i<scounts[rank]; i++)  std::cout << rbuf[i] << " ";
  //   std::cout << std::endl;

  if (find_arg_idx(argc, argv, "-h") >= 0)
  {
    std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
    return 0;
  }

  int N = find_int_arg(argc, argv, "-N", 100000); // global size

  assert(N % size == 0);
  // NEW MATRIX 1D decomposition
  // Making sparse matrix using eigen instead of map.
  // Compute the local submatrix size and indices
  int r, s;   // # of proccessor along sides
  int mi, ni; // index of rank along dimensions
  int mx, nx; // length of dimensions of local matrix
  int row_offset, col_offset;

  r = sqrt(size);
  s=r;
  // if (r*(r+2)<=size) {
  //   s = r+2;
  // }
  // else if (r*(r+1)<=size) {
  //   s = r+1;
  // }
  // else {
  //   s = r;
  // }

  if (rank < r*s) {
    // find the i's of the processor rank; need to exclude unused
    mi = rank % r;
    ni = rank / r;

    // starts
    row_offset =  (N*ni) / s;
    col_offset =  (N*mi) / r;

    // find the size of local matrices
    mx = (mi < r - 1) ? ((N*(mi+1)) / r) - col_offset : N - col_offset;
    nx = (ni < s - 1) ? ((N*(ni+1)) / s) - row_offset : N - row_offset; 
  }


  // std::cout << "Defined the local sparse matrix" << std::endl;
  // std::cout << "mi: " << mi << std::endl;
  // std::cout << "ni: " << ni << std::endl;
  // std::cout << "mx: " << mx << std::endl;
  // std::cout << "nx: " << nx << std::endl;
  // std::cout << "col_offset: " << col_offset << std::endl;
  // std::cout << "row_offset: " << row_offset << std::endl;

  double map_time = MPI_Wtime();

  // Allocate memory for the local submatrix in CSR format
  Eigen::SparseMatrix<double, Eigen::RowMajor, int> A_local(nx, mx);
  std::vector<Eigen::Triplet<double>> triplets;

  std::cout << "Allocated memory for sparse matrix" << std::endl;

  // Fill in the local submatrix

  for (int i = row_offset; i < nx + row_offset; i++)
  {
    int j, v_ij;
    if (i >= col_offset && i < (mx + col_offset) )  {
      j = i;
      v_ij = 2.0;
    // When pushing back to each processors local sparse matrix we must do (i - row_offset), this is because each processor, will have
    // (nx x N) array, and so even though we are on some value of nx, its relative to the current processor.
      triplets.push_back(T(i - row_offset, j - col_offset, v_ij));
    }

    if (i - 1 >= col_offset && (i - 1) < (mx + col_offset))
    {
      j = i - 1;
      v_ij = -1;
      triplets.push_back(T(i - row_offset, j - col_offset, v_ij));
    }

    if (i + 1 >= col_offset && (i + 1) < (mx + col_offset))
    {
      j = i + 1;
      v_ij = -1;
      triplets.push_back(T(i - row_offset, j - col_offset, v_ij));
    }

    // if (i + N < N)
    // {
    //   j = i + N;
    //   v_ij = -1;
    //   triplets.push_back(T(i - row_offset, j - col_offset, v_ij));
    // }

    // if (i - N >= 0)
    // {
    //   j = i - N;
    //   v_ij = -1;
    //   triplets.push_back(T(i - row_offset, j - col_offset, v_ij));
    // }

    std::cout << "Filled the local sparse matrix" << std::endl;
  }

  

  // Construct the local submatrix
  A_local.setFromTriplets(triplets.begin(), triplets.end());

  std::cout << "Constructed the following local sparse matrix on processor: " << rank << "\n" << A_local << std::endl;


  // if (rank == 0)
  // {  
  //   std::cout << "Print matrix " << std::endl;
  //   std::cout << M << std::endl;
  //   std::cout << "Values: ";
  //   for (double *p= M.valuePtr(); p != M.valuePtr() + M.nonZeros(); p++)  std::cout << *p << " ";
  //   std::cout << std::endl;
  //   std::cout << "Inner: ";
  //   for (int *q= M.innerIndexPtr(); q != M.innerIndexPtr() + M.nonZeros(); q++)  std::cout << *q << " ";
  //   std::cout << std::endl;
  //   std::cout << "Outer: ";
  //   for (int *r= M.outerIndexPtr(); r != M.outerIndexPtr() + M.rows(); r++)  std::cout << *r << " ";
  //   std::cout << std::endl;
  // }

  MPI_Barrier(MPI_COMM_WORLD);
  

 // ORIGINAL IMPLEMENTATION
  // local rows of the 1D Laplacian matrix; local column indices start at -1 for rank > 0
  int n = N / size; // number of local rows

  // row-distributed matrix
  

  MapMatrix A(n, N);

  int offset = n * rank;

  for (int i = 0; i < n; i++)
  {
    A.Assign(i, i) = 2.0;
    if (offset + i - 1 >= 0)
      A.Assign(i, i - 1) = -1;
    if (offset + i + 1 < N)
      A.Assign(i, i + 1) = -1;
    if (offset + i + N < N)
      A.Assign(i, i + N) = -1;
    if (offset + i - N >= 0)
      A.Assign(i, i - N) = -1;
  }

  // // prints map
  // for (const auto &elem : A.data)
  // {
  //   std::cout << elem.first.first << " " << elem.first.second << " " << elem.second << "\n";
  // }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << "wall time for Map: " << MPI_Wtime() - map_time << std::endl;

  // initial guess
  std::vector<double> x(n, 0);

  // right-hand side
  std::vector<double> b(n, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  CG(A, b, x);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << "wall time for CG: " << MPI_Wtime() - time << std::endl;

  //std::vector<double> r = A * x + (-1) * b;
  std::vector<double> r = spmat_vec_mult_2D(A, x, t) + (-1) * b;

   MPI_Gatherv(r.data(), n, MPI_DOUBLE, total_r.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double err = Norm(r) / Norm(b);
  if (rank == 0)
    std::cout << "|Ax-b|/|b| = " << err << std::endl;

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}