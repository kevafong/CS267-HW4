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

 std::vector<double> spmat_vec_mult_2D (Eigen::SparseMatrix<double> M, std::vector<double> V, int r)  {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  // mutiply subset of matrix with subset of vector
  Eigen::VectorXd b(V.size());
  for (int i = 0; i < V.size(); i++)
    b[i] = V[i];
  Eigen::VectorXd xe = M * b;

  // reduce along rows


  if (rank % r == 0) {
    Eigen::VectorXd x_buffer(xe.size());
    for (int i = rank+1; i < rank + r; i++) {
      MPI_Recv(x_buffer.data(), xe.size(), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //MPI_ANY_SOURCE
      xe = xe + x_buffer;
    }
  }

  else {
    int rankhead = int(rank/r) * r;
    MPI_Ssend(xe.data(), xe.size(), MPI_DOUBLE, rankhead, 0, MPI_COMM_WORLD);
  }

  if (rank % r == 0) {
    Eigen::VectorXd x_buffer(V.size());
    for (int i = rank+1; i < rank + r; i++) {
      MPI_Ssend(xe.data(), xe.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
  }

  else {
    int rankhead = int(rank/r) * r;
    MPI_Recv(xe.data(), xe.size(), MPI_DOUBLE, rankhead, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  std::vector<double> x(V.size());
  for (int i = 0; i < V.size(); i++)
    x[i] = xe[i];
  return x;
};



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

// distributed conjugate gradient on 2D distributed matrix
void CG_SPM(const Eigen::SparseMatrix<double> &A,
            const std::vector<double> &b,
            std::vector<double> &x,
            int rows,
            double tol = 1e-6)
{

  assert(b.size() == A.rows());
  x.assign(b.size(), 0.);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(A);

  std::vector<double> r = b, z = (rank % (rows +1) ==0) ? prec(P, r) : offdiag_prec (r);

  std::vector<double> p = z, Ap = spmat_vec_mult_2D(A, p, rows);

  double np2 = (p, Ap), alpha = 0., beta = 0.;
  double nr = sqrt((z, r));
  double epsilon = tol * nr;

  std::vector<double> res = spmat_vec_mult_2D(A, x, rows);
  res += (-1) * b;

  double rres = sqrt((res, res));
  int rankhead = int(rank/rows) * rows;
  int rankdiag = int(rank/rows) * (rows+1);
  int num_it = 0;
  //std::cout << "rank"<<rank<<"head"<<rankhead<<"diag" << rankdiag<< std::endl;
  while (rres > 1e-5)
  {
    //std::cout << "rank"<<rank<<"it"<<num_it<<"rres" << rres<< std::endl;
    alpha = (nr * nr) / (np2);
    x += (+alpha) * p;
    r += (-alpha) * Ap;
    z = (rank % (rows +1) ==0) ? prec(P, r) : offdiag_prec (r);
    nr = sqrt((z, r));
    beta = (nr * nr) / (alpha * np2);
    p = z + beta * p;
    Ap = spmat_vec_mult_2D(A, p, rows);
    np2 = (p, Ap);

    if (rank % (rows + 1) == 0) {
      rres = sqrt((r, r));
      for (int i = rankhead; i <rankhead + rows; i++) {
        if (i != rank)  {
          MPI_Ssend(&rres, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
          std::cout << "rres passed from "<<rank<<" to "<< i <<" "<< rres<< std::endl;
        }
      }
    }
    else  {
      MPI_Recv(&rres, 1, MPI_DOUBLE, rankdiag, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    num_it++;
    if (rank == rankdiag && !(num_it % 1))
    {
      std::cout << "rank: " << rank << "\t";
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

  if (find_arg_idx(argc, argv, "-h") >= 0)
  {
    std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
    return 0;
  }

  int N = find_int_arg(argc, argv, "-N", 100000); // global size

  assert(N % size == 0);
  // NEW MATRIX 2D decomposition
  // Making sparse matrix using eigen instead of map.
  // Compute the local submatrix size and indices
  int t, s;   // # of proccessor along sides
  int mi, ni; // index of rank along dimensions
  int mx, nx; // length of dimensions of local matrix
  int row_offset, col_offset;

  t = sqrt(size);
  s=t;
  // if (r*(r+2)<=size) {
  //   s = r+2;
  // }
  // else if (r*(r+1)<=size) {
  //   s = r+1;
  // }
  // else {
  //   s = r;
  // }

  if (rank < t*s) {
    // find the i's of the processor rank; need to exclude unused
    mi = rank % t;
    ni = rank / t;

    // starts
    row_offset =  (N*ni) / s;
    col_offset =  (N*mi) / t;

    // find the size of local matrices
    mx = (mi < t - 1) ? ((N*(mi+1)) / t) - col_offset : N - col_offset;
    nx = (ni < s - 1) ? ((N*(ni+1)) / s) - row_offset : N - row_offset; 
  }

  double map_time = MPI_Wtime();

  // Allocate memory for the local submatrix in CSR format
  Eigen::SparseMatrix<double, Eigen::RowMajor, int> A_local(nx, mx);
  std::vector<Eigen::Triplet<double>> triplets;

  // Fill in the local submatrix

  for (int i = row_offset; i < nx + row_offset; i++)
  {
    int j, v_ij;
    if (i >= col_offset && i < (mx + col_offset) )  {
      j = i;
      v_ij = 2.0;
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
  }

  // Construct the local submatrix
  A_local.setFromTriplets(triplets.begin(), triplets.end());

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

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << "wall time for Map: " << MPI_Wtime() - map_time << std::endl;

  // initial guess
  std::vector<double> x(n, 0);

  // right-hand side
  std::vector<double> b(n, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  CG_SPM(A_local, b, x, t);

  std::vector<double> total_r(N);
  std::vector<double> total_b(N);

  int diagranks[t];
  for (int i=0; i<t; i++) {
    diagranks[i]= i*(t+1);
  }
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group diag_group;
  MPI_Group_incl(world_group, t, diagranks, &diag_group);
  MPI_Comm DIAG_COMM;
  MPI_Comm_create_group(MPI_COMM_WORLD, diag_group, 0, &DIAG_COMM);

  if (MPI_COMM_NULL != DIAG_COMM) {
    MPI_Gather(b.data(), N/t, MPI_DOUBLE, total_b.data(), N/t, MPI_DOUBLE, 0, DIAG_COMM);
  }

  if (rank==0)  {
    std::cout << "total: ";
    for (int i=0; i<N ;i++)  {
      std::cout<<total_b[i]<<",";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << "wall time for CG: " << MPI_Wtime() - time << std::endl;

  std::vector<double> r = spmat_vec_mult_2D(A_local, x, t) + (-1) * b;
  
  if (MPI_COMM_NULL != DIAG_COMM) {
  MPI_Gather(r.data(), N/t, MPI_DOUBLE, total_r.data(), N/t, MPI_DOUBLE, 0, DIAG_COMM); 
  }

  double err = Norm(total_r) / Norm(total_b);
  
  if (rank == 0)
    std::cout << "|Ax-b|/|b| = " << err << std::endl;

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}