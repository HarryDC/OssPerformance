#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>
#include <unsupported/Eigen/SparseExtra>

static void BM_lu_sparse_solve(benchmark::State& state)
{
	Eigen::SparseMatrix<double> matrix;
	Eigen::loadMarket(matrix, "../../../Data/distal-artery.mtx");

	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	Eigen::VectorXd rhs(Eigen::VectorXd::Random(matrix.cols()));
	Eigen::VectorXd result(matrix.cols());

	solver.compute(matrix);

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(result = solver.solve(rhs));
	}
	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_lu_sparse_solve);

static void BM_cg_sparse_solve(benchmark::State& state)
{
	Eigen::SparseMatrix<double> matrix;
	Eigen::loadMarket(matrix, "../../../Data/distal-artery.mtx");

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
	Eigen::VectorXd rhs(Eigen::VectorXd::Random(matrix.cols()));
	Eigen::VectorXd result(matrix.cols());

	solver.compute(matrix);

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(result = solver.solve(rhs));
	}
	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_cg_sparse_solve);

static void BM_inv_sparse_solve(benchmark::State& state)
{
	Eigen::SparseMatrix<double> matrix;
	Eigen::loadMarket(matrix, "../../../Data/distal-artery.mtx");

	Eigen::VectorXd rhs(Eigen::VectorXd::Random(matrix.cols()));
	Eigen::VectorXd result(matrix.cols());


	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> inverse(matrix.cols(), matrix.rows());
	inverse = matrix.toDense().inverse();

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(result = inverse * rhs);
	}
	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_inv_sparse_solve);