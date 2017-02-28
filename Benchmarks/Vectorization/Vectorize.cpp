#include <benchmark/benchmark.h>



#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Geometry>
#include <Eigen/src/StlSupport/StdVector.h>

#include <iostream>
#include <random>

// VS 2015 This doesn't seem to work, the allocator approach is needed !
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)

typedef Eigen::aligned_allocator<Eigen::Vector4d> Allocator4;
typedef Eigen::aligned_allocator<Eigen::Vector3d> Allocator3;

namespace
{
int low = 2 << 4;
int high = 2 << 10;
}

namespace
{
Eigen::SparseMatrix<double> createSparse(size_t rows, size_t cols, double density)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dens(0, 1);
	std::uniform_real_distribution<double> numbers(-0.1, 0.1);

	Eigen::SparseMatrix<double> result(rows, cols);
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < rows; ++j)
		{
			if (dens(gen) < density)
			{
				result.insert(i, j) = numbers(gen);
			}
		}
	}
	return result;
}

}

//////////////////////////////////////////////////////////////////////////
// Dense
//////////////////////////////////////////////////////////////////////////

static void BM_gemm_dense(benchmark::State& state)
{
	const size_t n = state.range(0);

	Eigen::MatrixXd matrix1(Eigen::MatrixXd::Random(n, n));
	Eigen::MatrixXd matrix2(Eigen::MatrixXd::Random(n, n));
	Eigen::MatrixXd matrix3;

	matrix3 = matrix1 * matrix2;

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(matrix3.noalias() = matrix1 * matrix2);
	}

	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_gemm_dense)->RangeMultiplier(2)->Range(low, high);

static void BM_gemv_dense(benchmark::State& state)
{
	const size_t n = state.range(0);

	Eigen::MatrixXd matrix1(Eigen::MatrixXd::Random(n, n));
	Eigen::VectorXd vector1(Eigen::VectorXd::Random(n));
	Eigen::VectorXd vector2;

	vector2 = matrix1 * vector1;

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(vector2.noalias() = matrix1 * vector1);
	}
	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_gemv_dense)->RangeMultiplier(2)->Range(low, high);

//////////////////////////////////////////////////////////////////////////
/// Sparse
//////////////////////////////////////////////////////////////////////////

static void BM_gemm_sparse(benchmark::State& state)
{
	const size_t n = state.range(0);

	Eigen::SparseMatrix<double> matrix1(createSparse(n, n, 0.10));
	Eigen::SparseMatrix<double> matrix2(createSparse(n, n, 0.10));
	Eigen::SparseMatrix<double> matrix3;

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(matrix3 = matrix1 * matrix2);
	}

	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_gemm_sparse)->RangeMultiplier(2)->Range(low, high);

static void BM_gemv_sparse(benchmark::State& state)
{
	const size_t n = state.range(0);

	Eigen::SparseMatrix<double> matrix1(createSparse(n, n, 0.10));
	Eigen::VectorXd vector1(Eigen::VectorXd::Random(n));
	Eigen::VectorXd vector2;

	vector2 = matrix1 * vector1;

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(vector2.noalias() = matrix1 * vector1);
	}
	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_gemv_sparse)->RangeMultiplier(2)->Range(low, high);


//////////////////////////////////////////////////////////////////////////
/// Matrix/Vector (i.e. Mesh Transformations
//////////////////////////////////////////////////////////////////////////

static void BM_4x4mv4m_vector(benchmark::State& state)
{
	const size_t n = state.range(0);

	typedef Eigen::aligned_allocator<Eigen::Vector4f> Allocator;

	std::vector<Eigen::Vector4d> vertices(n);
	std::vector<Eigen::Vector4d> result(n);
	Eigen::Matrix4d matrix(Eigen::Matrix4d::Random());

	for (size_t i = 0; i < n; ++i)
	{
		vertices[i] = Eigen::Vector4d::Random();
	}

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < n; ++i)
		{
			result[i].noalias() = matrix * vertices[i] ;
		}
	}
	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_4x4mv4m_vector)->RangeMultiplier(2)->Range(low, high);

static void BM_transformv3m_vector(benchmark::State& state)
{
	const size_t n = state.range(0);


	std::vector <Eigen::Vector3d> vertices(n);
	std::vector<Eigen::Vector3d> result(n);
	Eigen::Transform<double, 3, Eigen::Isometry> transform;

	transform.makeAffine();
	transform.linear() = Eigen::AngleAxis<double>(0.1, Eigen::Vector3d::UnitX()).matrix();
	transform.translation() = Eigen::Vector3d::Random();

	for (size_t i = 0; i < n; ++i)
	{
		vertices[i] = Eigen::Vector3d::Random();
	}

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < n; ++i)
		{
			result[i] = transform * vertices[i];
		}
	}
	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_transformv3m_vector)->RangeMultiplier(2)->Range(low, high);

static void BM_transformv4m_vector(benchmark::State& state)
{
	const size_t n = state.range(0);

	typedef Eigen::aligned_allocator<Eigen::Vector4f> Allocator;

	std::vector<Eigen::Vector4d> vertices(n);
	std::vector<Eigen::Vector4d> result(n);
	Eigen::Transform<double, 3, Eigen::Isometry> transform;

	transform.makeAffine();
	transform.linear() = Eigen::AngleAxis<double>(0.1, Eigen::Vector3d::UnitX()).matrix();
	transform.translation() = Eigen::Vector3d::Random();

	for (size_t i = 0; i < n; ++i)
	{
		vertices[i] = Eigen::Vector4d::Random();
		vertices[i][3] = 1.0;
	}

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < n; ++i)
		{
			result[i] = transform * vertices[i];
		}
	}
	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_transformv4m_vector)->RangeMultiplier(2)->Range(low, high);


int main(int argc, char** argv)
{
	std::cout << Eigen::SimdInstructionSetsInUse() << "\n";

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}