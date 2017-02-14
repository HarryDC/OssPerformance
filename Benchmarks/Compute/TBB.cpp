#include <benchmark/benchmark.h>

#include <tbb/tbb.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/StdVector>

namespace
{
int low = 2 << 10;
int high = 2 << 18;

struct MatrixData
{
	Eigen::Matrix4d matrix;
	double result;
};


}


class ApplyDetVector
{
public:
	ApplyDetVector(const std::vector<Eigen::Matrix4d>& matrices, std::vector<double>* result) :
		m_matrices(matrices), m_result(result)
	{

	}

	void operator()(const tbb::blocked_range<size_t>& r) const
	{
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			(*m_result)[i] = m_matrices[i].determinant();
		}
	}
private:
	const std::vector<Eigen::Matrix4d>& m_matrices;
	std::vector<double>* m_result;
};

class ApplyDetStruct
{
public:
	ApplyDetStruct(std::vector<MatrixData>* matrices) :
		m_matrices(matrices)
	{

	}


	void operator()(const tbb::blocked_range<size_t>& r) const
	{

		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			(*m_matrices)[i].result = (*m_matrices)[i].matrix.determinant();
		}
	}
private:
	std::vector<MatrixData>* m_matrices;
};

static void BM_determinant_tbb_vectors_functor(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4d> matrices(n);

	// check determinants
	std::vector<double> result(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4d::Random();
	}


	while (state.KeepRunning())
	{
		tbb::parallel_for(tbb::blocked_range<size_t>(0, n), ApplyDetVector(matrices, &result));
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);

}

BENCHMARK(BM_determinant_tbb_vectors_functor)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinant_tbb_vectors_lambda(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4d> matrices(n);

	// check determinants
	std::vector<double> result(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4d::Random();
	}


	while (state.KeepRunning())
	{
		tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&matrices, &result](const tbb::blocked_range<size_t>& r)
		{
			for (size_t i = r.begin(); i < r.end(); ++i)
			{
				result[i] = matrices[i].determinant();
			}
		});
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);

}

BENCHMARK(BM_determinant_tbb_vectors_lambda)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinant_tbb_structs_functor(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<MatrixData> matrices(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i].matrix = Eigen::Matrix4d::Random();
	}


	while (state.KeepRunning())
	{
		tbb::parallel_for(tbb::blocked_range<size_t>(0, n), ApplyDetStruct(&matrices));
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);

}

BENCHMARK(BM_determinant_tbb_structs_functor)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinant_tbb_structs_lambda(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<MatrixData> matrices(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i].matrix = Eigen::Matrix4d::Random();
	}


	while (state.KeepRunning())
	{
		tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&matrices](const tbb::blocked_range<size_t>& r)
		{
			for (size_t i = r.begin(); i < r.end(); ++i)
			{
				matrices[i].result = matrices[i].matrix.determinant();
			}
		});
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);

}

BENCHMARK(BM_determinant_tbb_structs_lambda)->Range(low, high)->Unit(benchmark::kMicrosecond);

