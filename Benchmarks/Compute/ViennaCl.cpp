#include <benchmark/benchmark.h>

// #define  VIENNACL_DEBUG_CONTEXT 1

#include <SurgSim/Math/Vector.h>
#include "viennacl/vector.hpp"

namespace
{
int low = 2 << 10;
int high = 2 << 18;

const size_t threads = 8;

struct MatrixData
{
	Eigen::Matrix4d matrix;
	double result;
};

}

static void BM_ViennaCL_VecAddMultiply(benchmark::State& state)
{
	// viennacl::ocl::set_context_platform_index(0, 1);
	viennacl::ocl::set_context_device_type(1, viennacl::ocl::cpu_tag());
	viennacl::ocl::set_context_platform_index(0, 1);
	// std::cout << viennacl::ocl::current_device().info() << std::endl;

	SurgSim::Math::Vector eigenVector1 = SurgSim::Math::Vector::Random(state.range(0));
	SurgSim::Math::Vector eigenVector2 = SurgSim::Math::Vector::Random(state.range(0));
	SurgSim::Math::Vector eigenResult;
	eigenResult.resize(state.range(0));

	viennacl::scalar<double> factor = 2.0;
	viennacl::vector<double> vclVector1(state.range(0));
	viennacl::vector<double> vclVector2(state.range(0));
	viennacl::vector<double> vclResult(state.range(0));

	while (state.KeepRunning())
	{
		viennacl::copy(eigenVector1, vclVector1);
		viennacl::copy(eigenVector2, vclVector2);
		vclResult = vclVector1 * factor + vclVector2;
		viennacl::copy(vclResult, eigenResult);
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
}


BENCHMARK(BM_ViennaCL_VecAddMultiply)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_ViennaCL_VecAddMultiply_CPU(benchmark::State& state)
{
	SurgSim::Math::Vector eigenVector1 = SurgSim::Math::Vector::Random(state.range(0));
	SurgSim::Math::Vector eigenVector2 = SurgSim::Math::Vector::Random(state.range(0));
	SurgSim::Math::Vector eigenResult;
	eigenResult.resize(state.range(0));

	viennacl::scalar<double> factor = 2.0;
	viennacl::vector<double> vclVector(state.range(0));
	viennacl::vector<double> vclResult(state.range(0));

	while (state.KeepRunning())
	{
		eigenResult = eigenVector1 * factor + eigenVector2;
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
}


BENCHMARK(BM_ViennaCL_VecAddMultiply_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);



static void viennaClThroughputEigen(benchmark::State& state)
{
	SurgSim::Math::Vector eigenVector = SurgSim::Math::Vector::Random(state.range(0));
	viennacl::vector<double> vclVector(state.range(0));
	viennacl::copy(eigenVector, vclVector);
	while (state.KeepRunning())
	{
		viennacl::copy(eigenVector, vclVector);
		viennacl::copy(vclVector, eigenVector);
	}

	state.SetBytesProcessed(sizeof(double)*state.range(0)*state.iterations());
	state.SetItemsProcessed(state.range(0)*state.iterations());
}


static void BM_throughput_viennacl_eigen_CPU(benchmark::State& state)
{

	viennacl::ocl::switch_context(0);
	viennaClThroughputEigen(state);
}

BENCHMARK(BM_throughput_viennacl_eigen_CPU)->Range(1024, high);

static void BM_throughput_viennacl_eigen_GPU(benchmark::State& state)
{

	viennacl::ocl::switch_context(1);
	viennaClThroughputEigen(state);
}

BENCHMARK(BM_throughput_viennacl_eigen_GPU)->Range(1024, high);

