#include <benchmark/benchmark.h>

// #define  VIENNACL_DEBUG_CONTEXT 1

#include <SurgSim/Math/Vector.h>
#include "viennacl/vector.hpp"


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


BENCHMARK(BM_ViennaCL_VecAddMultiply)->Range(2 << 10, 2 << 18)->Unit(benchmark::kMicrosecond);

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


BENCHMARK(BM_ViennaCL_VecAddMultiply_CPU)->Range(2 << 10, 2 << 18)->Unit(benchmark::kMicrosecond);


static void toOpenClDevice(benchmark::State& state)
{
	SurgSim::Math::Vector eigenVector = SurgSim::Math::Vector::Random(state.range(0));
	viennacl::vector<double> vclVector(state.range(0));
	while (state.KeepRunning())
	{
		viennacl::copy(eigenVector, vclVector);
	}

	state.SetBytesProcessed(sizeof(double)*state.range(0)*state.iterations());
}

static void BM_Overhead_CopyToCPU(benchmark::State& state)
{

	viennacl::ocl::switch_context(0);
	toOpenClDevice(state);
}

BENCHMARK(BM_Overhead_CopyToCPU)->Range(2 << 10, 2 << 20);

static void BM_Overhead_CopyToGPU(benchmark::State& state)
{
	viennacl::ocl::switch_context(1);
	toOpenClDevice(state);
}

BENCHMARK(BM_Overhead_CopyToGPU)->Range(2 << 10, 2 << 20);



static void fromOpenClDevice(benchmark::State& state)
{
	SurgSim::Math::Vector eigenVector = SurgSim::Math::Vector::Random(state.range(0));
	viennacl::vector<double> vclVector(state.range(0));
	viennacl::copy(eigenVector, vclVector);
	while (state.KeepRunning())
	{
		viennacl::copy(vclVector, eigenVector);
	}

	state.SetBytesProcessed(sizeof(double)*state.range(0)*state.iterations());
}


static void BM_Overhead_CopyFromCPU(benchmark::State& state)
{

	viennacl::ocl::switch_context(0);
	fromOpenClDevice(state);
}

BENCHMARK(BM_Overhead_CopyFromCPU)->Range(2 << 10, 2 << 20);

static void BM_Overhead_CopyFromGPU(benchmark::State& state)
{

	viennacl::ocl::switch_context(1);
	fromOpenClDevice(state);
}

BENCHMARK(BM_Overhead_CopyFromGPU)->Range(2 << 10, 2 << 20);

