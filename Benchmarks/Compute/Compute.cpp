#include <benchmark/benchmark.h>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1
#define BOOST_COMPUTE_USE_OFFLINE_CACHE 1

#include <memory>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>

#include <boost/compute/function.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types/fundamental.hpp>
#include <boost/compute/container/mapped_view.hpp>

#include <future>

int low = 2 << 10;
int high = 2 << 18;

namespace compute = boost::compute;

compute::device getDevice(int type)
{
	auto devices = compute::system::devices();
	compute::device testDevice;
	for (const auto& device : devices)
	{
		if (device.type() & type)
		{
			return device;
			break;
		}
	}
	return compute::device();
}

static void runOpenClBenchmarkSubIndex(benchmark::State& state, int type)
{
	auto testDevice = getDevice(type);
	compute::context context(testDevice);
	compute::command_queue queue(context, testDevice);

	// std::cout << "device: " << testDevice.name() << std::endl;

	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4d> matrices(n);

	// check determinants
	std::vector<double> host_determinants(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4d::Random();
	}

	// copy matrices to the device
	using compute::double16_;

	// M = [ s0 s4 s8 sc ]
	//     [ s1 s5 s9 sd ]
	//     [ s2 s6 sa se ]
	//     [ s3 s7 sb sf ]

	/// Probably wrong, with the elements 3/2 and 2/3 does not quite matter for the benchmark test
	// this is the method that eigen uses so we are doing the same number of multiplications as with the eigen
	// .determinant call
	BOOST_COMPUTE_FUNCTION(double, determinantFast4x4, (const double16_ m),
	{
		return
		((m.s0 * m.s5 - m.s1 * m.s4) * (m.sa * m.sf - m.sb * m.se))
		- ((m.s0 * m.s6 - m.s2 * m.s4) * (m.s9 * m.sf - m.sb * m.sd))
		+ ((m.s0 * m.sb - m.s3 * m.s4) * (m.s9 * m.se - m.sa * m.sd))
		+ ((m.s1 * m.s6 - m.s2 * m.s5) * (m.s8 * m.sf - m.sb * m.sc))
		- ((m.s1 * m.s7 - m.s3 * m.s5) * (m.s8 * m.se - m.sa * m.sc))
		+ ((m.s2 * m.s7 - m.s3 * m.s6) * (m.s8 * m.sd - m.s9 * m.sc));
	});

	compute::vector<double16_> input(n, context);
	compute::vector<double> determinants(n, context);
	{
		compute::copy(matrices.begin(), matrices.end(), input.begin(), queue);

		// calculate determinants on the gpu
		compute::transform(
			input.begin(), input.end(), determinants.begin(), determinantFast4x4, queue
		);

		compute::copy(
			determinants.begin(), determinants.end(), host_determinants.begin(), queue
		);
		queue.finish();
	}

	while (state.KeepRunning())
	{

		compute::copy(matrices.begin(), matrices.end(), input.begin(), queue);

		// calculate determinants on the gpu
		compute::transform(
			input.begin(), input.end(), determinants.begin(), determinantFast4x4, queue
		);

		compute::copy(
			determinants.begin(), determinants.end(), host_determinants.begin(), queue
		);

		queue.finish();
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double));
}

static void BM_determinant_OpenCL_CPU(benchmark::State& state)
{
	runOpenClBenchmarkSubIndex(state, compute::device::cpu);
}
BENCHMARK(BM_determinant_OpenCL_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);


static void BM_determinant_OpenCL_GPU(benchmark::State& state)
{
	runOpenClBenchmarkSubIndex(state, compute::device::gpu);
}

BENCHMARK(BM_determinant_OpenCL_GPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void runOpenClBenchmarkMappedView(benchmark::State& state, int type)
{
	auto testDevice = getDevice(type);
	compute::context context(testDevice);
	compute::command_queue queue(context, testDevice);

	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4d> matrices(n);

	// check determinants
	std::vector<double> host_determinants(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4d::Random();
	}

	// copy matrices to the device
	using compute::double16_;

	// M = [ s0 s4 s8 sc ]
	//     [ s1 s5 s9 sd ]
	//     [ s2 s6 sa se ]
	//     [ s3 s7 sb sf ]

	/// Probably wrong, with the elements 3/2 and 2/3 does not quite matter for the benchmark test
	// this is the method that eigen uses so we are doing the same number of multiplications as with the eigen
	// .determinant call
	BOOST_COMPUTE_FUNCTION(double, determinantFast4x4, (const double16_ m),
	{
		return
		((m.s0 * m.s5 - m.s1 * m.s4) * (m.sa * m.sf - m.sb * m.se))
		- ((m.s0 * m.s6 - m.s2 * m.s4) * (m.s9 * m.sf - m.sb * m.sd))
		+ ((m.s0 * m.sb - m.s3 * m.s4) * (m.s9 * m.se - m.sa * m.sd))
		+ ((m.s1 * m.s6 - m.s2 * m.s5) * (m.s8 * m.sf - m.sb * m.sc))
		- ((m.s1 * m.s7 - m.s3 * m.s5) * (m.s8 * m.se - m.sa * m.sc))
		+ ((m.s2 * m.s7 - m.s3 * m.s6) * (m.s8 * m.sd - m.s9 * m.sc));
	});

	std::vector<double16_> matrixData(n);

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < 16; ++j)
		{
			matrixData[i][j] = matrices[i](j);
		}
	}


	compute::mapped_view<double16_> input(&matrixData[0], n, context);
	compute::mapped_view<double> determinants(&host_determinants[0], n, context);

	{

		determinants.map(queue);
		// calculate determinants on the gpu
		compute::transform(
			input.begin(), input.end(), determinants.begin(), determinantFast4x4, queue
		);

		determinants.unmap(queue);
		queue.finish();
	}

	while (state.KeepRunning())
	{

		determinants.map(queue);

		// calculate determinants on the gpu
		compute::transform(
			input.begin(), input.end(), determinants.begin(), determinantFast4x4, queue);

		determinants.unmap(queue);
		queue.finish();
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double));

}

static void BM_determinantMapped_OpenCL_CPU(benchmark::State& state)
{
	runOpenClBenchmarkMappedView(state, compute::device::cpu);
}
BENCHMARK(BM_determinantMapped_OpenCL_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinantMapped_OpenCL_GPU(benchmark::State& state)
{
	runOpenClBenchmarkMappedView(state, compute::device::gpu);
}
BENCHMARK(BM_determinantMapped_OpenCL_GPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinant_CPU(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4d> matrices(n);

	// check determinants
	std::vector<double> host_determinants(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4d::Random();
	}

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < n; i++)
		{
			host_determinants[i] = matrices[i].determinant();
		}
	}

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double));

}

BENCHMARK(BM_determinant_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

void det(const std::vector<Eigen::Matrix4d>& matrices, size_t low, size_t high, std::vector<double>* result)
{
	if (high > matrices.size())
	{
		high = matrices.size();
	}

	for (size_t i = low; i < high; ++i)
	{
		(*result)[i] = matrices[i].determinant();
	}
}


// static void BM_threaded_determinant_CPU(benchmark::State& state)
// {
// 	const size_t n = state.range(0);
// 	std::vector<Eigen::Matrix4d> matrices(n);
//
// 	// check determinants
// 	std::vector<double> result(n);
//
// 	for (size_t i = 0; i < n; i++)
// 	{
// 		matrices[i] = Eigen::Matrix4d::Random();
// 	}
//
//
// 	size_t threads = 8;
// 	size_t count = n / threads;
//
// 	std::vector<std::future<void>> m_futures(count);
//
//
// 	while (state.KeepRunning())
// 	{
// 		for (size_t i = 0; i < count; i++)
// 		{
// 			m_futures[i] = std::async(std::launch::async, det, matrices, count * i, count * (i + 1), &result);
// 		}
// 		for (const auto& future : m_futures)
// 		{
// 			future.wait();
// 		}
// 	}
//
// 	state.SetItemsProcessed(state.range(0)*state.iterations());
// 	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double));
//
// }
//
// BENCHMARK(BM_threaded_determinant_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);






