#include <benchmark/benchmark.h>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

#include <memory>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>

#include <boost/compute/function.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types/fundamental.hpp>
#include <future>

namespace compute = boost::compute;

static void runOpenClBenchmark(benchmark::State& state, int type)
{

	// get default device and setup context

	auto devices = compute::system::devices();
	compute::device testDevice;
	for (const auto& device : devices)
	{
		if (device.type() & type)
		{
			testDevice = device;
			break;
		}
	}

	compute::context context(testDevice);
	compute::command_queue queue(context, testDevice);

	// std::cout << "device: " << testDevice.name() << std::endl;

	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4f> matrices(n);

	// check determinants
	std::vector<float> host_determinants(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4f::Random();
	}

	// copy matrices to the device
	using compute::float16_;

	// function returning the determinant of a 4x4 matrix.
	BOOST_COMPUTE_FUNCTION(float, determinant4x4, (const float16_ m),
	{
		return m.s0* m.s5* m.sa * m.sf + m.s0* m.s6* m.sb * m.sd + m.s0* m.s7* m.s9 * m.se +
		m.s1* m.s4* m.sb * m.se + m.s1* m.s6* m.s8 * m.sf + m.s1* m.s7* m.sa * m.sc +
		m.s2* m.s4* m.s9 * m.sf + m.s2* m.s5* m.sb * m.sc + m.s2* m.s7* m.s8 * m.sd +
		m.s3* m.s4* m.sa * m.sd + m.s3* m.s5* m.s8 * m.se + m.s3* m.s6* m.s9 * m.sc -
		m.s0* m.s5* m.sb * m.se - m.s0* m.s6* m.s9 * m.sf - m.s0* m.s7* m.sa * m.sd -
		m.s1* m.s4* m.sa * m.sf - m.s1* m.s6* m.sb * m.sc - m.s1* m.s7* m.s8 * m.se -
		m.s2* m.s4* m.sb * m.sd - m.s2* m.s5* m.s8 * m.sf - m.s2* m.s7* m.s9 * m.sc -
		m.s3* m.s4* m.s9 * m.se - m.s3* m.s5* m.sa * m.sc - m.s3* m.s6* m.s8* m.sd;
	});

	// M = [ s0 s4 s8 sc ]
	//     [ s1 s5 s9 sd ]
	//     [ s2 s6 sa se ]
	//     [ s3 s7 sb sf ]

	/// Probably wrong, with the elements 3/2 and 2/3 does not quite matter for the benchmark test
	// this is the method that eigen uses so we are doing the same number of multiplications as with the eigen
	// .determinant call
	BOOST_COMPUTE_FUNCTION(float, determinantFast4x4, (const float16_ m),
	{
		return
		((m.s0 * m.s5 - m.s1 * m.s4) * (m.sa * m.sf - m.sb * m.se))
		- ((m.s0 * m.s6 - m.s2 * m.s4) * (m.s9 * m.sf - m.sb * m.sd))
		+ ((m.s0 * m.sb - m.s3 * m.s4) * (m.s9 * m.se - m.sa * m.sd))
		+ ((m.s1 * m.s6 - m.s2 * m.s5) * (m.s8 * m.sf - m.sb * m.sc))
		- ((m.s1 * m.s7 - m.s3 * m.s5) * (m.s8 * m.se - m.sa * m.sc))
		+ ((m.s2 * m.s7 - m.s3 * m.s6) * (m.s8 * m.sd - m.s9 * m.sc));
	});

	compute::vector<float16_> input(n, context);
	compute::vector<float> determinants(n, context);

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
	}
}

static void BM_determinant_OpenCL_CPU(benchmark::State& state)
{
	runOpenClBenchmark(state, compute::device::cpu);
}
BENCHMARK(BM_determinant_OpenCL_CPU)->Range(2 << 10, 2 << 18);


static void BM_determinant_OpenCL_GPU(benchmark::State& state)
{
	runOpenClBenchmark(state, compute::device::gpu);
}

BENCHMARK(BM_determinant_OpenCL_GPU)->Range(2 << 10, 2 << 18);



static void BM_determinant_CPU(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4f> matrices(n);

	// check determinants
	std::vector<float> host_determinants(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4f::Random();
	}

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < n; i++)
		{
			host_determinants[i] = matrices[i].determinant();
		}
	}
}

BENCHMARK(BM_determinant_CPU)->Range(2 << 10, 2 << 18);

float det(const Eigen::Matrix4f& matrix)
{
	return matrix.determinant();
}

static void BM_determinant_CPU_threaded(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4f> matrices(n);

	// check determinants
	std::vector<float> host_determinants(n);


	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4f::Random();
	}

	std::vector<std::future<float>> m_futures(n);

	static auto f = [](const Eigen::Matrix4f & m)
	{
		return m.determinant();
	};

	while (state.KeepRunning())
	{
		for (size_t i = 0; i < n; i++)
		{
			m_futures[i] = std::async(std::launch::async, f, matrices[i]);
		}

		for (size_t i = 0; i < n; i++)
		{
			host_determinants[i] = m_futures[i].get();
		}

	}
}

BENCHMARK(BM_determinant_CPU_threaded)->Range(2 << 10, 2 << 18);

BENCHMARK_MAIN();



