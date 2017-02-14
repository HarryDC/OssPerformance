#include <benchmark/benchmark.h>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1
#define BOOST_COMPUTE_USE_OFFLINE_CACHE 1

#include <memory>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/StdVector>

#include <CL/cl.h>


#include <boost/compute/function.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types/fundamental.hpp>
#include <boost/compute/container/mapped_view.hpp>

#include <future>

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

size_t verify(const std::vector<Eigen::Matrix4d>& matrices, const std::vector<double>& determinants)
{
	size_t result = 0;
	for (int i = 0; i < matrices.size(); ++i)
	{
		double det = matrices[i].determinant();
		double diff = std::abs(det - determinants[i]);
		//std::cout << det << ", " << determinants[i] << ", " << diff << "\n";
		if (diff > 0.00001)
		{
			++result;
		}
	}

	if (result > 0)
	{
		std::cout << result << " determinants did not match.\n";
	}

	return result;
}




bool det1(const std::vector<Eigen::Matrix4d>& matrices, size_t low,
		  size_t high, std::vector<double>* result)
{
	if (high > matrices.size())
	{
		high = matrices.size();
	}

	for (size_t i = low; i < high; ++i)
	{
		(*result)[i] = matrices[i].determinant();
	}

	return true;
}


bool det2(std::vector<MatrixData>* matrices, size_t low,
		  size_t high)
{
	if (high > matrices->size())
	{
		high = matrices->size();
	}

	for (size_t i = low; i < high; ++i)
	{
		(*matrices)[i].result = (*matrices)[i].matrix.determinant();
	}

	return true;
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

// 	return bruteforce_det4_helper(m, 0, 1, 2, 3)
// 		- bruteforce_det4_helper(m, 0, 2, 1, 3)
// 		+ bruteforce_det4_helper(m, 0, 3, 1, 2)
// 		+ bruteforce_det4_helper(m, 1, 2, 0, 3)
// 		- bruteforce_det4_helper(m, 1, 3, 0, 2)
// 		+ bruteforce_det4_helper(m, 2, 3, 0, 1);
//

// 	return (matrix.coeff(j, 0) * matrix.coeff(k, 1) - matrix.coeff(k, 0) * matrix.coeff(j, 1))
// 		* (matrix.coeff(m, 2) * matrix.coeff(n, 3) - matrix.coeff(n, 2) * matrix.coeff(m, 3));
// }


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
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);
}

static void BM_determinant_opencl_CPU(benchmark::State& state)
{
	runOpenClBenchmarkSubIndex(state, compute::device::cpu);
}
BENCHMARK(BM_determinant_opencl_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);


static void BM_determinant_opencl_GPU(benchmark::State& state)
{
	runOpenClBenchmarkSubIndex(state, compute::device::gpu);
}

BENCHMARK(BM_determinant_opencl_GPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

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
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);

}

static void BM_determinantMapped_opencl_CPU(benchmark::State& state)
{
	runOpenClBenchmarkMappedView(state, compute::device::cpu);
}
BENCHMARK(BM_determinantMapped_opencl_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinantMapped_opencl_GPU(benchmark::State& state)
{
	runOpenClBenchmarkMappedView(state, compute::device::gpu);
}
BENCHMARK(BM_determinantMapped_opencl_GPU)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinant_CPU(benchmark::State& state)
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
		det1(matrices, 0, n, &result);
	}

	verify(matrices, result);

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);

}

BENCHMARK(BM_determinant_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond);


static void BM_threaded_determinant_vec_CPU(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<Eigen::Matrix4d> matrices(n);

	// check determinants

	std::vector<double> results(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i] = Eigen::Matrix4d::Random();
	}


	size_t count = n / threads;

	std::vector<std::future<bool>> m_futures(threads);


	while (state.KeepRunning())
	{
		for (size_t i = 0; i < threads; i++)
		{
			m_futures[i] = std::async(std::launch::async, det1, matrices, count * i, count * (i + 1), &results);
			//det(matrices, count * i, count * (i + 1), &result);
		}
		for (auto& future : m_futures)
		{
			future.get();
		}
	}

	verify(matrices, results);

	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);
}


BENCHMARK(BM_threaded_determinant_vec_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond)->UseRealTime();



static void BM_threaded_determinant_struct_CPU(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<MatrixData> matrices(n);

	std::vector<double> results(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i].matrix = Eigen::Matrix4d::Random();
	}


	size_t count = n / threads;

	std::vector<std::future<bool>> m_futures(threads);


	while (state.KeepRunning())
	{
		for (size_t i = 0; i < threads; i++)
		{
			m_futures[i] = std::async(std::launch::async, det2, &matrices, count * i, count * (i + 1));
			//det2(&matrices, count * i, count * (i + 1));
		}
		for (auto& future : m_futures)
		{
			future.get();
		}
	}


	state.SetItemsProcessed(state.range(0)*state.iterations());
	state.SetBytesProcessed(state.range(0)*state.iterations() * sizeof(double) * 16);
}


BENCHMARK(BM_threaded_determinant_struct_CPU)->Range(low, high)->Unit(benchmark::kMicrosecond)->UseRealTime();



static void BM_thread_launch_overhead(benchmark::State& state)
{

	auto f = [](const size_t val)
	{
		return val + 1;
	};

	const size_t threads = state.range(0);

	std::vector<std::future<size_t>> m_futures(threads);

	while (state.KeepRunning())
	{
		size_t sum = 0;
		for (size_t i = 0; i < threads; ++i)
		{
			m_futures[i] = std::async(std::launch::async, f, i);
		}

		for (auto& future : m_futures)
		{
			sum += future.get();
		}
	}
}


//BENCHMARK(BM_thread_launch_overhead)->Range(1, 64);

static void BM_determinant_eigen_pointers(benchmark::State& state)
{
	Eigen::Matrix4d matrix(Eigen::Matrix4d::Random());
	double result;

	Eigen::Matrix4d* mp = &matrix;
	double* rp = &result;

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize((*rp) = mp->determinant());
	}

	state.SetItemsProcessed(state.iterations());
	state.SetBytesProcessed(state.iterations() * sizeof(double) * 16);
}

BENCHMARK(BM_determinant_eigen_pointers);

static void BM_determinant_eigen_call(benchmark::State& state)
{
	std::vector<Eigen::Matrix4d> matrices(1);
	std::vector<double> result(1);

	matrices[0] = Eigen::Matrix4d::Random();

	while (state.KeepRunning())
	{
		det1(matrices, 0, 1, &result);
	}

	state.SetItemsProcessed(state.iterations());
	state.SetBytesProcessed(state.iterations() * sizeof(double) * 16);
}

BENCHMARK(BM_determinant_eigen_call);

static void BM_determinant_eigen_stack(benchmark::State& state)
{
	Eigen::Matrix4d matrix(Eigen::Matrix4d::Random());
	double result;

	while (state.KeepRunning())
	{
		benchmark::DoNotOptimize(result = matrix.determinant());
	}

	state.SetItemsProcessed(state.iterations());
	state.SetBytesProcessed(state.iterations() * sizeof(double) * 16);
}

BENCHMARK(BM_determinant_eigen_stack);

static void BM_determinant_openmp_vectors(benchmark::State& state)
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
		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			result[i] = matrices[i].determinant();
		}

	}

	state.SetItemsProcessed(state.iterations()*n);
	state.SetBytesProcessed(state.iterations() * sizeof(double) * 16);
}

BENCHMARK(BM_determinant_openmp_vectors)->Range(low, high)->Unit(benchmark::kMicrosecond);

static void BM_determinant_openmp_structs(benchmark::State& state)
{
	const size_t n = state.range(0);
	std::vector<MatrixData> matrices(n);

	for (size_t i = 0; i < n; i++)
	{
		matrices[i].matrix = Eigen::Matrix4d::Random();
	}
	while (state.KeepRunning())
	{
		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			matrices[i].result = matrices[i].matrix.determinant();
		}

	}

	state.SetItemsProcessed(state.iterations()*n);
	state.SetBytesProcessed(state.iterations() * sizeof(double) * 16);
}

BENCHMARK(BM_determinant_openmp_structs)->Range(low, high)->Unit(benchmark::kMicrosecond);


static void runOpenClBenchmarkThroughput(benchmark::State& state, int type)
{
	auto testDevice = getDevice(type);
	compute::context context(testDevice);
	compute::command_queue queue(context, testDevice);

	const size_t n = state.range(0);
	std::vector<double> data(n);

	compute::vector<double> gpuTarget(n, context);


	while (state.KeepRunning())
	{
		compute::copy(data.begin(), data.end(), gpuTarget.begin(), queue);
		compute::copy(gpuTarget.begin(), gpuTarget.end(), data.begin(), queue);
	}

	state.SetItemsProcessed(state.iterations() * n);
	state.SetBytesProcessed(state.iterations() * sizeof(double) * n);
}

static void BM_throughput_opencl_CPU(benchmark::State& state)
{
	runOpenClBenchmarkThroughput(state, compute::device::cpu);
}

BENCHMARK(BM_throughput_opencl_CPU)->Range(low, high);

static void BM_throughput_opencl_GPU(benchmark::State& state)
{
	runOpenClBenchmarkThroughput(state, compute::device::gpu);
}

BENCHMARK(BM_throughput_opencl_GPU)->Range(1024, high);