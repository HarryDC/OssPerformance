#include <benchmark/benchmark.h>

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

static void initViennaCLContexts()
{
	int cpuContext = -1;
	int gpuContext = -1;
	size_t context = 0;
	auto platforms = viennacl::ocl::get_platforms();
	for (auto& platform : platforms)
	{
		for (auto& device : platform.devices())
		{
			std::cout << device.info();

			if ((CL_DEVICE_TYPE_CPU & device.type()) != 0 && cpuContext < 0)
			{
				viennacl::ocl::setup_context(0, device);
				cpuContext = 0;
			}

			if ((CL_DEVICE_TYPE_GPU & device.type()) != 0 && gpuContext < 0)
			{
				viennacl::ocl::setup_context(1, device);
				gpuContext = 1;
			}
		}
	}
}

int main(int argc, char** argv)
{

	//initViennaCLContexts();

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}