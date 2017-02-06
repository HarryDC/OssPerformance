#include <benchmark/benchmark.h>


int main(int argc, char** argv)
{

	//initViennaCLContexts();

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}