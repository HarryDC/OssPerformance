#include <benchmark/benchmark.h>

#include <memory>
#include <vector>
#include <list>

struct Crossing
{
	int id;
	size_t segmentId;
	double segmentLocation;
	bool isAbove;
	size_t sortId;
	Crossing(int id, size_t segmentId, double segmentLocation, bool isAbove) :
		id(id), segmentId(segmentId), segmentLocation(segmentLocation), isAbove(isAbove), sortId(0) {}
};

template <class C>
static void BM_ContainerPushBack(benchmark::State& state)
{
	double loc = 0.0;
	while (state.KeepRunning())
	{
		C container;
		for (int i = 0; i < state.range(0); ++i)
		{
			Crossing c(i, 0, loc, true);
			container.push_back(c);
			loc += 0.01;
		}
	}
}

BENCHMARK_TEMPLATE(BM_ContainerPushBack, std::vector<Crossing>)->Range(16, 8 << 5);
BENCHMARK_TEMPLATE(BM_ContainerPushBack, std::list<Crossing>)->Range(16, 8 << 5);

template <class C>
static void BM_ContainerEmplaceback(benchmark::State& state)
{
	double loc = 0.0;
	while (state.KeepRunning())
	{
		C container;
		for (int i = 0; i < state.range(0); ++i)
		{
			container.emplace_back(i, 0, loc, true);
			loc += 0.01;
		}
	}
}

BENCHMARK_TEMPLATE(BM_ContainerEmplaceback, std::vector<Crossing>)->Range(16, 8 << 5);
BENCHMARK_TEMPLATE(BM_ContainerEmplaceback, std::list<Crossing>)->Range(16, 8 << 5);

template <class C>
static void BM_ContainerPushBackAndDelete(benchmark::State& state)
{
	double loc = 0.0;
	while (state.KeepRunning())
	{
		C container;
		for (int i = 0; i < state.range(0); ++i)
		{
			Crossing c(i, 0, loc, true);
			container.push_back(c);
			loc += 0.01;
		}

		auto it = container.begin();
		for (int i = 0; i < state.range(0) - 4; ++i)
		{
			if (it->id % 3 == 0)
			{
				it = container.erase(it);
				++i;
				it = container.erase(it);
				++i;
			}
			++it;
		}
	}
}
BENCHMARK_TEMPLATE(BM_ContainerPushBackAndDelete, std::vector<Crossing>)->Range(16, 8 << 5);
BENCHMARK_TEMPLATE(BM_ContainerPushBackAndDelete, std::list<Crossing>)->Range(16, 8 << 5);


