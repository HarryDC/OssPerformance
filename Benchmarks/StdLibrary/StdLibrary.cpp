#include <benchmark/benchmark.h>

#include <memory>

struct Test
{
	int a;
	int b;
};

struct Derived : public Test
{
	int c;
};

static int GetATestSharedConstRef(const std::shared_ptr<Test>& test)
{
	return test->a;
}

static int GetATestSharedCopy(std::shared_ptr<Test> test)
{
	return test->a;
}


static int GetADerivedShared(const std::shared_ptr<Derived>& test)
{
	return test->a;
}


static int GetATestPtr(const Test* test)
{
	return test->a;
}

static int GetAConstRef(const Test& test)
{
	return test.a;
}



static void BM_StdSharedCreation(benchmark::State& state)
{
	std::vector<std::shared_ptr<Test>> vec;
	while (state.KeepRunning())
	{
		auto variable = std::make_shared<Test>();
		vec.push_back(variable);
	}
}

BENCHMARK(BM_StdSharedCreation);

static void BM_StdSharedCreationMove(benchmark::State& state)
{
	std::vector<std::shared_ptr<Test>> vec;
	while (state.KeepRunning())
	{
		auto variable = std::make_shared<Test>();
		vec.push_back(std::move(variable));
	}
}
BENCHMARK(BM_StdSharedCreationMove);

static void BM_HeapAllocation(benchmark::State& state)
{
	std::vector<Test*> vec;
	while (state.KeepRunning())
	{
		auto variable = new Test;
		vec.push_back(variable);
	}
}

BENCHMARK(BM_HeapAllocation);

static void BM_UniquePtr(benchmark::State& state)
{
	std::vector<std::unique_ptr<Test>> vec;
	while (state.KeepRunning())
	{
		std::unique_ptr<Test> ptr(new Test);
		vec.push_back(std::move(ptr));
	}
}

BENCHMARK(BM_UniquePtr);

static void BM_StackAllocation(benchmark::State& state)
{
	std::vector<Test> vec;
	while (state.KeepRunning())
	{
		Test variable;
		vec.push_back(variable);
	}
}
BENCHMARK(BM_StackAllocation);

static void BM_StackToShared(benchmark::State& state)
{
	std::vector<std::shared_ptr<Test>> vec;
	while (state.KeepRunning())
	{
		Test variable;
		vec.push_back(std::make_shared<Test>(variable));
	}
}
BENCHMARK(BM_StackToShared);

static void BM_PassSharedConstRef(benchmark::State& state)
{
	auto variable = std::make_shared<Test>();
	variable->a = 1;
	variable->b = 2;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetATestSharedConstRef(variable);
	}
}

BENCHMARK(BM_PassSharedConstRef);

static void BM_PassSharedDerefToPointer(benchmark::State& state)
{
	auto variable = std::make_shared<Test>();
	variable->a = 1;
	variable->b = 2;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetATestPtr(variable.get());
	}
}

BENCHMARK(BM_PassSharedDerefToPointer);

static void BM_PassSharedDerefToConstRef(benchmark::State& state)
{
	auto variable = std::make_shared<Test>();
	variable->a = 1;
	variable->b = 2;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetAConstRef(*variable);
	}
}

BENCHMARK(BM_PassSharedDerefToConstRef);


static void BM_PassSharedCopy(benchmark::State& state)
{
	auto variable = std::make_shared<Test>();
	variable->a = 1;
	variable->b = 2;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetATestSharedCopy(variable);
	}
}

BENCHMARK(BM_PassSharedCopy);

static void BM_PassSharedDerived(benchmark::State& state)
{
	auto variable = std::make_shared<Derived>();
	variable->a = 1;
	variable->b = 2;
	variable->c = 3;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetATestSharedConstRef(variable);
	}
}

BENCHMARK(BM_PassSharedDerived);

static void BM_PassConstRef(benchmark::State& state)
{
	Test variable;
	variable.a = 1;
	variable.b = 2;


	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetAConstRef(variable);
	}
}

BENCHMARK(BM_PassConstRef);

static void BM_PassPtr(benchmark::State& state)
{
	Test variable;
	variable.a = 1;
	variable.b = 2;

	auto ptr = &variable;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetATestPtr(ptr);
	}
}

BENCHMARK(BM_PassPtr);

static void BM_PassDerivedConstRef(benchmark::State& state)
{
	Derived variable;
	variable.a = 1;
	variable.b = 2;
	variable.c = 3;


	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetAConstRef(variable);
	}
}

BENCHMARK(BM_PassDerivedConstRef);

static void BM_PassDerivedPtr(benchmark::State& state)
{
	Derived variable;
	variable.a = 1;
	variable.b = 2;
	variable.c = 3;

	auto ptr = &variable;

	size_t i = 0;
	while (state.KeepRunning())
	{
		i += GetATestPtr(ptr);
	}
}

BENCHMARK(BM_PassDerivedPtr);


BENCHMARK_MAIN();