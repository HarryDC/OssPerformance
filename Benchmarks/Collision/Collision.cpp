#include <benchmark/benchmark.h>

#include <memory>

#include <SurgSim/Collision/CollisionPair.h>
#include <SurgSim/Math/Vector.h>


static void BM_SharedPtrContact(benchmark::State& state)
{
	std::pair<SurgSim::DataStructures::Location, SurgSim::DataStructures::Location> locs;

	while (state.KeepRunning())
	{
		std::vector<std::shared_ptr<SurgSim::Collision::Contact>> contacts;
		for (int i = 0; i < 5; ++i)
		{
			auto contact = std::make_shared<SurgSim::Collision::Contact>(
							   SurgSim::Collision::COLLISION_DETECTION_TYPE_DISCRETE, 2.0, 1.0,
							   SurgSim::Math::Vector3d::Zero(), SurgSim::Math::Vector3d::Zero(), locs);
			contacts.push_back(contact);
		}
	}
}

BENCHMARK(BM_SharedPtrContact);

static void BM_SharedPtrContactList(benchmark::State& state)
{
	std::pair<SurgSim::DataStructures::Location, SurgSim::DataStructures::Location> locs;

	while (state.KeepRunning())
	{
		std::list<std::shared_ptr<SurgSim::Collision::Contact>> contacts;
		for (int i = 0; i < 5; ++i)
		{
			auto contact = std::make_shared<SurgSim::Collision::Contact>(
							   SurgSim::Collision::COLLISION_DETECTION_TYPE_DISCRETE, 2.0, 1.0,
							   SurgSim::Math::Vector3d::Zero(), SurgSim::Math::Vector3d::Zero(), locs);
			contacts.push_back(contact);
		}
	}
}

BENCHMARK(BM_SharedPtrContactList);

static void BM_SharedPtrContactEmplaceBackOdd(benchmark::State& state)
{
	std::vector<std::shared_ptr<SurgSim::Collision::Contact>> contacts;
	std::pair<SurgSim::DataStructures::Location, SurgSim::DataStructures::Location> locs;

	while (state.KeepRunning())
	{
		contacts.push_back(std::make_shared<SurgSim::Collision::Contact>(
							   SurgSim::Collision::COLLISION_DETECTION_TYPE_DISCRETE, 2.0, 1.0,
							   SurgSim::Math::Vector3d::Zero(), SurgSim::Math::Vector3d::Zero(), locs));
	}
}

BENCHMARK(BM_SharedPtrContactEmplaceBackOdd);

static void BM_UniquePtrContact(benchmark::State& state)
{
	std::vector<std::unique_ptr<SurgSim::Collision::Contact>> contacts;
	std::pair<SurgSim::DataStructures::Location, SurgSim::DataStructures::Location> locs;

	while (state.KeepRunning())
	{
		std::unique_ptr<SurgSim::Collision::Contact> contact(new SurgSim::Collision::Contact(
					SurgSim::Collision::COLLISION_DETECTION_TYPE_DISCRETE, 2.0, 1.0,
					SurgSim::Math::Vector3d::Zero(), SurgSim::Math::Vector3d::Zero(), locs));
		contacts.push_back(std::move(contact));
	}
}

BENCHMARK(BM_UniquePtrContact);


static void BM_StackContactPushBack(benchmark::State& state)
{
	std::vector<SurgSim::Collision::Contact> contacts;
	std::pair<SurgSim::DataStructures::Location, SurgSim::DataStructures::Location> locs;

	while (state.KeepRunning())
	{
		SurgSim::Collision::Contact contact(
			SurgSim::Collision::COLLISION_DETECTION_TYPE_DISCRETE, 2.0, 1.0,
			SurgSim::Math::Vector3d::Zero(), SurgSim::Math::Vector3d::Zero(), locs);
		contacts.push_back(contact);
	}
}

BENCHMARK(BM_StackContactPushBack);

static void BM_StackContactEmplaceBack(benchmark::State& state)
{
	std::vector<SurgSim::Collision::Contact> contacts;
	std::pair<SurgSim::DataStructures::Location, SurgSim::DataStructures::Location> locs;

	while (state.KeepRunning())
	{
		contacts.emplace_back(SurgSim::Collision::COLLISION_DETECTION_TYPE_DISCRETE, 2.0, 1.0,
							  SurgSim::Math::Vector3d::Zero(), SurgSim::Math::Vector3d::Zero(), locs);
	}
}

BENCHMARK(BM_StackContactEmplaceBack);



BENCHMARK_MAIN();