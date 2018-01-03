#include <benchmark/benchmark.h>

#include <memory>
#include <iostream>

#include "SurgSim/DataStructures/AabbTree.h"
#include "SurgSim/DataStructures/AabbTreeIntersectionVisitor.h"
#include "SurgSim/DataStructures/AabbTreeNode.h"
#include "SurgSim/DataStructures/TriangleMesh.h"
#include "SurgSim/Math/Aabb.h"
#include "SurgSim/Math/MeshShape.h"
#include "SurgSim/Math/RigidTransform.h"
#include "SurgSim/Math/Vector.h"
#include "SurgSim/Math/Quaternion.h"
#include "SurgSim/Testing/MathUtilities.h"
#include "SurgSim/Framework/ApplicationData.h"

#include "SurgSim/DataStructures/AabbTree2.h"

using SurgSim::DataStructures::AabbTree;
using SurgSim::DataStructures::AabbTreeData;
using SurgSim::Math::MeshShape;
using SurgSim::Math::Aabbd;

namespace {
	SurgSim::Math::Vector3d offset(0.0, 0.1125, 0.0);
}
static void BM_BruteForceBuild(benchmark::State& state)
{
	auto tree = std::make_shared<AabbTree>(3);

	auto mesh = std::make_shared<MeshShape>();
	mesh->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));

	std::list<AabbTreeData::Item> items;


	SurgSim::Math::Quaterniond quat = SurgSim::Math::makeRotationQuaternion(0.01, SurgSim::Math::Vector3d(0.0, 1.0, 0.0));
	SurgSim::Math::RigidTransform3d trans = SurgSim::Math::RigidTransform3d::Identity();

	while (state.KeepRunning())
	{
		items.clear();
		for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
		{
			auto triangle = mesh->getTriangle(i);
			Aabbd aabb(SurgSim::Math::makeAabb(
				mesh->getVertex(triangle.verticesId[0]).position,
				mesh->getVertex(triangle.verticesId[1]).position,
				mesh->getVertex(triangle.verticesId[2]).position));
			items.emplace_back(std::make_pair(std::move(aabb), i));
		}
		tree->set(std::move(items));
	}
}


BENCHMARK(BM_BruteForceBuild);

static void BM_Update(benchmark::State& state)
{
	auto tree = std::make_shared<AabbTree>(3);

	auto mesh = std::make_shared<MeshShape>();
	mesh->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));

	std::list<AabbTreeData::Item> items;
	for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
	{
		auto triangle = mesh->getTriangle(i);
		Aabbd aabb(SurgSim::Math::makeAabb(
			mesh->getVertex(triangle.verticesId[0]).position,
			mesh->getVertex(triangle.verticesId[1]).position,
			mesh->getVertex(triangle.verticesId[2]).position));
		items.emplace_back(std::make_pair(std::move(aabb), i));
	}
	tree->set(std::move(items));

	SurgSim::Math::Quaterniond quat = SurgSim::Math::makeRotationQuaternion(0.01, SurgSim::Math::Vector3d(0.0, 1.0, 0.0));
	SurgSim::Math::RigidTransform3d trans = SurgSim::Math::RigidTransform3d::Identity();

	std::vector<Aabbd> bounds;
	bounds.reserve(mesh->getNumTriangles());
	while (state.KeepRunning())
	{
		bounds.clear();
		for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
		{
			auto triangle = mesh->getTriangle(i);
			Aabbd aabb(SurgSim::Math::makeAabb(
				mesh->getVertex(triangle.verticesId[0]).position,
				mesh->getVertex(triangle.verticesId[1]).position,
				mesh->getVertex(triangle.verticesId[2]).position));
			bounds.push_back(std::move(aabb));
		}
		tree->updateBounds(bounds);
	}
}


BENCHMARK(BM_Update);

static void BM_SpatialJoin(benchmark::State& state)
{
		auto tree = std::make_shared<AabbTree>(3);

		auto meshA = std::make_shared<MeshShape>();
		meshA->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));

		auto meshB = std::make_shared<MeshShape>();
		meshB->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));

		auto rhsPose = SurgSim::Math::makeRigidTranslation(offset);
		meshB->transform(rhsPose);

		// update the AABB trees
		meshA->update();
		meshB->update();

		auto aabbA = meshA->getAabbTree();
		auto aabbB = meshB->getAabbTree();

		size_t items = 0;

		std::vector <std::pair<size_t, size_t>> triangles;
		while (state.KeepRunning())
		{
			triangles.clear();
			auto actualIntersection = aabbA->spatialJoin(*aabbB);
			for (auto& pair : actualIntersection)
			{
				auto& leftData = (static_cast<AabbTreeData*>(pair.first->getData().get())->getData());
				auto& rightData = (static_cast<AabbTreeData*>(pair.second->getData().get())->getData());
				SURGSIM_ASSERT(pair.first->getNumChildren() == 0);
				SURGSIM_ASSERT(pair.second->getNumChildren() == 0);

				for (auto& leftTri : leftData)
				{
					for (auto& rightTri : rightData)
					{
						if (leftTri.first.intersects(rightTri.first))
						{
							triangles.emplace_back(leftTri.second, rightTri.second);
						}
					}
				}
			}
			items += triangles.size();
		}
		state.SetItemsProcessed(items);
}

BENCHMARK(BM_SpatialJoin);


static void BM_BruteForceBuild2(benchmark::State& state)
{
	auto tree = std::make_shared<SurgSim::Experimental::AabbTree>();

	auto mesh = std::make_shared<MeshShape>();
	mesh->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));

	std::vector<Aabbd> aabbs;
	std::vector<size_t> indices;
	aabbs.reserve(mesh->getNumTriangles());
	indices.reserve(mesh->getNumTriangles());

	SurgSim::Math::Quaterniond quat = SurgSim::Math::makeRotationQuaternion(0.01, SurgSim::Math::Vector3d(0.0, 1.0, 0.0));
	SurgSim::Math::RigidTransform3d trans = SurgSim::Math::RigidTransform3d::Identity();

	while (state.KeepRunning())
	{
		aabbs.clear();
		indices.clear();
		for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
		{
			auto triangle = mesh->getTriangle(i);
			Aabbd aabb(SurgSim::Math::makeAabb(
				mesh->getVertex(triangle.verticesId[0]).position,
				mesh->getVertex(triangle.verticesId[1]).position,
				mesh->getVertex(triangle.verticesId[2]).position));
			aabbs.emplace_back(std::move(aabb));
			indices.push_back(i);
		}
		tree->build(aabbs, &indices);
	}
}


BENCHMARK(BM_BruteForceBuild2);

static void BM_Update2(benchmark::State& state)
{
	auto tree = std::make_shared<SurgSim::Experimental::AabbTree>();

	auto mesh = std::make_shared<MeshShape>();
	mesh->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));

	std::vector<Aabbd> aabbs;
	std::vector<size_t> indices;
	aabbs.reserve(mesh->getNumTriangles());
	indices.reserve(mesh->getNumTriangles());

	for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
	{
		auto triangle = mesh->getTriangle(i);
		Aabbd aabb(SurgSim::Math::makeAabb(
			mesh->getVertex(triangle.verticesId[0]).position,
			mesh->getVertex(triangle.verticesId[1]).position,
			mesh->getVertex(triangle.verticesId[2]).position));
		aabbs.emplace_back(std::move(aabb));
		indices.push_back(i);
	}
	tree->build(aabbs, &indices);

	SurgSim::Math::Quaterniond quat = SurgSim::Math::makeRotationQuaternion(0.01, SurgSim::Math::Vector3d(0.0, 1.0, 0.0));
	SurgSim::Math::RigidTransform3d trans = SurgSim::Math::RigidTransform3d::Identity();

	std::vector<Aabbd> bounds;
	bounds.reserve(mesh->getNumTriangles());
	while (state.KeepRunning())
	{
		bounds.clear();
		for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
		{
			auto triangle = mesh->getTriangle(i);
			Aabbd aabb(SurgSim::Math::makeAabb(
				mesh->getVertex(triangle.verticesId[0]).position,
				mesh->getVertex(triangle.verticesId[1]).position,
				mesh->getVertex(triangle.verticesId[2]).position));
			bounds.push_back(std::move(aabb));
		}
		tree->update(bounds);
	}
}

BENCHMARK(BM_Update2);

static void fillTree(std::shared_ptr<SurgSim::Experimental::AabbTree> tree,
	std::shared_ptr<MeshShape> mesh)
{

	std::vector<Aabbd> aabbs;
	std::vector<size_t> indices;
	aabbs.reserve(mesh->getNumTriangles());
	indices.reserve(mesh->getNumTriangles());

	for (size_t i = 0; i < mesh->getNumTriangles(); ++i)
	{
		auto triangle = mesh->getTriangle(i);
		Aabbd aabb(SurgSim::Math::makeAabb(
			mesh->getVertex(triangle.verticesId[0]).position,
			mesh->getVertex(triangle.verticesId[1]).position,
			mesh->getVertex(triangle.verticesId[2]).position));
		aabbs.emplace_back(std::move(aabb));
		indices.push_back(i);
	}
	tree->build(aabbs, &indices);
}

static void BM_SpatialJoin2Recursive(benchmark::State& state)
{
	auto treeA = std::make_shared<SurgSim::Experimental::AabbTree>();
	auto meshA = std::make_shared<MeshShape>();
	meshA->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));
	fillTree(treeA, meshA);

	auto treeB = std::make_shared<SurgSim::Experimental::AabbTree>();
	auto meshB = std::make_shared<MeshShape>();
	meshB->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));
	auto rhsPose = SurgSim::Math::makeRigidTranslation(offset);
	meshB->transform(rhsPose);
	fillTree(treeB, meshB);

	std::vector<std::pair<size_t, size_t>> result;
	treeA->recursiveSpatialJoin(*treeB, 0, 0, &result);
	size_t items = 0;
	while (state.KeepRunning())
	{
		result.clear();
		treeA->recursiveSpatialJoin2(*treeB, 0, 0, &result);
		//treeA->recursiveSpatialJoin(*treeB,0, 0, &result);
		items += result.size();
	}
	state.SetItemsProcessed(items);
}

BENCHMARK(BM_SpatialJoin2Recursive);
static void BM_SpatialJoin2(benchmark::State& state)
{
	auto treeA = std::make_shared<SurgSim::Experimental::AabbTree>();
	auto meshA = std::make_shared<MeshShape>();
	meshA->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));
	fillTree(treeA, meshA);

	auto treeB = std::make_shared<SurgSim::Experimental::AabbTree>();
	auto meshB = std::make_shared<MeshShape>();
	meshB->load("arm_collision.ply", SurgSim::Framework::ApplicationData(std::vector<std::string>{"."}));
	auto rhsPose = SurgSim::Math::makeRigidTranslation(offset);
	meshB->transform(rhsPose);
	fillTree(treeB, meshB);

	std::vector<std::pair<size_t, size_t>> result;
	while (state.KeepRunning())
	{
		result.clear();
		treeA->spatialJoin(*treeB, &result);
	}
}

BENCHMARK(BM_SpatialJoin2);


BENCHMARK_MAIN();