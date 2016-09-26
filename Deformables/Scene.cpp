#include "Scene.h"

#include "SurgSim/Framework/Framework.h"
#include "SurgSim/Physics/Physics.h"
#include "SurgSim/Graphics/Graphics.h"
#include "SurgSim/Math/Math.h"
#include "SurgSim/Blocks/Blocks.h"

namespace po = boost::program_options;

void setupManagers(std::shared_ptr<SurgSim::Framework::Runtime> runtime,
				   boost::program_options::variables_map& options)
{
	using SurgSim::Framework::Logger;
	SurgSim::Framework::Logger::getLoggerManager()->setThreshold(options["debugLevel"].as<int>());
	SurgSim::Framework::Logger::getLogger("Math/MlcpGaussSeidelSolver")->setThreshold(
		SurgSim::Framework::LOG_LEVEL_SEVERE);

	auto behaviorManager = std::make_shared<SurgSim::Framework::BehaviorManager>();
	auto physicsManager = std::make_shared<SurgSim::Physics::PhysicsManager>();

	if (options["showPhysicsTiming"].as<bool>())
	{
		Logger::getLogger(physicsManager->getName())->setThreshold(SurgSim::Framework::LOG_LEVEL_DEBUG);
	}

	auto graphicsManager = std::make_shared<SurgSim::Graphics::OsgManager>();
	graphicsManager->setRate(60.0);
//
// 	runtime->addManager(graphicsManager);
// 	runtime->addManager(behaviorManager);
	runtime->addManager(physicsManager);
}

void loadScene(std::shared_ptr<SurgSim::Framework::Runtime> runtime,
			   const boost::program_options::variables_map& options)
{

	auto count = options["elementCount"].as<int>();
	for (int i = 0; i < count; ++i)
	{
		runtime->getScene()->addSceneElements(runtime->duplicateSceneElements("sheet.yaml"));
	}
	runtime->addSceneElements("view.yaml");
}

