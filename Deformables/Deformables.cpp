// This file is a part of the SimQuest OpenSurgSim extension.
// Copyright 2012-2015, SimQuest Solutions Inc.

#include <iostream>
#include <boost/program_options.hpp>

#include "SurgSim/Framework/Runtime.h"
#include "SurgSim/Framework/Scene.h"

#include "Scene.h"


namespace po = boost::program_options;

boost::program_options::options_description createCommandLineOptions()
{
	boost::program_options::options_description commandLine("Allowed options");
	commandLine.add_options()("help", "produce help message")
	("config-file", boost::program_options::value<std::string>()->default_value("config.txt"), "The config file to use")
	("showPhysicsTiming", boost::program_options::value<bool>()->default_value(true),
	 "whether to display physics loop timing.")
	("debugLevel", boost::program_options::value<int>()->default_value(2),
	 "what debug level to use, debug, info, warn [default], severe, critical")
	("elementCount", boost::program_options::value<int>()->default_value(1), "How many deformables to add.")
	;
	return commandLine;
}

int main(int argc, char* argv[])
{
	auto commandLine = createCommandLineOptions();
	po::variables_map variables;
	try
	{
		po::store(po::parse_command_line(argc, argv, commandLine), variables);
	}
	catch (po::error& e)
	{
		std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
		std::cerr << commandLine << std::endl;
		return 1;
	}

	if (variables.count("help"))
	{
		std::cout << commandLine << "\n";
		return 1;
	}

	auto runtime = std::make_shared<SurgSim::Framework::Runtime>(variables["config-file"].as<std::string>());
	setupManagers(runtime, variables);
	loadScene(runtime, variables);
	runtime->execute();

	return 0;
}
