
#include <memory>

#include <boost/program_options.hpp>

namespace SurgSim
{
namespace Framework
{
class Scene;
class Runtime;
}
}

void setupManagers(std::shared_ptr<SurgSim::Framework::Runtime> runtime,
				   boost::program_options::variables_map& variables);

void loadScene(std::shared_ptr<SurgSim::Framework::Runtime> runtime,
			   const boost::program_options::variables_map& options);

