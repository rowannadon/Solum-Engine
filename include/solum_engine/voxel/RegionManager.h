#include "solum_engine/voxel/Region.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/jobsystem/job_system.hpp"

#include <unordered_map>
class RegionManager {
private:
    std::unordered_map<RegionCoord, std::unique_ptr<Region>> regions_;
    jobsystem::JobSystem* js;

public:
    RegionManager(jobsystem::JobSystem *jobsys) : js(jobsys) {};

    bool addRegion(const RegionCoord& coord);
    bool removeRegion(const RegionCoord& coord);

    Region* getRegion(const RegionCoord& coord);

    void update();
};