#include "glm/glm.hpp"
#include "solum_engine/voxel/Column.h"
#include "solum_engine/voxel/RegionState.h"
#include "solum_engine/resources/Coords.h"
#include "solum_engine/resources/Constants.h"

class Region {
private:
    RegionCoord coord_;
    RegionState state;
    std::array<Column, REGION_TOTAL_COLUMNS> columns_;

public:
    explicit Region(RegionCoord coord) : 
        coord_(coord),
        columns_(make_array<REGION_TOTAL_COLUMNS>([&](auto I) {
            constexpr int idx = static_cast<int>(I);
            const int x = idx % REGION_COLS;
            const int y = idx / REGION_COLS;
            return Column(region_local_to_column(coord_, x, y));
        })) {};

    void update();

    Column* getColumn(int x, int y) {
        if (x >= REGION_COLS || y >= REGION_COLS || x < 0 || y < 0) {
            return nullptr;
        }

        return &columns_[y * REGION_COLS + x];
    }
};