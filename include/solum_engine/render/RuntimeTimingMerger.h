#pragma once

#include "solum_engine/render/RuntimeTiming.h"

RuntimeTimingSnapshot mergeRuntimeTimingSnapshots(const RuntimeTimingSnapshot& gpuTiming,
                                                  const RuntimeTimingSnapshot& streamingTiming);
