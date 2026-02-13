#include <atomic>

class ChunkState {
	std::atomic<uint32_t> blockEpoch;
	std::atomic<uint32_t> lightEpoch;
	std::atomic<uint32_t> lodEpoch;
	std::atomic<uint32_t> meshL0Epoch;

	std::atomic<uint32_t> dirtyFlags;

};