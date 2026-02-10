#pragma once
#include <glm/glm.hpp>
#include <array>

#define CHUNK_SIZE 32
#define CHUNK_SIZE_P 34

constexpr float PI = 3.14159265359f;

enum Direction {
    PlusX,
    MinusX,
    PlusY,
    MinusY,
    PlusZ,
    MinusZ
};
