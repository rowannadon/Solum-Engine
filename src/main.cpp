// In main.cpp
#include "solum_engine/core/Application.h"

int main() {
    Application app;

    if (!app.Initialize()) {
        return 1;
    }

    // Warning: this is still not Emscripten-friendly, see below
    while (app.IsRunning()) {
        app.MainLoop();
    }

    app.Terminate();

    return 0;
}

