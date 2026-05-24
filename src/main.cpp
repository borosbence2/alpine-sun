// alpine-sun — entry point.
//
// Phase 0A skeleton. Opens a window and links forfun_core to validate the
// library boundary. No rendering yet — that arrives in Phase 1 (terrain MVP)
// once the engine primitives (Device, Swapchain, FrameContext) are extracted
// from helmet_demo's monolithic main.cpp in Phase 0B.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "vk_helpers.h"   // forfun_core public header — proves include path
#include "types.h"        // forfun_core shared types — Vertex, UBO, etc.

#include <cstdio>
#include <cstdlib>

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "alpine-sun", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }

    std::printf("alpine-sun: window open, forfun_core linked (kShadowMapSize=%u).\n",
                kShadowMapSize);
    std::printf("Press ESC to quit.\n");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
