#include <SDL2/SDL.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "network.hpp"
#include "train.hpp"

#define W 28
#define H 28
#define SCALE 30
#define BRUSH_RADIUS 2

static uint8_t canvas[H][W];

void draw_brush(int cx, int cy, bool erase) {
    int r2 = BRUSH_RADIUS * BRUSH_RADIUS;
    float r2di = 1.0f / static_cast<float>(r2);

    for (int y = std::max(cy - BRUSH_RADIUS, 0); y < std::min(cy + BRUSH_RADIUS, H); ++y) {
        int dy = y - cy;
        for (int x = std::max(cx - BRUSH_RADIUS, 0); x < std::min(cx + BRUSH_RADIUS, W); ++x) {
            int dx = x - cx;
            int dxdy2 = dx * dx + dy * dy;
            if (dxdy2 <= r2) {
                float blend = std::max(1.0f - (static_cast<float>(dxdy2) * r2di), 0.0f);
                uint8_t diff = static_cast<uint8_t>(255.0f * blend);
                int pixel = canvas[y][x] + (erase ? -diff : diff);
                canvas[y][x] = static_cast<uint8_t>(std::clamp(pixel, 0, 255));
            }
        }
    }
}

int main(int argc, char **argv) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("NEURAL",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       W * SCALE, H * SCALE, SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        fprintf(stderr, "CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    int hidden[1] = {256};
    neural::Network network(W * H, hidden, sizeof(hidden) / sizeof(hidden[0]), 10);
    if (!network.load("network.bin")) {
        neural::train(network, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 0.05f);
        network.save("network.bin");
    }

    bool running = true;
    bool mouse_down = false;
    bool erase = false;
    bool grid = true;

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    erase = e.button.button == SDL_BUTTON_RIGHT;
                    mouse_down = true;
                    break;
                case SDL_MOUSEBUTTONUP:
                    mouse_down = false;
                    break;
                case SDL_KEYDOWN:
                    switch (e.key.keysym.sym) {
                        case SDLK_SPACE:
                            memset(canvas, 0, sizeof(canvas));
                            break;
                        case SDLK_g:
                            grid = !grid;
                            break;
                        case SDLK_RETURN: {
                            float input[H * W];
                            uint8_t* c = &canvas[0][0];
                            for (int i = 0; i < sizeof(canvas); ++i) {
                                input[i] = static_cast<float>(c[i]) / 255.0f;
                            }

                            float buffer[H * W];
                            neural::process_image(input, buffer, H, W);

                            float digits[10];
                            network.compute(input, digits);

                            int digit = neural::Network::argmax(digits, 10);
                            std::cout << digit << std::endl;
                            break;
                        }
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }

        if (mouse_down) {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            draw_brush(mx / SCALE, my / SCALE, erase);
        }

        // render: draw each canvas pixel as a scaled rect
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_Rect cell;
        cell.w = SCALE;
        cell.h = SCALE;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                uint8_t v = canvas[y][x];
                SDL_SetRenderDrawColor(renderer, v, v, v, 255);
                cell.x = x * SCALE;
                cell.y = y * SCALE;
                SDL_RenderFillRect(renderer, &cell);
            }
        }

        if (grid) {
            SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
            for (int i = 0; i <= W; ++i) {
                int x = i * SCALE;
                SDL_RenderDrawLine(renderer, x, 0, x, H * SCALE);
            }

            for (int i = 0; i <= H; ++i) {
                int y = i * SCALE;
                SDL_RenderDrawLine(renderer, 0, y, W * SCALE, y);
            }
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(5);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}