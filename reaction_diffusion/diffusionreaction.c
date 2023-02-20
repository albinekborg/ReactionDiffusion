#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <SDL2/SDL.h>

#define WINDOW_SIZE 1000

#define N 800

#define DIFFUSION_U 1
#define DIFFUSION_V 0.3
#define DECAY 0.062
#define FEED 0.06
#define DT 1
#define KERNEL_SIZE 3

int PADDED_N = N+2;

void laplacian(float** u, float** v, float kernel[3][3], float** new_u, float** new_v) {
    int row, col, krow, kcol;
    float sum_u, sum_v;
    // Wrapping
    /* 
    float ut, ub, ur, ul, vt, vb, vr, vl;
    for (int ix = 0; ix < PADDED_N; ix++) {
        // Handle left and right boundaries
        ul = u[ix][(PADDED_N-2+N)%N+1];
        ur = u[ix][2%N+1];
        vl = v[ix][(PADDED_N-2+N)%N+1];
        vr = v[ix][2%N+1];

        // Handle top and bottom boundaries
        ut = u[(PADDED_N-2+N)%N+1][ix];
        ub = u[2%N+1][ix];
        vt = v[(PADDED_N-2+N)%N+1][ix];
        vb = v[2%N+1][ix];

        // Calculate sum_u and sum_v using ut, ub, etc.
        for (krow = -1; krow <= 1; krow++) {
            for (kcol = -1; kcol <= 1; kcol++) {
                sum_u += kernel[krow+1][kcol+1] * u[(row+krow+N)%N+1][(col+kcol+N)%N+1];
                sum_v += kernel[krow+1][kcol+1] * v[(row+krow+N)%N+1][(col+kcol+N)%N+1];
            }
        }
        new_u[row][col] = sum_u;
        new_v[row][col] = sum_v;
    } */
    
    // Update the displayed field
    for (row = 1; row < N+1; row++) {
        for (col = 1; col < N+1; col++) {
            sum_u = 0.0;
            sum_v = 0.0;
            for (krow = -1; krow <= 1; krow++) {
                for (kcol = -1; kcol <= 1; kcol++) {
                    sum_u += kernel[krow+1][kcol+1] * u[row+krow][col+kcol];
                    sum_v += kernel[krow+1][kcol+1] * v[row+krow][col+kcol];
                }
            }
            new_u[row][col] = sum_u;
            new_v[row][col] = sum_v;
        }
    }
}

void dynamics(float conv_kernel[3][3], float** u, float** v, float** temp_u, float** temp_v) {
    laplacian(u, v, conv_kernel, temp_u, temp_v);
    for (int i = 0; i < N+2; i++) {
        for (int j = 0; j < N+2; j++) {
	    float nf2 = (float) N;
	    float fi = (float) i;
	    float fj = (float) j;
            float alpha = 1.f + (fi-nf2)/6000.f;
            float beta = 1.f + (fj-nf2)/1000.f;
	    // Reaction
            u[i][j] += (DIFFUSION_U*temp_u[i][j] - u[i][j]*v[i][j]*v[i][j] + beta*FEED*(1.f-u[i][j]))*DT;
            v[i][j] += (DIFFUSION_V*temp_v[i][j] + u[i][j]*v[i][j]*v[i][j] - (alpha*DECAY+beta*FEED)*v[i][j])*DT;
	    // Update
            u[i][j] = fmaxf(0.0f, fminf(1.0f, u[i][j]));
            v[i][j] = fmaxf(0.0f, fminf(1.0f, v[i][j]));
        }
    }
}


int update_graphics(float** u, SDL_Window* window, SDL_Surface* surface, SDL_Renderer* renderer, SDL_Event event) {
    
    // Check if user wants to close the window
    while (SDL_PollEvent(&event)) {
        if (event.type==SDL_QUIT) {
	    SDL_DestroyWindow(window);
	    SDL_Quit();
	    return 0;
        }
    }
    
    // Set pixel values on surface
    SDL_LockSurface(surface);
    for (int row = 1; row < N+1; row++) {
        for (int col = 1; col < N+1; col++) {
	    int grey_value = u[row][col]*255.f;
	    Uint32 pixel = SDL_MapRGB(surface->format, grey_value, grey_value, grey_value);
	    ((Uint32*)surface->pixels)[col*N+row] = pixel;
        }
    }

    // Render new surface
    SDL_UnlockSurface(surface);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
    
    return 1;
}

void initialize(float** u, float** v) {
    int range_from = PADDED_N/2 - PADDED_N/10;
    int range_to = PADDED_N/2 + PADDED_N/10;
    for (int i = 1; i < PADDED_N; i++) {
        for (int j = 1; j < PADDED_N; j++) {
            u[i][j] = 0.6f + 0.2f*((float)rand() / RAND_MAX);
            v[i][j] = 0.0f + 0.2f*((float)rand() / RAND_MAX);
        }
    }
    for (int i = range_from; i < range_to; i++) {
        for (int j = range_from; j < range_to; j++) {
            v[i][j] = 0.5f;
            u[i][j] = 0.25f;
        }
    }
}

int main(int argc, char *argv[]) {

    // Initialise random number generator
    srand(time(NULL));

    // Set u, v, temp_u, temp_v
    float** u      = (float**)malloc(PADDED_N*sizeof(float*));
    float** v      = (float**)malloc(PADDED_N*sizeof(float*));
    float** temp_u = (float**)malloc(PADDED_N*sizeof(float*));
    float** temp_v = (float**)malloc(PADDED_N*sizeof(float*));
    for (int i = 0; i < PADDED_N; i++) {
        u[i]      = (float*)calloc(PADDED_N, sizeof(float));
        v[i]      = (float*)calloc(PADDED_N, sizeof(float));
        temp_u[i] = (float*)calloc(PADDED_N, sizeof(float));
        temp_v[i] = (float*)calloc(PADDED_N, sizeof(float));
    }
    initialize(u, v);
    
    // Set laplacian kernel
    float conv_kernel[KERNEL_SIZE][KERNEL_SIZE] = {{0.05f, 0.20f, 0.05f},
                                                   {0.20f,-1.00f, 0.20f},
                                                   {0.05f, 0.20f, 0.05f}};
    
    // Initialize window and related stuff
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Graphic window!", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_SIZE, WINDOW_SIZE, SDL_WINDOW_SHOWN);    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    SDL_Surface* surface = SDL_CreateRGBSurface(0, N, N, 32, 0, 0, 0, 0);
    SDL_Event event;

    // Run simulation
    while (update_graphics(u, window, surface, renderer, event)) {
        dynamics(conv_kernel, u, v, temp_u, temp_v);
        dynamics(conv_kernel, u, v, temp_u, temp_v);
        dynamics(conv_kernel, u, v, temp_u, temp_v);
    }

    // Free matrices
    for (int i = 0; i < N; i++) {
        free(u[i]);
        free(v[i]);
        free(temp_u[i]);
        free(temp_v[i]);
    }
    free(v);
    free(u);
    free(temp_u);
    free(temp_v);

    return 0;
}
