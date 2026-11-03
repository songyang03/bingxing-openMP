#include "iso3dfd.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int grid_make(Grid *g, int nx, int ny, int nz) {
    const int limit = INT_MAX - 2 * STENCIL_RADIUS;
    if (!g || nx < 1 || ny < 1 || nz < 1 || nx > limit || ny > limit || nz > limit)
        return 0;
    g->nx = nx; g->ny = ny; g->nz = nz;
    g->sx = nx + 16; g->sy = ny + 16; g->sz = nz + 16;
    if ((size_t)g->sy > SIZE_MAX / (size_t)g->sz) return 0;
    g->plane = (size_t)g->sy * (size_t)g->sz;
    if ((size_t)g->sx > SIZE_MAX / g->plane) return 0;
    g->count = (size_t)g->sx * g->plane;
    return g->count <= SIZE_MAX / sizeof(float);
}
float *field_alloc(size_t count) {
    float *p;
    if (count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "[ABORT] allocation size overflow\n"); exit(1);
    }
    p = (float *)calloc(count, sizeof(float));
    if (!p) { fprintf(stderr, "[ABORT] allocation failed\n"); exit(1); }
    return p;
}
float *field_copy(const float *source, size_t count) {
    float *p = field_alloc(count);
    memcpy(p, source, count * sizeof(float));
    return p;
}
static uint32_t random_word(uint32_t *state) {
    *state = (uint32_t)((uint64_t)*state * UINT32_C(1664525) + UINT32_C(1013904223));
    return *state;
}
static float random_signed(uint32_t *state) {
    return (float)((int)(random_word(state) >> 8) - 8388608) / 8388608.0f;
}
void initialize(const Grid *g, float *field, const char *mode, uint32_t seed) {
    uint32_t state = seed;
    const int random = strcmp(mode, "random") == 0;
    memset(field, 0, g->count * sizeof(float));
    if (random)
        for (int x = 0; x < g->nx; ++x)
            for (int y = 0; y < g->ny; ++y)
                for (int z = 0; z < g->nz; ++z)
                    field[grid_index(g,x,y,z)] = 0.1f * random_signed(&state);
    if (strcmp(mode, "impulse") == 0)
        field[grid_index(g,g->nx/2,g->ny/2,g->nz/2)] = 1.0f;
}
int halos_zero(const Grid *g, const float *a) {
    for (int x = 0; x < g->sx; ++x)
        for (int y = 0; y < g->sy; ++y)
            for (int z = 0; z < g->sz; ++z) {
                const int inside = x >= 8 && x < 8 + g->nx && y >= 8 && y < 8 + g->ny
                                && z >= 8 && z < 8 + g->nz;
                const size_t p = (size_t)x * g->plane + (size_t)y * (size_t)g->sz + (size_t)z;
                if (!inside && a[p] != 0.0f) return 0;
            }
    return 1;
}
Error compare_fields(const float *result, const float *reference, size_t count) {
    Error e = {1, 0.0, 0.0, 0};
    for (size_t i = 0; i < count; ++i) {
        double d, tolerance;
        if (!isfinite(result[i]) || !isfinite(reference[i])) {
            e.pass = 0; ++e.failures; e.max_abs = e.max_scaled = INFINITY;
            continue;
        }
        d = fabs((double)result[i] - (double)reference[i]);
        tolerance = 1e-5 + 1e-4 * fabs((double)reference[i]);
        if (d > e.max_abs) e.max_abs = d;
        if (d / tolerance > e.max_scaled) e.max_scaled = d / tolerance;
        if (d > tolerance) { e.pass = 0; ++e.failures; }
    }
    return e;
}
