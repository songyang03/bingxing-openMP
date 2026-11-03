#ifndef PP2026_ISO3DFD_H
#define PP2026_ISO3DFD_H
#include <stddef.h>
#include <stdint.h>

#define STENCIL_RADIUS 8

typedef struct {
    int nx, ny, nz, sx, sy, sz;
    size_t plane, count;
} Grid;

typedef struct { int threads_used; } RunStats;
typedef RunStats (*Kernel)(const Grid *, float *, int, int);

int grid_make(Grid *g, int nx, int ny, int nz);
static inline size_t grid_index(const Grid *g, int x, int y, int z) {
    return (size_t)(x + STENCIL_RADIUS) * g->plane
         + (size_t)(y + STENCIL_RADIUS) * (size_t)g->sz + (size_t)(z + STENCIL_RADIUS);
}
static inline int kernel_args_valid(const Grid *g, const float *field, int steps, int threads) {
    return g && field && steps >= 0 && threads > 0;
}

/* field is input u^0 and output u^T; T=0 leaves it unchanged. */
RunStats run_serial(const Grid *g, float *field, int steps, int threads);
RunStats run_parallel_basic(const Grid *g, float *field, int steps, int threads);
RunStats run_parallel_optimized(const Grid *g, float *field, int steps, int threads);

/* The external framework owns allocation and data initialization. */
float *field_alloc(size_t count);
float *field_copy(const float *source, size_t count);
void initialize(const Grid *g, float *field, const char *mode, uint32_t seed);
int halos_zero(const Grid *g, const float *field);

typedef struct {
    int pass;
    double max_abs, max_scaled;
    size_t failures;
} Error;
Error compare_fields(const float *result, const float *reference, size_t count);
#endif
