#include "iso3dfd.h"
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward Euler for diffusion: p_next = p + 0.01 * Laplacian(p).
   The time step is already included in these spatial stencil coefficients. */
static const float coefficients[9] = {
    1.0f + 0.01f * 3.0f * (-1077749.0f / 352800.0f),
    0.01f * (16.0f / 9.0f), -0.01f * (14.0f / 45.0f),
    0.01f * (112.0f / 1485.0f), -0.01f * (7.0f / 396.0f),
    0.01f * (112.0f / 32175.0f), -0.01f * (2.0f / 3861.0f),
    0.01f * (16.0f / 315315.0f), -0.01f * (1.0f / 411840.0f)
};
static inline float update_point(const Grid *g, size_t i, const float *current) {
    float value = coefficients[0] * current[i];
    for (int r = 1; r <= STENCIL_RADIUS; ++r) {
        const size_t dz = (size_t)r, dy = dz * (size_t)g->sz, dx = dz * g->plane;
        value += coefficients[r] * (current[i+dz] + current[i-dz]
                                + current[i+dy] + current[i-dy]
                                + current[i+dx] + current[i-dx]);
    }
    return value;
}
RunStats run_serial(const Grid *g, float *field, int steps, int threads) {
    RunStats stats = {1};
    float *scratch, *current, *next;
    if (!kernel_args_valid(g,field,steps,threads)) {
        stats.threads_used = 0; return stats;
    }
    if (steps == 0) return stats;
    scratch = field_alloc(g->count);
    current = field; next = scratch;
    for (int t = 0; t < steps; ++t) {
        for (int x = 0; x < g->nx; ++x)
            for (int y = 0; y < g->ny; ++y)
                for (int z = 0; z < g->nz; ++z) {
                    const size_t i = grid_index(g,x,y,z);
                    next[i] = update_point(g,i,current);
                }
        { float *tmp = current; current = next; next = tmp; }
    }
    if (current != field) memcpy(field,current,g->count * sizeof(float));
    free(scratch);
    return stats;
}
