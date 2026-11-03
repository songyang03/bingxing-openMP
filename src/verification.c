#include "verification.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int checks;
static void check(int ok, const char *label, const char *detail) {
    ++checks;
    if (!ok) {
        fflush(stdout);
        fprintf(stderr,"[validation] FAIL: %s: %s\n[ABORT] experiment stopped\n",label,detail);
        exit(2);
    }
}
/* Independent coordinate-based computation: form the Laplacian in double,
   then apply p_next = p + 0.01 * Laplacian(p) and round each step to float. */
static void oracle(const Grid *g, float *field, int steps) {
    const double c[9] = {
        3.0 * (-1077749.0 / 352800.0), 16.0 / 9.0, -14.0 / 45.0,
        112.0 / 1485.0, -7.0 / 396.0, 112.0 / 32175.0,
        -2.0 / 3861.0, 16.0 / 315315.0, -1.0 / 411840.0
    };
    float *current = field_copy(field,g->count), *next = field_alloc(g->count);
    for (int t = 0; t < steps; ++t) {
        for (int x = 0; x < g->nx; ++x)
            for (int y = 0; y < g->ny; ++y)
                for (int z = 0; z < g->nz; ++z) {
                    const size_t i = grid_index(g,x,y,z);
                    double lap = c[0] * (double)current[i];
                    for (int r = 1; r <= 8; ++r) {
                        const double sum = (double)current[grid_index(g,x,y,z+r)]
                                         + (double)current[grid_index(g,x,y,z-r)]
                                         + (double)current[grid_index(g,x,y+r,z)]
                                         + (double)current[grid_index(g,x,y-r,z)]
                                         + (double)current[grid_index(g,x+r,y,z)]
                                         + (double)current[grid_index(g,x-r,y,z)];
                        lap += c[r] * sum;
                    }
                    next[i] = (float)((double)current[i] + 0.01 * lap);
                }
        { float *tmp = current; current = next; next = tmp; }
    }
    memcpy(field,current,g->count * sizeof(float));
    free(current); free(next);
}
typedef struct {
    const char *name;
    int nx, ny, nz, steps;
    const char *init;
    uint32_t seed;
} Case;
static const Case cases[] = {
    {"T0",3,5,7,0,"random",123},
    {"zero",9,10,11,7,"zero",1},
    {"one_cell_T1",1,1,1,1,"impulse",1},
    {"one_cell_even",1,1,1,6,"random",456},
    {"thin_x",1,3,17,5,"random",12},
    {"thin_y",17,1,3,4,"random",13},
    {"thin_z",3,17,1,5,"random",14},
    {"radius8_impulse",17,19,21,1,"impulse",1},
    {"odd_steps",9,10,11,5,"random",123},
    {"even_steps",13,17,19,6,"random",UINT32_MAX},
    {"seed_zero",8,9,10,9,"random",0},
    {"longer",11,12,13,31,"random",2026}
};
static void run_case(const Case *tc, Kernel kernel, const char *name, int threads) {
    Grid g;
    float *initial, *reference, *first = NULL;
    char label[160];
    check(grid_make(&g,tc->nx,tc->ny,tc->nz),tc->name,"test grid");
    initial = field_alloc(g.count);
    initialize(&g,initial,tc->init,tc->seed);
    if (strcmp(tc->name,"radius8_impulse") == 0)
        initial[grid_index(&g,g.nx/2,g.ny/2,g.nz/2)] = 4096.0f;
    reference = field_copy(initial,g.count);
    oracle(&g,reference,tc->steps);
    snprintf(label,sizeof(label),"%s/%s/threads=%d",name,tc->name,threads);
    for (int repeat = 0; repeat < 3; ++repeat) {
        float *field = field_copy(initial,g.count);
        const RunStats stats = kernel(&g,field,tc->steps,threads);
        check(compare_fields(field,reference,g.count).pass,label,"final field");
        check(halos_zero(&g,field),label,"zero halo");
        check(stats.threads_used >= 1 && stats.threads_used <= threads,label,"team size");
        if (repeat == 0) first = field_copy(field,g.count);
        else check(compare_fields(field,first,g.count).pass,label,"repeated execution");
        free(field);
    }
    free(first); free(initial); free(reference);
}
static void analytic_tests(Kernel kernel, const char *name) {
    Grid g;
    float *field;
    const double center = 1.0 + 0.01 * 3.0 * (-1077749.0 / 352800.0);
    grid_make(&g,1,1,1);
    field = field_alloc(g.count);
    initialize(&g,field,"impulse",1);
    kernel(&g,field,5,1);
    check(fabs((double)field[grid_index(&g,0,0,0)] - pow(center,5)) < 1e-6,
          name,"one cell: p(T) = center^T");
    free(field);
    grid_make(&g,17,17,17);
    field = field_alloc(g.count);
    for (int x = 0; x < g.nx; ++x)
        for (int y = 0; y < g.ny; ++y)
            for (int z = 0; z < g.nz; ++z) field[grid_index(&g,x,y,z)] = 1.0f;
    kernel(&g,field,1,1);
    check(fabs((double)field[grid_index(&g,8,8,8)] - 1.0) < 1e-6,
          name,"constant interior: zero Laplacian at center");
    free(field);
}
static void validation_tests(Kernel kernel) {
    Grid g;
    float *field;
    const float zero[1] = {0.0f}, nan_value[1] = {NAN}, inf_value[1] = {INFINITY};
    check(!grid_make(&g,0,1,1),"arguments","zero dimension");
    check(!grid_make(&g,INT_MAX,1,1),"arguments","halo dimension overflow");
    check(!grid_make(&g,2000000000,2000000000,2000000000),"arguments","product overflow");
    grid_make(&g,1,1,1);
    field = field_alloc(g.count);
    check(kernel(&g,field,-1,1).threads_used == 0,"arguments","negative steps");
    check(kernel(&g,field,1,0).threads_used == 0,"arguments","zero threads");
    check(kernel(NULL,field,1,1).threads_used == 0,"arguments","null grid");
    check(kernel(&g,NULL,1,1).threads_used == 0,"arguments","null field");
    check(!compare_fields(nan_value,zero,1).pass,"comparison","reject NaN");
    check(!compare_fields(inf_value,zero,1).pass,"comparison","reject infinity");
    free(field);
}
int run_validation_suite(Kernel kernel, const char *name) {
    const size_t case_count = sizeof(cases)/sizeof(cases[0]);
    const int threads[] = {1,2,4};
    checks = 0;
#if defined(_OPENMP) && _OPENMP >= 200805
    omp_sched_t saved_policy;
    int saved_chunk;
    omp_get_schedule(&saved_policy,&saved_chunk);
    omp_set_schedule(omp_sched_static,0);
#endif
    for (size_t i = 0; i < case_count; ++i) run_case(&cases[i],run_serial,"serial",1);
    for (size_t n = 0; n < sizeof(threads)/sizeof(threads[0]); ++n)
        for (size_t i = 0; i < case_count; ++i) run_case(&cases[i],kernel,name,threads[n]);
#if defined(_OPENMP) && _OPENMP >= 200805
    {
        const omp_sched_t policies[] = {omp_sched_dynamic,omp_sched_guided};
        for (size_t n = 0; n < sizeof(policies)/sizeof(policies[0]); ++n) {
            omp_set_schedule(policies[n],1);
            for (size_t i = 0; i < case_count; ++i) run_case(&cases[i],kernel,"runtime_schedule",4);
        }
    }
    omp_set_schedule(saved_policy,saved_chunk);
#endif
    analytic_tests(run_serial,"serial");
    analytic_tests(kernel,name);
    validation_tests(kernel);
    printf("[validation] backend=%s suite=PASS checks=%d\n",name,checks);
    return checks;
}
void verify_field(const Grid *g, const float *field, const float *reference,
                  RunStats stats, int requested_threads, const char *context) {
    const Error e = compare_fields(field,reference,g->count);
    const int halo = halos_zero(g,field);
    const int team = stats.threads_used >= 1 && stats.threads_used <= requested_threads;
    const int pass = e.pass && halo && team;
    printf("[validation] %s result=%s max_abs=%.9g max_scaled=%.9g failed_values=%zu halo=%d team=%d\n",
           context,pass ? "PASS" : "FAIL",e.max_abs,e.max_scaled,e.failures,halo,team);
    if (!pass) {
        fflush(stdout);
        fprintf(stderr,"[ABORT] %s: experiment stopped\n",context);
        exit(2);
    }
}
