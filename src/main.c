#include "verification.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _OPENMP
#error "Compile the experiment framework with OpenMP enabled."
#endif
#include <omp.h>

typedef struct {
    int nx, ny, nz, steps, threads, repeat, chunk, check_only;
    uint32_t seed;
    const char *init, *backend, *schedule;
} Options;
static void abort_config(const char *message) {
    fprintf(stderr,"[ABORT] %s\n",message); exit(1);
}
static long long integer(const char *text, long long lo, long long hi) {
    char *end;
    long long n;
    errno = 0;
    n = strtoll(text,&end,10);
    if (errno || end == text || *end != '\0' || n < lo || n > hi)
        abort_config("integer argument out of range");
    return n;
}
static void usage(void) {
    printf("Usage: iso3dfd NX NY NZ T THREADS [options]\n"
           "  --backend serial|basic|optimized   default: basic\n"
           "  --init zero|impulse|random   default: random\n"
           "  --seed UINT32               default: 123\n"
           "  --repeat R                  default: 3 measured runs\n"
           "  --schedule static|dynamic|guided   default: static\n"
           "  --chunk C                   default: 0 (runtime default chunk)\n"
           "  iso3dfd --check-only        check both parallel implementations\n"
           "  iso3dfd --help\n"
           "Correctness checks precede every measured run.\n");
}
static Options parse(int argc, char **argv) {
    Options o = {32,32,32,32,1,3,0,0,123,"random","basic","static"};
    if (argc == 2 && strcmp(argv[1],"--check-only") == 0) { o.check_only = 1; return o; }
    if (argc < 6) abort_config("expected NX NY NZ T THREADS");
    o.nx = (int)integer(argv[1],1,INT_MAX-16);
    o.ny = (int)integer(argv[2],1,INT_MAX-16);
    o.nz = (int)integer(argv[3],1,INT_MAX-16);
    o.steps = (int)integer(argv[4],0,INT_MAX);
    o.threads = (int)integer(argv[5],1,INT_MAX);
    for (int i = 6; i < argc; ++i) {
        const char *key = argv[i], *value;
        if (i + 1 >= argc) abort_config("missing option value");
        value = argv[++i];
        if (strcmp(key,"--backend") == 0) o.backend = value;
        else if (strcmp(key,"--init") == 0) o.init = value;
        else if (strcmp(key,"--seed") == 0) o.seed = (uint32_t)integer(value,0,UINT32_MAX);
        else if (strcmp(key,"--repeat") == 0) o.repeat = (int)integer(value,1,10000);
        else if (strcmp(key,"--chunk") == 0) o.chunk = (int)integer(value,0,INT_MAX);
        else if (strcmp(key,"--schedule") == 0) o.schedule = value;
        else abort_config("unknown option");
    }
    if (strcmp(o.backend,"serial") != 0 && strcmp(o.backend,"basic") != 0
        && strcmp(o.backend,"optimized") != 0)
        abort_config("backend must be serial, basic or optimized");
    if (strcmp(o.backend,"serial") == 0 && o.threads != 1)
        abort_config("serial backend requires THREADS=1");
    if (strcmp(o.init,"zero") != 0 && strcmp(o.init,"impulse") != 0 && strcmp(o.init,"random") != 0)
        abort_config("unknown initialization mode");
    if (strcmp(o.schedule,"static") != 0 && strcmp(o.schedule,"dynamic") != 0 && strcmp(o.schedule,"guided") != 0)
        abort_config("unknown scheduling policy");
    return o;
}
static int compare_times(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    Options o;
    Grid g;
    Kernel kernel;
    RunStats stats = {1};
    float *initial, *reference, *field;
    double *times, median, updates;
    size_t bytes, mid;
    if (argc == 2 && strcmp(argv[1],"--help") == 0) { usage(); return 0; }
    o = parse(argc,argv);
    if (!grid_make(&g,o.nx,o.ny,o.nz)) abort_config("invalid dimensions or grid size overflow");
    omp_set_dynamic(0);
#if _OPENMP >= 200805
    {
        const omp_sched_t policy = strcmp(o.schedule,"static") == 0 ? omp_sched_static :
                                  strcmp(o.schedule,"dynamic") == 0 ? omp_sched_dynamic : omp_sched_guided;
        omp_set_schedule(policy,o.chunk);
    }
#else
    if (strcmp(o.schedule,"static") != 0 || o.chunk != 0)
        abort_config("runtime schedule options require OpenMP 3.0; set the compute loop policy instead");
    printf("note=runtime schedule control unavailable; using compute loop policy\n");
#endif
    kernel = strcmp(o.backend,"serial") == 0 ? run_serial :
             strcmp(o.backend,"basic") == 0 ? run_parallel_basic : run_parallel_optimized;
    if (o.check_only) {
        run_validation_suite(run_parallel_basic,"basic");
        run_validation_suite(run_parallel_optimized,"optimized");
        return 0;
    }
    run_validation_suite(kernel,o.backend);
    bytes = g.count * sizeof(float);
    initial = field_alloc(g.count);
    initialize(&g,initial,o.init,o.seed);
    reference = field_copy(initial,g.count);
    run_serial(&g,reference,o.steps,1);
    field = field_alloc(g.count);
    times = (double *)malloc((size_t)o.repeat * sizeof(double));
    if (!times) abort_config("allocation failed");
    printf("backend=%s grid=%dx%dx%d steps=%d requested_threads=%d init=%s seed=%" PRIu32 "\n",
           o.backend,g.nx,g.ny,g.nz,o.steps,o.threads,o.init,o.seed);
    printf("schedule=%s chunk=%d (effective only with schedule(runtime))\n",o.schedule,o.chunk);
    for (int r = 0; r < o.repeat; ++r) {
        char label[64];
        double begin;
        /* Untimed verification also warms up the same kernel and input. */
        memcpy(field,initial,bytes);
        stats = kernel(&g,field,o.steps,o.threads);
        snprintf(label,sizeof(label),"run=%d precheck",r+1);
        verify_field(&g,field,reference,stats,o.threads,label);
        memcpy(field,initial,bytes);
        begin = omp_get_wtime();
        stats = kernel(&g,field,o.steps,o.threads);
        times[r] = omp_get_wtime() - begin;
        snprintf(label,sizeof(label),"run=%d postcheck",r+1);
        verify_field(&g,field,reference,stats,o.threads,label);
        printf("[timing] run=%d seconds=%.9g actual_threads=%d\n",r+1,times[r],stats.threads_used);
    }
    qsort(times,(size_t)o.repeat,sizeof(double),compare_times);
    mid = (size_t)o.repeat / 2;
    median = o.repeat % 2 ? times[mid] : 0.5 * (times[mid-1] + times[mid]);
    updates = (double)g.nx * (double)g.ny * (double)g.nz * (double)o.steps;
    printf("median_seconds=%.9g min_seconds=%.9g max_seconds=%.9g actual_threads=%d Mupdates_per_second=%.9g\n",
           median,times[0],times[o.repeat-1],stats.threads_used,median > 0.0 ? updates/median/1e6 : 0.0);
    if (o.steps == 0) printf("note=T0 is for functional checks only\n");
    free(initial); free(reference); free(field); free(times);
    return 0;
}
