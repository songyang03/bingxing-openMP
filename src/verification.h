#ifndef PP2026_VERIFICATION_H
#define PP2026_VERIFICATION_H
#include "iso3dfd.h"
int run_validation_suite(Kernel kernel, const char *name);
void verify_field(const Grid *g, const float *field, const float *reference,
                  RunStats stats, int requested_threads, const char *context);
#endif
