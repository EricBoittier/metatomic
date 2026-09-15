#ifndef METATOMIC_LJ_MODEL_H
#define METATOMIC_LJ_MODEL_H

#include <stdint.h>

#include <metatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Shifted Lennard-Jones parameters used by the test plugin.
typedef struct LennardJones {
    double sigma;
    double epsilon;
    double cutoff;
    double shift;
    int32_t atomic_type;
    char length_unit[32];
    char energy_unit[32];
} LennardJones;

/// Fill ``model`` with a shifted Lennard-Jones implementation.
///
/// The model takes ownership of a heap-allocated copy of ``params``. Call
/// ``model->unload(model->data)`` when the model is no longer needed.
mta_status_t lennard_jones_fill_model(
    const LennardJones* params,
    mta_model_t* model
);

/// Parse ``options_json`` (string keys / string values) into ``params``.
///
/// Missing keys keep the defaults: ``sigma = 1``, ``epsilon = 1``,
/// ``cutoff = 3``, ``atomic_type = 1``, length unit ``Angstrom``, energy unit
/// ``eV``.
mta_status_t lennard_jones_params_from_json(
    const char* options_json,
    LennardJones* params
);

#ifdef __cplusplus
}
#endif

#endif /* METATOMIC_LJ_MODEL_H */
