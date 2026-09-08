// Defining a Lennard-Jones model
// ==============================
//
// This tutorial shows how to implement a metatomic model in C by filling an
// :c:type:`mta_model_t` vtable and registering it through a plugin.
//
// The running example is a **shifted Lennard-Jones** pair potential. The energy
// is a sum over neighbor pairs inside a spherical cutoff
//
// .. math::
//
//     E = \sum_{i<j}^{r_{ij} < r_c} \left[
//         4 \epsilon \left(
//             \left(\frac{\sigma}{r_{ij}}\right)^{12}
//             - \left(\frac{\sigma}{r_{ij}}\right)^{6}
//         \right) - E_{\mathrm{shift}}
//     \right],
//
// with :math:`E_{\mathrm{shift}}` chosen so the pair term is exactly zero at
// :math:`r_c`. Each pair is counted once (a half neighbor list) and half of
// the pair energy is assigned to each atom.
//
// .. note::
//
//     ``execute_inner`` is left as a stub here. Returning energy TensorMaps
//     (and forces as position gradients) is the next tutorial.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

// %%
//
// Model state
// -----------
//
// The model owns its LJ parameters. ``cutoff``, ``sigma``, and ``epsilon``
// match a typical argon-scale test setup (Å and eV). ``shift`` is
// :math:`4\epsilon[(\sigma/r_c)^{12} - (\sigma/r_c)^{6}]` so the energy
// goes to zero at the cutoff.

#define LJ_CUTOFF 3.4    /* Angstrom */
#define LJ_SIGMA 1.5     /* Angstrom */
#define LJ_EPSILON 23.0  /* eV */
#define LJ_ATOMIC_TYPE 12

typedef struct {
    double cutoff;
    double sigma;
    double epsilon;
    double shift;
    int32_t atomic_type;
} LennardJonesModel;

static double lj_shift(double cutoff, double sigma, double epsilon) {
    double sigma_rc = sigma / cutoff;
    double sigma_rc_6 = sigma_rc * sigma_rc * sigma_rc;
    sigma_rc_6 *= sigma_rc_6;
    return 4.0 * epsilon * (sigma_rc_6 * sigma_rc_6 - sigma_rc_6);
}

// Pair-list cutoffs in JSON use the IEEE-754 bit pattern as a hex string
// (see :ref:`core-json-pair-options`), so the engine sees the exact ``double``.
static void format_f64_hex(double value, char* buf, size_t n) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    snprintf(buf, n, "0x%" PRIx64, bits);
}

// %%
//
// Metadata callbacks
// ------------------
//
// Each callback writes a JSON document matching the schemas in
// :ref:`core-json-formats`. Prefer the typed forms
// (``"type": "metatomic_..."``) over older field names.

static mta_status_t lj_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

static mta_status_t lj_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"lennard-jones\","
        "\"authors\": [\"metatomic C tutorials\"],"
        "\"description\": \"Minimal shifted Lennard-Jones potential for engine integration tests\","
        "\"references\": {"
        "  \"model\": [],"
        "  \"architecture\": [],"
        "  \"implementation\": []"
        "},"
        "\"extra\": {\"potential\": \"shifted-lennard-jones\"}"
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_capabilities(const void* model_data, mta_string_t* out) {
    const LennardJonesModel* lj = (const LennardJonesModel*)model_data;
    char json[1024];
    snprintf(
        json,
        sizeof(json),
        "{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [{"
        "  \"type\": \"metatomic_quantity\","
        "  \"name\": \"energy\","
        "  \"unit\": \"eV\","
        "  \"gradients\": [\"positions\"],"
        "  \"sample_kind\": \"system\""
        "}],"
        "\"atomic_types\": [%d],"
        "\"interaction_range\": %.17g,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
        "}",
        lj->atomic_type,
        lj->cutoff
    );
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_supported_outputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "[{"
        "\"type\": \"metatomic_quantity\","
        "\"name\": \"energy\","
        "\"unit\": \"eV\","
        "\"gradients\": [\"positions\"],"
        "\"sample_kind\": \"system\""
        "}]"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_requested_pair_lists(const void* model_data, mta_string_t* out) {
    const LennardJonesModel* lj = (const LennardJonesModel*)model_data;
    char cutoff_hex[32];
    char json[512];
    format_f64_hex(lj->cutoff, cutoff_hex, sizeof(cutoff_hex));
    // Half list (each pair once), strict cutoff — same NeighborListOptions
    // as the pure-PyTorch Lennard-Jones test model.
    snprintf(
        json,
        sizeof(json),
        "[{"
        "\"type\": \"metatomic_pair_options\","
        "\"cutoff\": \"%s\","
        "\"full_list\": false,"
        "\"strict\": true,"
        "\"requestors\": [\"lennard-jones\"]"
        "}]",
        cutoff_hex
    );
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_requested_inputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t lj_execute_inner(
    void* model_data,
    const mta_system_t* const* systems,
    uintptr_t systems_count,
    const mts_labels_t* selected_atoms,
    const char* requested_outputs_json,
    mts_tensormap_t** outputs,
    uintptr_t outputs_count
) {
    (void)model_data;
    (void)systems;
    (void)systems_count;
    (void)selected_atoms;
    (void)requested_outputs_json;
    (void)outputs;
    (void)outputs_count;
    // Pair energy: 4*epsilon*((sigma/r)^12 - (sigma/r)^6) - shift, then
    // split half onto each atom. Forces are -dE/dr via positions gradients.
    // Filling the output TensorMaps is deferred until the C TensorMap helpers
    // in this tutorial series land.
    mta_set_last_error(
        "lennard-jones execute_inner is not implemented yet (WIP)",
        "lj_execute_inner",
        NULL,
        NULL
    );
    return MTA_INTERNAL_ERROR;
}

// %%
//
// Plugin registration
// -------------------
//
// Models are produced by plugins. For a single-file tutorial we register the
// plugin in-process with :c:func:`mta_register_plugin`. Shared-library plugins
// use the :c:macro:`MTA_REGISTER_PLUGIN` macro instead (see the next tutorial).

static mta_status_t lj_load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    (void)options_json;
    if (strcmp(load_from, "lennard-jones") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    LennardJonesModel* data = malloc(sizeof(LennardJonesModel));
    if (data == NULL) {
        mta_set_last_error("out of memory", "lj_load_model", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }
    data->cutoff = LJ_CUTOFF;
    data->sigma = LJ_SIGMA;
    data->epsilon = LJ_EPSILON;
    data->shift = lj_shift(LJ_CUTOFF, LJ_SIGMA, LJ_EPSILON);
    data->atomic_type = LJ_ATOMIC_TYPE;

    model->data = data;
    model->unload = lj_unload;
    model->metadata = lj_metadata;
    model->capabilities = lj_capabilities;
    model->supported_outputs = lj_supported_outputs;
    model->requested_pair_lists = lj_requested_pair_lists;
    model->requested_inputs = lj_requested_inputs;
    model->execute_inner = lj_execute_inner;
    return MTA_SUCCESS;
}

static int fail(mta_model_t* model, const char* what) {
    const char* message = NULL;
    mta_last_error(&message, NULL, NULL);
    fprintf(stderr, "%s: %s\n", what, message != NULL ? message : "(no message)");
    if (model != NULL && model->unload != NULL && model->data != NULL) {
        model->unload(model->data);
    }
    return EXIT_FAILURE;
}

// %%

int main(void) {
    static mta_plugin_t PLUGIN = {
        .abi_version = MTA_ABI_VERSION,
        .name = "tutorial-lj-plugin",
        .load_model = lj_load_model,
    };
    if (mta_register_plugin(PLUGIN) != MTA_SUCCESS) {
        return fail(NULL, "failed to register plugin");
    }

    mta_model_t model = {0};
    if (mta_load_model("lennard-jones", "{}", "tutorial-lj-plugin", &model)
        != MTA_SUCCESS) {
        return fail(NULL, "failed to load model");
    }

    mta_string_t metadata = NULL;
    if (model.metadata(model.data, &metadata) != MTA_SUCCESS) {
        return fail(&model, "failed to get metadata");
    }
    mta_string_t printed = NULL;
    if (mta_format_metadata(mta_string_view(metadata), &printed) != MTA_SUCCESS) {
        mta_string_free(metadata);
        return fail(&model, "failed to format metadata");
    }
    printf("%s\n", mta_string_view(printed));
    if (strstr(mta_string_view(printed), "lennard-jones") == NULL) {
        fprintf(stderr, "formatted metadata missing model name\n");
        mta_string_free(metadata);
        mta_string_free(printed);
        model.unload(model.data);
        return EXIT_FAILURE;
    }
    mta_string_free(metadata);
    mta_string_free(printed);

    mta_string_t pairs = NULL;
    if (model.requested_pair_lists(model.data, &pairs) != MTA_SUCCESS) {
        return fail(&model, "failed to get requested pair lists");
    }
    const char* pairs_json = mta_string_view(pairs);
    printf("requested pair lists: %s\n", pairs_json);
    if (strstr(pairs_json, "0x400b333333333333") == NULL
        || strstr(pairs_json, "\"full_list\": false") == NULL
        || strstr(pairs_json, "\"strict\": true") == NULL) {
        fprintf(stderr, "unexpected pair-list request: %s\n", pairs_json);
        mta_string_free(pairs);
        model.unload(model.data);
        return EXIT_FAILURE;
    }
    mta_string_free(pairs);

    printf("execute_inner is WIP; energy TensorMaps will be added later\n");

    model.unload(model.data);
    return EXIT_SUCCESS;
}

// %%
//
// Expected output
// ---------------
//
// ::
//
//     This is the lennard-jones model
//     ===============================
//
//     Minimal shifted Lennard-Jones potential for engine integration tests
//
//     Model authors
//     -------------
//
//     - metatomic C tutorials
//
//     requested pair lists: [{"type": "metatomic_pair_options","cutoff": "0x400b333333333333","full_list": false,"strict": true,"requestors": ["lennard-jones"]}]
//
//     execute_inner is WIP; energy TensorMaps will be added later
