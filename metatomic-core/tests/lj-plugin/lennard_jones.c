#include "lennard_jones.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor/dlpack/dlpack.h>

/* -------------------------------------------------------------------------- */
/* mts_array_t helpers (CPU, owned buffers)                                   */
/* -------------------------------------------------------------------------- */

typedef struct BasicMtsArray {
    void* data;
    uintptr_t ndim;
    uintptr_t shape[4];
    DLDataType dtype;
} BasicMtsArray;

static mts_data_origin_t BASIC_MTS_ARRAY_ORIGIN = 0;

static void array_destroy(void* array) {
    BasicMtsArray* a = (BasicMtsArray*)array;
    if (a == NULL) {
        return;
    }
    free(a->data);
    free(a);
}

static mts_status_t array_origin(const void* array, mts_data_origin_t* origin) {
    (void)array;
    if (BASIC_MTS_ARRAY_ORIGIN == 0) {
        mts_register_data_origin("lj-plugin-mts-array", &BASIC_MTS_ARRAY_ORIGIN);
    }
    *origin = BASIC_MTS_ARRAY_ORIGIN;
    return MTS_SUCCESS;
}

static mts_status_t array_device(const void* array, DLDevice* device) {
    (void)array;
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t array_dtype(const void* array, DLDataType* dtype) {
    *dtype = ((const BasicMtsArray*)array)->dtype;
    return MTS_SUCCESS;
}

typedef struct CustomDLPackContext {
    int64_t* shape;
    int64_t* strides;
} CustomDLPackContext;

static void dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self == NULL) {
        return;
    }
    CustomDLPackContext* ctx = (CustomDLPackContext*)self->manager_ctx;
    if (ctx != NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
    }
    free(self);
}

static DLManagedTensorVersioned* tensor_from_data(
    void* data,
    int32_t ndim,
    const int64_t* shape,
    DLDataType dtype
) {
    CustomDLPackContext* ctx = calloc(1, sizeof(CustomDLPackContext));
    DLManagedTensorVersioned* tensor = NULL;
    int64_t stride = 1;
    int32_t i = 0;

    if (ctx == NULL) {
        return NULL;
    }
    ctx->shape = malloc((size_t)ndim * sizeof(int64_t));
    ctx->strides = malloc((size_t)ndim * sizeof(int64_t));
    if (ctx->shape == NULL || ctx->strides == NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    memcpy(ctx->shape, shape, (size_t)ndim * sizeof(int64_t));
    for (i = ndim - 1; i >= 0; i--) {
        ctx->strides[i] = stride;
        stride *= shape[i];
    }

    tensor = calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
        return NULL;
    }
    tensor->version.major = DLPACK_MAJOR_VERSION;
    tensor->version.minor = DLPACK_MINOR_VERSION;
    tensor->manager_ctx = ctx;
    tensor->deleter = dlpack_deleter;
    tensor->flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    tensor->dl_tensor.data = data;
    tensor->dl_tensor.byte_offset = 0;
    tensor->dl_tensor.device.device_type = kDLCPU;
    tensor->dl_tensor.device.device_id = 0;
    tensor->dl_tensor.dtype = dtype;
    tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.shape = ctx->shape;
    tensor->dl_tensor.strides = ctx->strides;
    return tensor;
}

static mts_status_t array_as_dlpack(
    void* array,
    DLManagedTensorVersioned** tensor,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    BasicMtsArray* a = (BasicMtsArray*)array;
    int64_t shape[4];
    uintptr_t i = 0;
    (void)stream;
    (void)max_version;
    if (device.device_type != kDLCPU) {
        return MTS_CALLBACK_ERROR;
    }
    for (i = 0; i < a->ndim; i++) {
        shape[i] = (int64_t)a->shape[i];
    }
    *tensor = tensor_from_data(a->data, (int32_t)a->ndim, shape, a->dtype);
    return (*tensor != NULL) ? MTS_SUCCESS : MTS_CALLBACK_ERROR;
}

static mts_status_t array_shape(
    const void* array,
    const uintptr_t** shape,
    uintptr_t* shape_count
) {
    const BasicMtsArray* a = (const BasicMtsArray*)array;
    *shape = a->shape;
    *shape_count = a->ndim;
    return MTS_SUCCESS;
}

static int make_mts_array(
    const void* data,
    uintptr_t ndim,
    const uintptr_t* shape,
    DLDataType dtype,
    uintptr_t n_elements,
    struct mts_array_t* result
) {
    BasicMtsArray* raw = calloc(1, sizeof(BasicMtsArray));
    size_t data_size = n_elements * (size_t)(dtype.bits / 8);
    uintptr_t i = 0;

    if (raw == NULL) {
        return 0;
    }
    raw->data = malloc(data_size);
    if (raw->data == NULL) {
        free(raw);
        return 0;
    }
    memcpy(raw->data, data, data_size);
    raw->ndim = ndim;
    for (i = 0; i < ndim; i++) {
        raw->shape[i] = shape[i];
    }
    raw->dtype = dtype;

    memset(result, 0, sizeof(*result));
    result->ptr = raw;
    result->destroy = array_destroy;
    result->origin = array_origin;
    result->device = array_device;
    result->dtype = array_dtype;
    result->as_dlpack = array_as_dlpack;
    result->shape = array_shape;
    return 1;
}

static void fail(const char* message, const char* origin) {
    mta_set_last_error(message, origin, NULL, NULL);
}

/* -------------------------------------------------------------------------- */
/* JSON helpers: options are string keys / string values                      */
/* -------------------------------------------------------------------------- */

static int json_string_option(
    const char* json,
    const char* key,
    char* out,
    size_t out_n
) {
    char pattern[64];
    const char* p = NULL;
    size_t i = 0;

    if (json == NULL || key == NULL || out == NULL || out_n == 0) {
        return 0;
    }
    if (snprintf(pattern, sizeof(pattern), "\"%s\"", key) >= (int)sizeof(pattern)) {
        return 0;
    }
    p = strstr(json, pattern);
    if (p == NULL) {
        return 0;
    }
    p = strchr(p + strlen(pattern), ':');
    if (p == NULL) {
        return 0;
    }
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
        p++;
    }
    if (*p != '"') {
        return 0;
    }
    p++;
    while (p[i] != '\0' && p[i] != '"' && i + 1 < out_n) {
        out[i] = p[i];
        i++;
    }
    out[i] = '\0';
    return 1;
}

static int parse_double(const char* text, double* out) {
    char* end = NULL;
    double value = 0.0;
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || (end != NULL && *end != '\0')) {
        return 0;
    }
    *out = value;
    return 1;
}

static double lj_shift(double cutoff, double sigma, double epsilon) {
    double x = sigma / cutoff;
    double x6 = x * x * x * x * x * x;
    return 4.0 * epsilon * (x6 * x6 - x6);
}

mta_status_t lennard_jones_params_from_json(
    const char* options_json,
    LennardJones* params
) {
    char buffer[64];

    if (params == NULL) {
        fail("params is NULL", "lennard_jones_params_from_json");
        return MTA_INVALID_PARAMETER_ERROR;
    }

    params->sigma = 1.0;
    params->epsilon = 1.0;
    params->cutoff = 3.0;
    params->atomic_type = 1;
    strncpy(params->length_unit, "Angstrom", sizeof(params->length_unit) - 1);
    strncpy(params->energy_unit, "eV", sizeof(params->energy_unit) - 1);
    params->length_unit[sizeof(params->length_unit) - 1] = '\0';
    params->energy_unit[sizeof(params->energy_unit) - 1] = '\0';

    if (options_json != NULL && options_json[0] != '\0') {
        if (json_string_option(options_json, "sigma", buffer, sizeof(buffer))) {
            if (!parse_double(buffer, &params->sigma)) {
                fail("option 'sigma' must be a number", "lennard_jones_params_from_json");
                return MTA_INVALID_PARAMETER_ERROR;
            }
        }
        if (json_string_option(options_json, "epsilon", buffer, sizeof(buffer))) {
            if (!parse_double(buffer, &params->epsilon)) {
                fail("option 'epsilon' must be a number", "lennard_jones_params_from_json");
                return MTA_INVALID_PARAMETER_ERROR;
            }
        }
        if (json_string_option(options_json, "cutoff", buffer, sizeof(buffer))) {
            if (!parse_double(buffer, &params->cutoff)) {
                fail("option 'cutoff' must be a number", "lennard_jones_params_from_json");
                return MTA_INVALID_PARAMETER_ERROR;
            }
        }
        if (json_string_option(options_json, "atomic_type", buffer, sizeof(buffer))) {
            char* end = NULL;
            long value = strtol(buffer, &end, 10);
            if (end == buffer || *end != '\0') {
                fail(
                    "option 'atomic_type' must be an integer",
                    "lennard_jones_params_from_json"
                );
                return MTA_INVALID_PARAMETER_ERROR;
            }
            params->atomic_type = (int32_t)value;
        }
        if (json_string_option(options_json, "length_unit", buffer, sizeof(buffer))) {
            strncpy(params->length_unit, buffer, sizeof(params->length_unit) - 1);
            params->length_unit[sizeof(params->length_unit) - 1] = '\0';
        }
        if (json_string_option(options_json, "energy_unit", buffer, sizeof(buffer))) {
            strncpy(params->energy_unit, buffer, sizeof(params->energy_unit) - 1);
            params->energy_unit[sizeof(params->energy_unit) - 1] = '\0';
        }
    }

    if (!isfinite(params->sigma) || params->sigma <= 0.0
        || !isfinite(params->epsilon) || params->epsilon <= 0.0
        || !isfinite(params->cutoff) || params->cutoff <= 0.0)
    {
        fail(
            "sigma, epsilon and cutoff must be finite and positive",
            "lennard_jones_params_from_json"
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }

    params->shift = lj_shift(params->cutoff, params->sigma, params->epsilon);
    return MTA_SUCCESS;
}

static void lj_pair(
    double dx,
    double dy,
    double dz,
    const LennardJones* lj,
    double* energy,
    double force_on_first[3]
) {
    double r2 = dx * dx + dy * dy + dz * dz;
    double inv2 = 0.0;
    double inv6 = 0.0;
    double inv12 = 0.0;
    double dedr2 = 0.0;

    if (r2 <= 0.0 || r2 >= lj->cutoff * lj->cutoff) {
        *energy = 0.0;
        force_on_first[0] = 0.0;
        force_on_first[1] = 0.0;
        force_on_first[2] = 0.0;
        return;
    }

    inv2 = (lj->sigma * lj->sigma) / r2;
    inv6 = inv2 * inv2 * inv2;
    inv12 = inv6 * inv6;
    *energy = 4.0 * lj->epsilon * (inv12 - inv6) - lj->shift;

    /* dE/d(r^2) = (12 epsilon / r^2) (inv6 - 2 inv12)
     * r^2 = |pos_2 - pos_1|^2, so d(r^2)/d(pos_1) = -2 d
     * force on atom 1 is -dE/d(pos_1) = 2 dE/d(r^2) d */
    dedr2 = (12.0 * lj->epsilon / r2) * (inv6 - 2.0 * inv12);
    force_on_first[0] = 2.0 * dedr2 * dx;
    force_on_first[1] = 2.0 * dedr2 * dy;
    force_on_first[2] = 2.0 * dedr2 * dz;
}

static int format_pair_options(const LennardJones* lj, char* buf, size_t n) {
    uint64_t bits = 0;
    char cutoff_hex[32];
    memcpy(&bits, &lj->cutoff, sizeof(bits));
    snprintf(cutoff_hex, sizeof(cutoff_hex), "0x%" PRIx64, bits);
    return snprintf(
        buf,
        n,
        "{\"type\":\"metatomic_pair_list_options\",\"cutoff\":\"%s\","
        "\"full_list\":false,\"strict\":true,\"requestors\":[\"lj-plugin\"]}",
        cutoff_hex
    );
}

static mta_status_t lj_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

static mta_status_t lj_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\":\"metatomic_model_metadata\","
        "\"name\":\"Lennard-Jones\","
        "\"authors\":[\"metatomic\"],"
        "\"description\":\"Shifted Lennard-Jones pair potential for engine tests\","
        "\"references\":{"
        "  \"model\":[\"https://github.com/metatensor/lj-test\"],"
        "  \"architecture\":[],"
        "  \"implementation\":[\"https://github.com/metatensor/metatomic\"]"
        "},"
        "\"extra\":{}"
        "}"
    );
    if (*out == NULL) {
        fail("failed to allocate metadata JSON", "lj_metadata");
        return MTA_INTERNAL_ERROR;
    }
    return MTA_SUCCESS;
}

static mta_status_t lj_capabilities(const void* model_data, mta_string_t* out) {
    const LennardJones* lj = (const LennardJones*)model_data;
    char json[1024];
    int written = snprintf(
        json,
        sizeof(json),
        "{"
        "\"type\":\"metatomic_model_capabilities\","
        "\"outputs\":[{"
        "  \"type\":\"metatomic_quantity\","
        "  \"name\":\"energy\","
        "  \"unit\":\"%s\","
        "  \"gradients\":[\"positions\"],"
        "  \"sample_kind\":\"system\""
        "}],"
        "\"atomic_types\":[%" PRId32 "],"
        "\"interaction_range\":%.17g,"
        "\"length_unit\":\"%s\","
        "\"supported_devices\":[\"cpu\"],"
        "\"dtype\":\"float64\""
        "}",
        lj->energy_unit,
        lj->atomic_type,
        lj->cutoff,
        lj->length_unit
    );
    if (written < 0 || (size_t)written >= sizeof(json)) {
        fail("failed to format capabilities JSON", "lj_capabilities");
        return MTA_INTERNAL_ERROR;
    }
    *out = mta_string_create(json);
    if (*out == NULL) {
        fail("failed to allocate capabilities JSON", "lj_capabilities");
        return MTA_INTERNAL_ERROR;
    }
    return MTA_SUCCESS;
}

static mta_status_t lj_requested_pair_lists(const void* model_data, mta_string_t* out) {
    char options[512];
    char json[520];
    if (format_pair_options((const LennardJones*)model_data, options, sizeof(options)) < 0) {
        fail("failed to format pair-list options", "lj_requested_pair_lists");
        return MTA_INTERNAL_ERROR;
    }
    snprintf(json, sizeof(json), "[%s]", options);
    *out = mta_string_create(json);
    if (*out == NULL) {
        fail("failed to allocate pair-list JSON", "lj_requested_pair_lists");
        return MTA_INTERNAL_ERROR;
    }
    return MTA_SUCCESS;
}

static mta_status_t lj_requested_inputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");
    if (*out == NULL) {
        fail("failed to allocate requested inputs JSON", "lj_requested_inputs");
        return MTA_INTERNAL_ERROR;
    }
    return MTA_SUCCESS;
}

static mts_block_t* positions_gradient(const double* forces, uintptr_t n_atoms) {
    int32_t* sample_values = malloc(sizeof(int32_t) * n_atoms * 3);
    struct mts_array_t samples_array;
    struct mts_array_t xyz_array;
    struct mts_array_t prop_array;
    struct mts_array_t values;
    const mts_labels_t* samples = NULL;
    const mts_labels_t* xyz = NULL;
    const mts_labels_t* properties = NULL;
    const mts_labels_t* components[1];
    mts_block_t* gradient = NULL;
    double* data = NULL;
    uintptr_t j = 0;
    int32_t xyz_values[3] = {0, 1, 2};
    int32_t energy_value = 0;
    uintptr_t sample_shape[2];
    uintptr_t xyz_shape[2];
    uintptr_t prop_shape[2];
    uintptr_t values_shape[3];
    const char* sample_dims[3] = {"sample", "system", "atom"};
    const char* xyz_dims[1] = {"xyz"};
    const char* energy_dims[1] = {"energy"};
    DLDataType i32 = {.code = kDLInt, .bits = 32, .lanes = 1};
    DLDataType f64 = {.code = kDLFloat, .bits = 64, .lanes = 1};

    if (sample_values == NULL) {
        return NULL;
    }
    for (j = 0; j < n_atoms; j++) {
        sample_values[3 * j + 0] = 0;
        sample_values[3 * j + 1] = 0;
        sample_values[3 * j + 2] = (int32_t)j;
    }
    sample_shape[0] = n_atoms;
    sample_shape[1] = 3;
    if (!make_mts_array(sample_values, 2, sample_shape, i32, n_atoms * 3, &samples_array)) {
        free(sample_values);
        return NULL;
    }
    free(sample_values);
    samples = mts_labels(sample_dims, 3, samples_array);

    xyz_shape[0] = 3;
    xyz_shape[1] = 1;
    if (!make_mts_array(xyz_values, 2, xyz_shape, i32, 3, &xyz_array)) {
        mts_labels_free(samples);
        return NULL;
    }
    xyz = mts_labels(xyz_dims, 1, xyz_array);
    components[0] = xyz;

    prop_shape[0] = 1;
    prop_shape[1] = 1;
    if (!make_mts_array(&energy_value, 2, prop_shape, i32, 1, &prop_array)) {
        mts_labels_free(samples);
        mts_labels_free(xyz);
        return NULL;
    }
    properties = mts_labels(energy_dims, 1, prop_array);

    data = malloc(sizeof(double) * n_atoms * 3);
    if (data == NULL) {
        mts_labels_free(samples);
        mts_labels_free(xyz);
        mts_labels_free(properties);
        return NULL;
    }
    for (j = 0; j < n_atoms; j++) {
        data[j * 3 + 0] = -forces[j * 3 + 0];
        data[j * 3 + 1] = -forces[j * 3 + 1];
        data[j * 3 + 2] = -forces[j * 3 + 2];
    }
    values_shape[0] = n_atoms;
    values_shape[1] = 3;
    values_shape[2] = 1;
    if (!make_mts_array(data, 3, values_shape, f64, n_atoms * 3, &values)) {
        free(data);
        mts_labels_free(samples);
        mts_labels_free(xyz);
        mts_labels_free(properties);
        return NULL;
    }
    free(data);

    gradient = mts_block(values, samples, components, 1, properties);
    mts_labels_free(samples);
    mts_labels_free(xyz);
    mts_labels_free(properties);
    return gradient;
}

static mts_tensormap_t* energy_tensormap(
    double energy,
    const double* forces,
    uintptr_t n_atoms
) {
    struct mts_array_t values;
    struct mts_array_t samples_array;
    struct mts_array_t prop_array;
    struct mts_array_t key_array;
    const mts_labels_t* samples = NULL;
    const mts_labels_t* properties = NULL;
    const mts_labels_t* keys = NULL;
    mts_block_t* block = NULL;
    mts_block_t* gradient = NULL;
    mts_block_t* blocks[1];
    mts_tensormap_t* tensor = NULL;
    int32_t zero = 0;
    uintptr_t shape2[2] = {1, 1};
    const char* system_dims[1] = {"system"};
    const char* energy_dims[1] = {"energy"};
    const char* key_dims[1] = {"_"};
    DLDataType f64 = {.code = kDLFloat, .bits = 64, .lanes = 1};
    DLDataType i32 = {.code = kDLInt, .bits = 32, .lanes = 1};

    if (!make_mts_array(&energy, 2, shape2, f64, 1, &values)) {
        return NULL;
    }
    if (!make_mts_array(&zero, 2, shape2, i32, 1, &samples_array)) {
        values.destroy(values.ptr);
        return NULL;
    }
    samples = mts_labels(system_dims, 1, samples_array);
    if (!make_mts_array(&zero, 2, shape2, i32, 1, &prop_array)) {
        mts_labels_free(samples);
        values.destroy(values.ptr);
        return NULL;
    }
    properties = mts_labels(energy_dims, 1, prop_array);
    if (!make_mts_array(&zero, 2, shape2, i32, 1, &key_array)) {
        mts_labels_free(samples);
        mts_labels_free(properties);
        values.destroy(values.ptr);
        return NULL;
    }
    keys = mts_labels(key_dims, 1, key_array);

    block = mts_block(values, samples, NULL, 0, properties);
    if (block == NULL) {
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        return NULL;
    }

    gradient = positions_gradient(forces, n_atoms);
    if (gradient == NULL || mts_block_add_gradient(block, "positions", gradient) != MTS_SUCCESS) {
        mts_block_free(block);
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        return NULL;
    }

    blocks[0] = block;
    tensor = mts_tensormap(keys, blocks, 1);
    mts_labels_free(samples);
    mts_labels_free(properties);
    mts_labels_free(keys);
    return tensor;
}

static DLManagedTensorVersioned* block_f64_view(const mts_block_t* block) {
    struct mts_array_t array;
    DLManagedTensorVersioned* view = NULL;
    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {.major = DLPACK_MAJOR_VERSION, .minor = DLPACK_MINOR_VERSION};
    memset(&array, 0, sizeof(array));
    if (mts_block_data((mts_block_t*)block, &array) != MTS_SUCCESS) {
        return NULL;
    }
    if (array.as_dlpack(array.ptr, &view, cpu, NULL, version) != MTS_SUCCESS) {
        return NULL;
    }
    return view;
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
    const LennardJones* lj = (const LennardJones*)model_data;
    const mta_system_t* system = NULL;
    uintptr_t n_atoms = 0;
    char options[512];
    const mts_block_t* pairs = NULL;
    DLManagedTensorVersioned* disp_view = NULL;
    double* disp = NULL;
    uintptr_t n_pairs = 0;
    const mts_labels_t* pair_samples = NULL;
    const int32_t* sample_values = NULL;
    uintptr_t sample_count = 0;
    uintptr_t sample_size = 0;
    double* forces = NULL;
    double energy = 0.0;
    uintptr_t p = 0;
    uintptr_t idx = 0;
    mta_status_t status = MTA_SUCCESS;

    (void)selected_atoms;
    (void)requested_outputs_json;

    if (systems_count != 1) {
        fail("lj-plugin currently supports exactly one system", "lj_execute_inner");
        return MTA_INVALID_PARAMETER_ERROR;
    }

    system = systems[0];
    status = mta_system_size(system, &n_atoms);
    if (status != MTA_SUCCESS) {
        return status;
    }

    if (format_pair_options(lj, options, sizeof(options)) < 0) {
        fail("failed to format pair-list options", "lj_execute_inner");
        return MTA_INTERNAL_ERROR;
    }
    status = mta_system_get_pairs(system, options, &pairs);
    if (status != MTA_SUCCESS) {
        return status;
    }
    if (pairs == NULL) {
        fail("missing pair list for Lennard-Jones cutoff", "lj_execute_inner");
        return MTA_INVALID_PARAMETER_ERROR;
    }

    disp_view = block_f64_view(pairs);
    if (disp_view == NULL) {
        fail("failed to read pair displacements", "lj_execute_inner");
        return MTA_INTERNAL_ERROR;
    }
    disp = (double*)((uint8_t*)disp_view->dl_tensor.data + disp_view->dl_tensor.byte_offset);
    n_pairs = (uintptr_t)disp_view->dl_tensor.shape[0];

    pair_samples = mts_block_labels(pairs, 0);
    if (mts_labels_values_cpu(pair_samples, &sample_values, &sample_count, &sample_size)
        != MTS_SUCCESS)
    {
        mts_labels_free(pair_samples);
        disp_view->deleter(disp_view);
        fail("failed to read pair samples", "lj_execute_inner");
        return MTA_METATENSOR_ERROR;
    }

    forces = calloc(n_atoms * 3, sizeof(double));
    if (forces == NULL) {
        mts_labels_free(pair_samples);
        disp_view->deleter(disp_view);
        fail("out of memory", "lj_execute_inner");
        return MTA_INTERNAL_ERROR;
    }

    for (p = 0; p < n_pairs; p++) {
        double pair_energy = 0.0;
        double force_on_first[3];
        int32_t i = 0;
        int32_t j = 0;
        int k = 0;
        lj_pair(
            disp[3 * p + 0],
            disp[3 * p + 1],
            disp[3 * p + 2],
            lj,
            &pair_energy,
            force_on_first
        );
        energy += pair_energy;
        i = sample_values[p * sample_size + 0];
        j = sample_values[p * sample_size + 1];
        for (k = 0; k < 3; k++) {
            forces[(uintptr_t)i * 3 + (uintptr_t)k] += force_on_first[k];
            forces[(uintptr_t)j * 3 + (uintptr_t)k] -= force_on_first[k];
        }
    }

    mts_labels_free(pair_samples);
    disp_view->deleter(disp_view);

    for (idx = 0; idx < outputs_count; idx++) {
        outputs[idx] = energy_tensormap(energy, forces, n_atoms);
        if (outputs[idx] == NULL) {
            uintptr_t k = 0;
            for (k = 0; k < idx; k++) {
                mts_tensormap_free(outputs[k]);
                outputs[k] = NULL;
            }
            free(forces);
            fail("failed to build energy TensorMap", "lj_execute_inner");
            return MTA_INTERNAL_ERROR;
        }
    }
    free(forces);
    return MTA_SUCCESS;
}

mta_status_t lennard_jones_fill_model(const LennardJones* params, mta_model_t* model) {
    LennardJones* data = NULL;
    if (params == NULL || model == NULL) {
        fail("NULL pointer", "lennard_jones_fill_model");
        return MTA_INVALID_PARAMETER_ERROR;
    }
    data = malloc(sizeof(LennardJones));
    if (data == NULL) {
        fail("out of memory", "lennard_jones_fill_model");
        return MTA_INTERNAL_ERROR;
    }
    *data = *params;
    memset(model, 0, sizeof(*model));
    model->data = data;
    model->unload = lj_unload;
    model->metadata = lj_metadata;
    model->capabilities = lj_capabilities;
    model->requested_pair_lists = lj_requested_pair_lists;
    model->requested_inputs = lj_requested_inputs;
    model->execute_inner = lj_execute_inner;
    return MTA_SUCCESS;
}
