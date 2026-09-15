#include <string.h>

#include "lennard_jones.h"

static mta_status_t load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    LennardJones params;
    mta_status_t status;

    if (load_from == NULL
        || (strcmp(load_from, "lennard-jones") != 0 && strcmp(load_from, "lj") != 0))
    {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    status = lennard_jones_params_from_json(options_json, &params);
    if (status != MTA_SUCCESS) {
        return status;
    }
    return lennard_jones_fill_model(&params, model);
}

MTA_REGISTER_PLUGIN(register_plugin, {
    mta_plugin_t plugin = {
        .abi_version = MTA_ABI_VERSION,
        .name = "lj-plugin",
        .load_model = load_model,
    };
    return register_plugin(plugin);
});
