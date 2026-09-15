#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <catch.hpp>
#include <metatensor.hpp>
#include <metatensor/dlpack/dlpack.h>

#include "metatomic.h"

static double lj_unshifted(double r, double sigma, double epsilon) {
    auto x = sigma / r;
    auto x6 = x * x * x * x * x * x;
    return 4.0 * epsilon * (x6 * x6 - x6);
}

static mts_block_t* pair_block(double dz) {
    auto samples = metatensor::Labels(
        {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
        {{0, 1, 0, 0, 0}}
    );
    auto components = metatensor::Labels({"xyz"}, {{0}, {1}, {2}});
    std::vector<const mts_labels_t*> components_list = {components.as_mts_labels_t()};
    auto properties = metatensor::Labels({"distance"}, {{0}});

    auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{1, 3, 1},
        std::vector<double>{0.0, 0.0, dz}
    );
    auto values_mts = metatensor::DataArrayBase::to_mts_array(std::move(values));

    auto* block = mts_block(
        std::move(values_mts).release(),
        samples.as_mts_labels_t(),
        components_list.data(),
        components_list.size(),
        properties.as_mts_labels_t()
    );
    REQUIRE(block != nullptr);
    return block;
}

static mta_system_t* two_atom_system(double distance) {
    auto types_array = std::make_unique<metatensor::SimpleDataArray<int32_t>>(
        std::vector<uintptr_t>{2}, std::vector<int32_t>{1, 1}
    );
    auto types_mts = metatensor::DataArrayBase::to_mts_array(std::move(types_array));

    auto positions_array = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{2, 3},
        std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, distance}
    );
    auto positions_mts = metatensor::DataArrayBase::to_mts_array(std::move(positions_array));

    auto cell_array = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{3, 3},
        std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    );
    auto cell_mts = metatensor::DataArrayBase::to_mts_array(std::move(cell_array));

    auto pbc_array = std::make_unique<metatensor::SimpleDataArray<uint8_t>>(
        std::vector<uintptr_t>{3}, std::vector<uint8_t>{0, 0, 0}
    );
    auto pbc_mts = metatensor::DataArrayBase::to_mts_array(std::move(pbc_array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto* pbc_dlpack = pbc_mts.as_dlpack(cpu, nullptr, version);
    pbc_dlpack->dl_tensor.dtype.code = DLDataTypeCode::kDLBool;

    mta_system_t* system = nullptr;
    auto status = mta_system_create(
        "Angstrom",
        types_mts.as_dlpack(cpu, nullptr, version),
        positions_mts.as_dlpack(cpu, nullptr, version),
        cell_mts.as_dlpack(cpu, nullptr, version),
        pbc_dlpack,
        &system
    );
    REQUIRE(status == MTA_SUCCESS);
    REQUIRE(system != nullptr);
    return system;
}

static double energy_from_output(mts_tensormap_t* output) {
    mts_block_t* block = nullptr;
    REQUIRE(mts_tensormap_block_by_id(output, &block, 0) == MTS_SUCCESS);
    REQUIRE(block != nullptr);

    struct mts_array_t array;
    std::memset(&array, 0, sizeof(array));
    REQUIRE(mts_block_data(block, &array) == MTS_SUCCESS);

    DLManagedTensorVersioned* view = nullptr;
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    REQUIRE(array.as_dlpack(array.ptr, &view, cpu, nullptr, version) == MTS_SUCCESS);
    REQUIRE(view != nullptr);

    auto* data = reinterpret_cast<double*>(
        static_cast<uint8_t*>(view->dl_tensor.data) + view->dl_tensor.byte_offset
    );
    auto energy = data[0];
    view->deleter(view);
    return energy;
}

static std::string last_error_text() {
    const char* message = nullptr;
    const char* origin = nullptr;
    mta_last_error(&message, &origin, nullptr);
    return std::string(origin != nullptr ? origin : "<no origin>")
        + ": "
        + (message != nullptr ? message : "<no message>");
}

static void require_success(mta_status_t status, const std::string& what) {
    if (status != MTA_SUCCESS) {
        FAIL(what + " failed with status " + std::to_string(status) + " (" + last_error_text() + ")");
    }
}

static std::string first_json_object(const std::string& array) {
    auto start = array.find('{');
    auto end = array.rfind('}');
    REQUIRE(start != std::string::npos);
    REQUIRE(end != std::string::npos);
    REQUIRE(end > start);
    return array.substr(start, end - start + 1);
}

static void ensure_lj_plugin() {
    static bool loaded = false;
    if (loaded) {
        return;
    }
    auto status = mta_load_plugin(LJ_PLUGIN_PATH);
    if (status != MTA_SUCCESS) {
        FAIL(
            "mta_load_plugin failed: "
            + last_error_text()
            + " (path=" + std::string(LJ_PLUGIN_PATH) + ")"
        );
    }
    loaded = true;
}

TEST_CASE("Lennard-Jones C plugin") {
    ensure_lj_plugin();

    SECTION("unknown model is not supported") {
        mta_model_t model = {};
        auto status = mta_load_model("not-lj", "{}", "lj-plugin", &model);
        CHECK(status == MTA_INVALID_PARAMETER_ERROR);
    }

    SECTION("load, execute, and check energy") {
        mta_model_t model = {};
        auto status = mta_load_model(
            "lennard-jones",
            R"({"sigma":"1.0","epsilon":"1.0","cutoff":"3.0"})",
            "lj-plugin",
            &model
        );
        require_success(status, "mta_load_model");

        mta_string_t metadata = nullptr;
        require_success(model.metadata(model.data, &metadata), "model.metadata");
        auto metadata_str = std::string(mta_string_view(metadata));
        mta_string_free(metadata);
        CHECK(metadata_str.find("Lennard-Jones") != std::string::npos);

        mta_string_t pair_lists = nullptr;
        require_success(
            model.requested_pair_lists(model.data, &pair_lists),
            "model.requested_pair_lists"
        );
        REQUIRE(pair_lists != nullptr);
        std::string options = mta_string_view(pair_lists);
        mta_string_free(pair_lists);
        // requested_pair_lists returns a JSON array; add_pairs wants one object
        auto object = first_json_object(options);

        const double distance = 1.5;
        auto* system = two_atom_system(distance);
        status = mta_system_add_pairs(system, object.c_str(), pair_block(distance));
        require_success(
            status,
            "mta_system_add_pairs with " + object
        );

        const char* requested_outputs = R"([{
            "type": "metatomic_quantity",
            "name": "energy",
            "unit": "eV",
            "gradients": ["positions"],
            "sample_kind": "system"
        }])";

        mts_tensormap_t* output = nullptr;
        status = mta_execute_model(
            model,
            &system,
            1,
            nullptr,
            requested_outputs,
            true,
            &output,
            1
        );
        require_success(status, "mta_execute_model");
        REQUIRE(output != nullptr);

        auto expected = lj_unshifted(distance, 1.0, 1.0) - lj_unshifted(3.0, 1.0, 1.0);
        CHECK(energy_from_output(output) == Approx(expected).epsilon(1e-12));

        mts_tensormap_free(output);
        mta_system_free(system);
        REQUIRE(model.unload(model.data) == MTA_SUCCESS);
    }

    SECTION("alias load_from=lj") {
        mta_model_t model = {};
        auto status = mta_load_model("lj", "{}", "lj-plugin", &model);
        REQUIRE(status == MTA_SUCCESS);
        REQUIRE(model.unload(model.data) == MTA_SUCCESS);
    }
}
