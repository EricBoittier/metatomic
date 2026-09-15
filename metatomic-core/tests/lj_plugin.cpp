#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <catch.hpp>
#include <metatomic.hpp>
#include <metatensor.hpp>
#include <nlohmann/json.hpp>


template <typename T>
static metatomic::DLPackTensor dlpack_tensor(
    std::vector<uintptr_t> shape,
    std::vector<typename metatensor::SimpleDataArray<T>::storage_t> data
) {
    auto array = std::make_unique<metatensor::SimpleDataArray<T>>(
        std::move(shape), std::move(data)
    );
    auto mts_array = metatensor::DataArrayBase::to_mts_array(std::move(array));
    return metatomic::DLPackTensor(mts_array.as_dlpack(
        {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
    ));
}

static metatomic::System two_atom_system(double distance) {
    return metatomic::System(
        "Angstrom",
        dlpack_tensor<int32_t>({2}, {1, 1}),
        dlpack_tensor<double>({2, 3}, {0.0, 0.0, 0.0, 0.0, 0.0, distance}),
        dlpack_tensor<double>({3, 3}, std::vector<double>(9, 0.0)),
        dlpack_tensor<bool>({3}, {0, 0, 0})
    );
}

static metatensor::TensorBlock pair_block(double distance) {
    auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{1, 3, 1},
        std::vector<double>{0.0, 0.0, distance}
    );
    return metatensor::TensorBlock(
        std::move(values),
        metatensor::Labels(
            {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            {{0, 1, 0, 0, 0}}
        ),
        {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
        metatensor::Labels({"distance"}, {{0}})
    );
}

static metatomic::ExternalModel load_lj(const std::string& options = "{}") {
    return metatomic::ExternalModel(
        metatomic::load_model("lennard-jones", options, "lj-plugin")
    );
}

static std::string energy_request(bool positions_gradient) {
    auto output = metatomic::Quantity::builder()
        .name("energy")
        .unit("eV")
        .sample_kind(metatomic::SampleKind::System);
    if (positions_gradient) {
        output.add_gradient(metatomic::Gradients::Positions);
    }
    nlohmann::json json = std::vector<metatomic::Quantity>{output.build()};
    return json.dump();
}

static metatensor::TensorMap execute(
    metatomic::ExternalModel& model,
    metatomic::System& system,
    bool positions_gradient
) {
    auto requested_outputs = energy_request(positions_gradient);
    auto* system_pointer = system.as_mta_system_t();
    mts_tensormap_t* result = nullptr;
    metatomic::details::check_status(mta_execute_model(
        *model.as_mta_model_t(),
        &system_pointer,
        1,
        nullptr,
        requested_outputs.c_str(),
        true,
        &result,
        1
    ));
    REQUIRE(result != nullptr);
    return metatensor::TensorMap::unsafe_from_ptr(result);
}

static double lj_energy(double distance, double sigma, double epsilon, double cutoff) {
    auto ratio = sigma / distance;
    auto ratio6 = std::pow(ratio, 6);
    auto cutoff_ratio6 = std::pow(sigma / cutoff, 6);
    return 4.0 * epsilon * (
        ratio6 * ratio6 - ratio6
        - cutoff_ratio6 * cutoff_ratio6 + cutoff_ratio6
    );
}

TEST_CASE("Lennard-Jones plugin") {
    static bool loaded = false;
    if (!loaded) {
        metatomic::load_plugin(LJ_PLUGIN_PATH);
        loaded = true;
    }

    SECTION("rejects unsupported models and invalid options") {
        CHECK_THROWS(metatomic::load_model("not-lj", "{}", "lj-plugin"));
        CHECK_THROWS(load_lj(R"({"sigma":1.0})"));
        CHECK_THROWS(load_lj(R"({"unknown":"1"})"));
        CHECK_THROWS(load_lj(R"({"cutoff":"0"})"));
    }

    SECTION("reports model information") {
        auto model = load_lj();
        auto capabilities = model.capabilities();
        CHECK(capabilities.atomic_types() == std::vector<int64_t>{1});
        CHECK(capabilities.interaction_range() == 3.0);
        CHECK(capabilities.length_unit() == "Angstrom");
        CHECK(model.metadata().name() == "Lennard-Jones");

        auto pairs = model.requested_pair_lists();
        REQUIRE(pairs.size() == 1);
        CHECK(pairs[0].cutoff() == 3.0);
        CHECK_FALSE(pairs[0].full_list());
        CHECK(pairs[0].strict());
    }

    SECTION("computes energy and positions gradient") {
        const double distance = 1.5;
        auto model = load_lj(
            R"({"sigma":"1.0","epsilon":"1.0","cutoff":"3.0"})"
        );
        auto system = two_atom_system(distance);
        system.add_pairs(model.requested_pair_lists()[0], pair_block(distance));

        auto result = execute(model, system, true);
        auto block = result.block_by_id(0);
        auto values = block.values<double>();
        CHECK(values(0, 0) == Approx(
            lj_energy(distance, 1.0, 1.0, 3.0)
        ).epsilon(1e-12));

        auto gradient = block.gradient("positions").values<double>();
        const auto step = 1e-6;
        const auto finite_difference = (
            lj_energy(distance + step, 1.0, 1.0, 3.0)
            - lj_energy(distance - step, 1.0, 1.0, 3.0)
        ) / (2.0 * step);
        CHECK(gradient(0, 2, 0) == Approx(-finite_difference).epsilon(1e-8));
        CHECK(gradient(1, 2, 0) == Approx(finite_difference).epsilon(1e-8));
    }

    SECTION("omits an unrequested gradient") {
        auto model = load_lj();
        auto system = two_atom_system(1.5);
        system.add_pairs(model.requested_pair_lists()[0], pair_block(1.5));

        auto result = execute(model, system, false);
        CHECK(result.block_by_id(0).gradients_list().empty());
    }

    SECTION("rejects selected atoms and multiple systems") {
        auto model = load_lj();
        auto system = two_atom_system(1.5);
        auto* system_pointer = system.as_mta_system_t();
        auto requested_outputs = energy_request(false);
        mts_tensormap_t* result = nullptr;

        auto selected_atoms = metatensor::Labels({"system", "atom"}, {{0, 0}});
        CHECK(model.as_mta_model_t()->execute_inner(
            model.as_mta_model_t()->data,
            &system_pointer,
            1,
            selected_atoms.as_mts_labels_t(),
            requested_outputs.c_str(),
            &result,
            1
        ) != MTA_SUCCESS);
        CHECK(result == nullptr);

        const mta_system_t* systems[2] = {system_pointer, system_pointer};
        CHECK(model.as_mta_model_t()->execute_inner(
            model.as_mta_model_t()->data,
            systems,
            2,
            nullptr,
            requested_outputs.c_str(),
            &result,
            1
        ) != MTA_SUCCESS);
        CHECK(result == nullptr);
    }
}
