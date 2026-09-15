#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <metatomic.hpp>
#include <nlohmann/json.hpp>


namespace {

struct LennardJonesOptions {
    double sigma = 1.0;
    double epsilon = 1.0;
    double cutoff = 3.0;
    int32_t atomic_type = 1;
    std::string length_unit = "Angstrom";
    std::string energy_unit = "eV";
};

double parse_double(const nlohmann::json& json, const std::string& name, double fallback) {
    if (!json.contains(name)) {
        return fallback;
    }

    const auto value = json[name].get<std::string>();
    // std::stod follows LC_NUMERIC; "1.0" then fails on a comma-decimal locale.
    std::istringstream in(value);
    in.imbue(std::locale::classic());
    double result = 0.0;
    in >> std::noskipws >> result;
    if (!in || in.get() != std::char_traits<char>::eof()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be a number");
    }
    return result;
}

int32_t parse_atomic_type(const nlohmann::json& json) {
    if (!json.contains("atomic_type")) {
        return 1;
    }

    const auto value = json["atomic_type"].get<std::string>();
    size_t parsed = 0;
    int64_t result = 0;
    try {
        result = std::stoll(value, &parsed);
    } catch (const std::exception&) {
        throw metatomic::Error("Lennard-Jones option 'atomic_type' must be an integer");
    }
    if (parsed != value.size()) {
        throw metatomic::Error("Lennard-Jones option 'atomic_type' must be an integer");
    }
    if (result < std::numeric_limits<int32_t>::min()
        || result > std::numeric_limits<int32_t>::max())
    {
        throw metatomic::Error("Lennard-Jones option 'atomic_type' is out of range");
    }
    return static_cast<int32_t>(result);
}

LennardJonesOptions parse_options(const char* options_json) {
    auto json = nlohmann::json::parse(options_json == nullptr ? "{}" : options_json);
    if (!json.is_object()) {
        throw metatomic::Error("Lennard-Jones options must be a JSON object");
    }

    const std::vector<std::string> allowed = {
        "sigma", "epsilon", "cutoff", "atomic_type", "length_unit", "energy_unit"
    };
    for (const auto& item: json.items()) {
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end()) {
            throw metatomic::Error("unknown Lennard-Jones option: '" + item.key() + "'");
        }
    }

    LennardJonesOptions options;
    options.sigma = parse_double(json, "sigma", options.sigma);
    options.epsilon = parse_double(json, "epsilon", options.epsilon);
    options.cutoff = parse_double(json, "cutoff", options.cutoff);
    options.atomic_type = parse_atomic_type(json);
    if (json.contains("length_unit")) {
        options.length_unit = json["length_unit"].get<std::string>();
    }
    if (json.contains("energy_unit")) {
        options.energy_unit = json["energy_unit"].get<std::string>();
    }

    if (!std::isfinite(options.sigma) || options.sigma <= 0.0) {
        throw metatomic::Error("Lennard-Jones option 'sigma' must be finite and positive");
    }
    if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
        throw metatomic::Error("Lennard-Jones option 'epsilon' must be finite and positive");
    }
    if (!std::isfinite(options.cutoff) || options.cutoff <= 0.0) {
        throw metatomic::Error("Lennard-Jones option 'cutoff' must be finite and positive");
    }
    if (options.length_unit.empty()) {
        throw metatomic::Error("Lennard-Jones option 'length_unit' must not be empty");
    }
    if (options.energy_unit.empty()) {
        throw metatomic::Error("Lennard-Jones option 'energy_unit' must not be empty");
    }

    return options;
}

struct Calculation {
    double energy = 0.0;
    std::vector<double> positions_gradient;
};

metatensor::TensorMap energy_output(
    const Calculation& calculation,
    bool include_positions_gradient
) {
    auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
        std::vector<uintptr_t>{1, 1}, std::vector<double>{calculation.energy}
    );
    auto properties = metatensor::Labels({"energy"}, {{0}});
    auto block = metatensor::TensorBlock(
        std::move(values), metatensor::Labels({"system"}, {{0}}), {}, properties
    );

    if (include_positions_gradient) {
        auto atom_count = calculation.positions_gradient.size() / 3;
        auto sample_values = std::vector<int32_t>();
        sample_values.reserve(3 * atom_count);
        for (size_t atom = 0; atom < atom_count; atom++) {
            sample_values.insert(
                sample_values.end(), {0, 0, static_cast<int32_t>(atom)}
            );
        }

        auto gradient_values = std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{atom_count, 3, 1},
            calculation.positions_gradient
        );
        auto gradient = metatensor::TensorBlock(
            std::move(gradient_values),
            metatensor::Labels(
                {"sample", "system", "atom"}, sample_values.data(), atom_count
            ),
            {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
            properties
        );
        block.add_gradient("positions", std::move(gradient));
    }

    std::vector<metatensor::TensorBlock> blocks;
    blocks.push_back(std::move(block));
    return metatensor::TensorMap(
        metatensor::Labels({"_"}, {{0}}), std::move(blocks)
    );
}

class LennardJones final: public metatomic::BaseModel {
public:
    explicit LennardJones(LennardJonesOptions options):
        options_(std::move(options)),
        pair_options_(metatomic::PairListOptions::builder()
            .cutoff(options_.cutoff)
            .full_list(false)
            .strict(true)
            .add_requestor("lj-plugin")
            .build())
    {}

    metatomic::ModelCapabilities capabilities() const final {
        auto energy = metatomic::Quantity::builder()
            .name("energy")
            .unit(options_.energy_unit)
            .sample_kind(metatomic::SampleKind::System)
            .add_gradient(metatomic::Gradients::Positions)
            .build();

        return metatomic::ModelCapabilities::builder()
            .atomic_types({options_.atomic_type})
            .interaction_range(options_.cutoff)
            .length_unit(options_.length_unit)
            .supported_devices({metatomic::ModelCapabilities::Device::CPU})
            .dtype(metatomic::ModelCapabilities::DType::Float64)
            .add_output(energy)
            .build();
    }

    metatomic::ModelMetadata metadata() const final {
        return metatomic::ModelMetadata::builder()
            .name("Lennard-Jones")
            .add_author("metatomic")
            .description("Shifted Lennard-Jones pair potential for engine tests")
            .add_reference("model", "https://github.com/metatensor/lj-test")
            .add_reference("implementation", "https://github.com/metatensor/metatomic")
            .build();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const final {
        return {pair_options_};
    }

    std::vector<metatomic::Quantity> requested_inputs() const final {
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) final {
        if (systems.size() != 1) {
            throw metatomic::Error("Lennard-Jones plugin requires exactly one system");
        }
        if (selected_atoms != nullptr) {
            throw metatomic::Error("Lennard-Jones plugin does not support selected atoms");
        }

        for (const auto& output: requested_outputs) {
            validate_output(output);
        }
        if (requested_outputs.empty()) {
            return {};
        }

        const auto calculation = calculate(systems[0]);
        std::vector<metatensor::TensorMap> outputs;
        outputs.reserve(requested_outputs.size());
        for (const auto& output: requested_outputs) {
            const auto& gradients = output.gradients();
            auto positions = std::find(
                gradients.begin(), gradients.end(), metatomic::Gradients::Positions
            );
            outputs.push_back(energy_output(calculation, positions != gradients.end()));
        }
        return outputs;
    }

private:
    void validate_output(const metatomic::Quantity& output) const {
        if (output.name() != "energy") {
            throw metatomic::Error("Lennard-Jones plugin only supports the 'energy' output");
        }
        if (output.unit() != options_.energy_unit) {
            throw metatomic::Error(
                "Lennard-Jones plugin requires energy unit '" + options_.energy_unit + "'"
            );
        }
        if (output.sample_kind() != metatomic::SampleKind::System) {
            throw metatomic::Error("Lennard-Jones energy must use system samples");
        }
        for (const auto gradient: output.gradients()) {
            if (gradient != metatomic::Gradients::Positions) {
                throw metatomic::Error(
                    "Lennard-Jones plugin only supports positions gradients"
                );
            }
        }
    }

    Calculation calculate(const metatomic::System& system) const {
        auto pairs = system.pairs(pair_options_);
        auto displacements = pairs.values<double>();
        const auto pair_samples = pairs.samples().values_cpu();
        if (displacements.shape().size() != 3
            || displacements.shape()[1] != 3
            || displacements.shape()[2] != 1)
        {
            throw metatomic::Error("Lennard-Jones pair values must have shape (pairs, 3, 1)");
        }
        if (pair_samples.shape().size() != 2 || pair_samples.shape()[1] < 2) {
            throw metatomic::Error("Lennard-Jones pair samples must identify two atoms");
        }
        if (pair_samples.shape()[0] != displacements.shape()[0]) {
            throw metatomic::Error("Lennard-Jones pair samples and values have different sizes");
        }

        Calculation result;
        result.positions_gradient.resize(3 * system.size(), 0.0);
        const auto cutoff_squared = options_.cutoff * options_.cutoff;
        const auto sigma_squared = options_.sigma * options_.sigma;
        const auto cutoff_ratio_squared = sigma_squared / cutoff_squared;
        const auto cutoff_ratio_sixth = cutoff_ratio_squared
            * cutoff_ratio_squared * cutoff_ratio_squared;
        const auto shift = 4.0 * options_.epsilon
            * (cutoff_ratio_sixth * cutoff_ratio_sixth - cutoff_ratio_sixth);

        for (size_t pair = 0; pair < displacements.shape()[0]; pair++) {
            const auto dx = displacements(pair, 0, 0);
            const auto dy = displacements(pair, 1, 0);
            const auto dz = displacements(pair, 2, 0);
            const auto distance_squared = dx * dx + dy * dy + dz * dz;
            if (distance_squared <= 0.0) {
                throw metatomic::Error("Lennard-Jones pair distance must be positive");
            }
            if (distance_squared >= cutoff_squared) {
                continue;
            }

            const auto ratio_squared = sigma_squared / distance_squared;
            const auto ratio_sixth = ratio_squared * ratio_squared * ratio_squared;
            const auto ratio_twelfth = ratio_sixth * ratio_sixth;
            result.energy += 4.0 * options_.epsilon
                * (ratio_twelfth - ratio_sixth) - shift;

            const auto first = pair_samples(pair, 0);
            const auto second = pair_samples(pair, 1);
            if (first < 0 || second < 0
                || static_cast<size_t>(first) >= system.size()
                || static_cast<size_t>(second) >= system.size())
            {
                throw metatomic::Error("Lennard-Jones pair contains an invalid atom index");
            }

            const auto energy_derivative = 12.0 * options_.epsilon / distance_squared
                * (ratio_sixth - 2.0 * ratio_twelfth);
            const double displacement[3] = {dx, dy, dz};
            for (size_t xyz = 0; xyz < 3; xyz++) {
                const auto gradient = -2.0 * energy_derivative * displacement[xyz];
                result.positions_gradient[3 * static_cast<size_t>(first) + xyz] += gradient;
                result.positions_gradient[3 * static_cast<size_t>(second) + xyz] -= gradient;
            }
        }

        return result;
    }

    LennardJonesOptions options_;
    metatomic::PairListOptions pair_options_;
};

mta_status_t load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    if (load_from == nullptr
        || (std::string(load_from) != "lennard-jones" && std::string(load_from) != "lj"))
    {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    return metatomic::details::catch_exceptions([&]() {
        if (model == nullptr) {
            throw metatomic::Error("model output pointer must not be null");
        }
        *model = metatomic::BaseModel::to_mta_model(
            std::make_unique<LennardJones>(parse_options(options_json))
        );
    });
}

} // namespace


MTA_REGISTER_PLUGIN(register_plugin, {
    mta_plugin_t plugin = {};
    plugin.abi_version = MTA_ABI_VERSION;
    plugin.name = "lj-plugin";
    plugin.load_model = load_model;
    return register_plugin(plugin);
});
