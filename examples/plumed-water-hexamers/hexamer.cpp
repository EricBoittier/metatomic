// Analytic nonperiodic hexamer CVs with position and homogeneous-strain gradients.
#include <array>
#include <cmath>
#include <metatomic.hpp>

namespace mta = metatomic;
namespace mts = metatensor;

class Hexamer final: public mta::BaseModel {
public:
    explicit Hexamer(bool coordination): coordination_(coordination) {}

    mta::ModelCapabilities capabilities() const override {
        return mta::ModelCapabilities::builder()
            .atomic_types({1, 8}).interaction_range(100.0)
            .length_unit("angstrom")
            .supported_devices({mta::ModelCapabilities::Device::CPU})
            .dtype(mta::ModelCapabilities::DType::Float64)
            .add_output(mta::Quantity::builder().name("feature").unit("")
                .sample_kind(mta::SampleKind::System)
                .gradients({mta::Gradients::Positions, mta::Gradients::Strain}).build()).build();
    }
    mta::ModelMetadata metadata() const override {
        return mta::ModelMetadata::builder()
            .name(coordination_ ? "Mean oxygen coordination" : "Oxygen radius of gyration")
            .description("Analytic nonperiodic hexamer descriptor with analytic gradients")
            .build();
    }
    std::vector<mta::PairListOptions> requested_pair_lists() const override { return {}; }
    std::vector<mta::Quantity> requested_inputs() const override { return {}; }

    std::vector<mts::TensorMap> execute_inner(
        const std::vector<mta::System>& systems,
        std::optional<mts::Labels> selected,
        const std::vector<mta::Quantity>& outputs
    ) override {
        if (selected) throw mta::Error("selected atoms are not supported");
        std::vector<mts::TensorMap> result;
        for (const auto& output: outputs) {
            if (output.name() != "feature") {
                throw mta::Error("Only feature is supported");
            }
            std::vector<double> values;
            std::vector<int32_t> samples, gradient_samples, strain_samples;
            std::vector<double> position_gradients, strain_gradients;
            for (size_t s = 0; s < systems.size(); ++s) {
                const auto& system = systems[s];
                if (system.size() == 0) continue; // PLUMED shape probe
                auto positions = system.positions();
                auto types = system.types();
                const auto& p = positions->dl_tensor;
                const auto& t = types->dl_tensor;
                auto* xyz = reinterpret_cast<const double*>(static_cast<const char*>(p.data) + p.byte_offset);
                auto* z = reinterpret_cast<const int32_t*>(static_cast<const char*>(t.data) + t.byte_offset);
                std::vector<std::array<double, 3>> oxygen;
                std::vector<size_t> oxygen_ids;
                for (size_t i = 0; i < system.size(); ++i) {
                    if (z[i * (t.strides ? t.strides[0] : 1)] == 8) {
                        std::array<double, 3> r;
                        for (size_t k = 0; k < 3; ++k) {
                            r[k] = xyz[i * (p.strides ? p.strides[0] : 3) + k * (p.strides ? p.strides[1] : 1)];
                        }
                        oxygen.push_back(r);
                        oxygen_ids.push_back(i);
                    }
                }
                if (oxygen.size() != 6) throw mta::Error("Expected six oxygen atoms");
                // This core revision requires a finite interaction range.
                for (const auto& a: oxygen) for (const auto& b: oxygen) {
                    double r2 = 0;
                    for (size_t k = 0; k < 3; ++k) r2 += std::pow(a[k] - b[k], 2);
                    if (r2 > 100.0 * 100.0) throw mta::Error("Cluster exceeds the 100 A supported diameter");
                }
                double value = 0;
                std::array<std::array<double, 3>, 6> gradient{};
                if (coordination_) {
                    for (size_t i = 0; i < 6; ++i) {
                        for (size_t j = i + 1; j < 6; ++j) {
                            double r2 = 0;
                            for (size_t k = 0; k < 3; ++k) r2 += std::pow(oxygen[i][k] - oxygen[j][k], 2);
                            const double u = r2 / (3.2 * 3.2);
                            const double denominator = 1.0 + u * u * u;
                            value += 1.0 / (3.0 * denominator);
                            const double factor = -2.0 * u * u / (3.2 * 3.2 * denominator * denominator);
                            for (size_t k = 0; k < 3; ++k) {
                                const double g = factor * (oxygen[i][k] - oxygen[j][k]);
                                gradient[i][k] += g;
                                gradient[j][k] -= g;
                            }
                        }
                    }
                } else {
                    std::array<double, 3> center{};
                    for (const auto& r: oxygen) for (size_t k = 0; k < 3; ++k) center[k] += r[k] / 6;
                    for (const auto& r: oxygen) for (size_t k = 0; k < 3; ++k) value += std::pow(r[k] - center[k], 2) / 6;
                    value = std::sqrt(value);
                    if (value == 0) throw mta::Error("Radius gradient is undefined for coincident oxygens");
                    for (size_t i = 0; i < 6; ++i) for (size_t k = 0; k < 3; ++k)
                        gradient[i][k] = (oxygen[i][k] - center[k]) / (6.0 * value);
                }
                const auto sample = static_cast<int32_t>(values.size());
                std::vector<double> atom_gradients(3 * system.size(), 0.0);
                std::array<double, 9> strain{};
                for (size_t i = 0; i < 6; ++i) {
                    for (size_t k = 0; k < 3; ++k) atom_gradients[3 * oxygen_ids[i] + k] = gradient[i][k];
                    // r' = r F: dCV/dF_ab = sum_i r_ia dCV/dr_ib.
                    for (size_t a = 0; a < 3; ++a) for (size_t b = 0; b < 3; ++b)
                        strain[3 * a + b] += oxygen[i][a] * gradient[i][b];
                }
                for (size_t i = 0; i < system.size(); ++i) {
                    gradient_samples.insert(gradient_samples.end(), {sample, static_cast<int32_t>(s), static_cast<int32_t>(i)});
                }
                position_gradients.insert(position_gradients.end(), atom_gradients.begin(), atom_gradients.end());
                strain_samples.push_back(sample);
                strain_gradients.insert(strain_gradients.end(), strain.begin(), strain.end());
                values.push_back(value);
                samples.push_back(static_cast<int32_t>(s));
            }
            const auto count = values.size();
            auto block = mts::TensorBlock(
                std::make_unique<mts::SimpleDataArray<double>>(std::vector<uintptr_t>{count, 1}, std::move(values)),
                mts::Labels({"system"}, samples.data(), count), {}, mts::Labels({"cv"}, {{0}}));
            for (auto requested: output.gradients()) {
                if (requested == mta::Gradients::Positions) {
                    const size_t n = gradient_samples.size() / 3;
                    block.add_gradient("positions", mts::TensorBlock(
                        std::make_unique<mts::SimpleDataArray<double>>(std::vector<uintptr_t>{n, 3, 1}, position_gradients),
                        mts::Labels({"sample", "system", "atom"}, gradient_samples.data(), n),
                        {mts::Labels({"xyz"}, {{0}, {1}, {2}})}, mts::Labels({"cv"}, {{0}})));
                } else if (requested == mta::Gradients::Strain) {
                    block.add_gradient("strain", mts::TensorBlock(
                        std::make_unique<mts::SimpleDataArray<double>>(std::vector<uintptr_t>{count, 3, 3, 1}, strain_gradients),
                        mts::Labels({"sample"}, strain_samples.data(), count),
                        {mts::Labels({"xyz_1"}, {{0}, {1}, {2}}), mts::Labels({"xyz_2"}, {{0}, {1}, {2}})},
                        mts::Labels({"cv"}, {{0}})));
                } else throw mta::Error("Unsupported gradient");
            }
            std::vector<mts::TensorBlock> blocks;
            blocks.push_back(std::move(block));
            result.emplace_back(mts::Labels({"_"}, {{0}}), std::move(blocks));
        }
        return result;
    }
private:
    bool coordination_;
};

std::unique_ptr<mta::BaseModel> load_hexamer(const std::string& name, const std::map<std::string, std::string>&) {
    if (name == "oxygen-rg") return std::make_unique<Hexamer>(false);
    if (name == "oxygen-cn") return std::make_unique<Hexamer>(true);
    return nullptr;
}
MTA_REGISTER_CXX_PLUGIN("water-hexamer", load_hexamer);
