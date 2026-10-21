// Analytic nonperiodic hexamer CVs with position and homogeneous-strain gradients.
#include <array>
#include <cmath>
#include <fstream>
#include <metatomic.hpp>

namespace mta = metatomic;
namespace mts = metatensor;

class Hexamer final: public mta::BaseModel {
public:
    explicit Hexamer(bool coordination): coordination_(coordination) {}

    explicit Hexamer(const std::string& path): coordination_(false) {
        std::ifstream file(path);
        size_t count = 0;
        file >> count;
        if (count != 18) throw mta::Error("Target must contain 18 atoms");
        std::string line;
        std::getline(file, line); std::getline(file, line);
        for (size_t i = 0; i < count; ++i) {
            std::string symbol;
            std::array<double, 3> r;
            if (!(file >> symbol >> r[0] >> r[1] >> r[2])) throw mta::Error("Invalid target XYZ");
            if (symbol != (i % 3 == 0 ? "O" : "H")) throw mta::Error("Target requires OHH order");
            reference_.push_back(r);
        }
    }

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
            .name(!reference_.empty() ? "Aligned all-atom RMSD" : (coordination_ ? "Mean oxygen coordination" : "Oxygen radius of gyration"))
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
                    if (!reference_.empty() || z[i * (t.strides ? t.strides[0] : 1)] == 8) {
                        std::array<double, 3> r;
                        for (size_t k = 0; k < 3; ++k) {
                            r[k] = xyz[i * (p.strides ? p.strides[0] : 3) + k * (p.strides ? p.strides[1] : 1)];
                        }
                        oxygen.push_back(r);
                        oxygen_ids.push_back(i);
                    }
                }
                if (oxygen.size() != (reference_.empty() ? 6 : 18)) throw mta::Error("Unexpected atom count");
                // This core revision requires a finite interaction range.
                for (const auto& a: oxygen) for (const auto& b: oxygen) {
                    double r2 = 0;
                    for (size_t k = 0; k < 3; ++k) r2 += std::pow(a[k] - b[k], 2);
                    if (r2 > 100.0 * 100.0) throw mta::Error("Cluster exceeds the 100 A supported diameter");
                }
                double value = 0;
                std::vector<std::array<double, 3>> gradient(oxygen.size());
                if (!reference_.empty()) {
                    // Optimal proper rotation from Horn's quaternion matrix.
                    // A symmetric Jacobi eigensolver avoids an external linear algebra dependency.
                    std::array<double, 3> center{}, target_center{};
                    for (size_t i = 0; i < 18; ++i) for (size_t k = 0; k < 3; ++k) {
                        center[k] += oxygen[i][k] / 18.0;
                        target_center[k] += reference_[i][k] / 18.0;
                    }
                    double h[3][3] = {};
                    for (size_t i = 0; i < 18; ++i) for (size_t a = 0; a < 3; ++a)
                        for (size_t b = 0; b < 3; ++b)
                            h[a][b] += (oxygen[i][a] - center[a]) * (reference_[i][b] - target_center[b]);
                    double n[4][4] = {
                        {h[0][0]+h[1][1]+h[2][2], h[1][2]-h[2][1], h[2][0]-h[0][2], h[0][1]-h[1][0]},
                        {h[1][2]-h[2][1], h[0][0]-h[1][1]-h[2][2], h[0][1]+h[1][0], h[0][2]+h[2][0]},
                        {h[2][0]-h[0][2], h[0][1]+h[1][0], -h[0][0]+h[1][1]-h[2][2], h[1][2]+h[2][1]},
                        {h[0][1]-h[1][0], h[0][2]+h[2][0], h[1][2]+h[2][1], -h[0][0]-h[1][1]+h[2][2]}
                    };
                    double eigenvectors[4][4] = {};
                    for (size_t i = 0; i < 4; ++i) eigenvectors[i][i] = 1.0;
                    for (size_t iteration = 0; iteration < 100; ++iteration) {
                        size_t a = 0, b = 1;
                        for (size_t i = 0; i < 4; ++i) for (size_t j = i+1; j < 4; ++j)
                            if (std::abs(n[i][j]) > std::abs(n[a][b])) { a=i; b=j; }
                        if (std::abs(n[a][b]) < 1e-13) break;
                        const double theta = 0.5 * std::atan2(2*n[a][b], n[b][b]-n[a][a]);
                        const double c = std::cos(theta), t = std::sin(theta);
                        const double aa=n[a][a], bb=n[b][b], ab=n[a][b];
                        n[a][a]=c*c*aa-2*c*t*ab+t*t*bb;
                        n[b][b]=t*t*aa+2*c*t*ab+c*c*bb;
                        n[a][b]=n[b][a]=0;
                        for (size_t k = 0; k < 4; ++k) {
                            if (k != a && k != b) {
                                const double ka=n[k][a], kb=n[k][b];
                                n[k][a]=n[a][k]=c*ka-t*kb;
                                n[k][b]=n[b][k]=t*ka+c*kb;
                            }
                            const double va=eigenvectors[k][a], vb=eigenvectors[k][b];
                            eigenvectors[k][a]=c*va-t*vb;
                            eigenvectors[k][b]=t*va+c*vb;
                        }
                    }
                    size_t largest = 0;
                    for (size_t k = 1; k < 4; ++k) if (n[k][k] > n[largest][largest]) largest=k;
                    const double w=eigenvectors[0][largest], x=eigenvectors[1][largest];
                    const double y=eigenvectors[2][largest], zq=eigenvectors[3][largest];
                    // Column-vector rotation mapping current coordinates to reference.
                    const double rotation[3][3] = {
                        {1-2*(y*y+zq*zq), 2*(x*y-zq*w), 2*(x*zq+y*w)},
                        {2*(x*y+zq*w), 1-2*(x*x+zq*zq), 2*(y*zq-x*w)},
                        {2*(x*zq-y*w), 2*(y*zq+x*w), 1-2*(x*x+y*y)}
                    };
                    for (size_t i = 0; i < 18; ++i) for (size_t a = 0; a < 3; ++a) {
                        double target = 0;
                        for (size_t b = 0; b < 3; ++b)
                            target += rotation[b][a] * (reference_[i][b] - target_center[b]);
                        const double delta = oxygen[i][a] - center[a] - target;
                        value += delta*delta / 18.0;
                        gradient[i][a] = delta / 18.0;
                    }
                    value = std::sqrt(value + 1e-16);
                    // Envelope theorem: the derivative of the optimal rotation cancels.
                    for (auto& g: gradient) for (double& x: g) x /= value;
                } else if (coordination_) {
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
                for (size_t i = 0; i < oxygen.size(); ++i) {
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
    std::vector<std::array<double, 3>> reference_;
};

std::unique_ptr<mta::BaseModel> load_hexamer(const std::string& name, const std::map<std::string, std::string>&) {
    if (name.rfind("rmsd-target:", 0) == 0) return std::make_unique<Hexamer>(name.substr(12));
    if (name == "oxygen-rg") return std::make_unique<Hexamer>(false);
    if (name == "oxygen-cn") return std::make_unique<Hexamer>(true);
    return nullptr;
}
MTA_REGISTER_CXX_PLUGIN("water-hexamer", load_hexamer);
