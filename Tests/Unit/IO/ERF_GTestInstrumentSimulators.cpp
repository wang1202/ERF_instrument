#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include "ERF_IndexDefines.H"
#include "ERF_InstrumentSimulators.H"

namespace {

template <typename T>
class ScopedParmParseValue
{
public:
    ScopedParmParseValue (const char* prefix, const char* name, const T& value)
        : m_pp(prefix), m_name(name)
    {
        m_had_previous = m_pp.query(m_name, m_previous);
        m_pp.remove(m_name);
        m_pp.add(m_name, value);
    }

    ~ScopedParmParseValue ()
    {
        m_pp.remove(m_name);
        if (m_had_previous) {
            m_pp.add(m_name, m_previous);
        }
    }

private:
    amrex::ParmParse m_pp;
    std::string m_name;
    T m_previous{};
    bool m_had_previous = false;
};

template <typename T>
class ScopedParmParseArray
{
public:
    ScopedParmParseArray (const char* prefix, const char* name, const amrex::Vector<T>& values)
        : m_pp(prefix), m_name(name)
    {
        m_had_previous = m_pp.queryarr(m_name, m_previous);
        m_pp.remove(m_name);
        m_pp.addarr(m_name, values);
    }

    ~ScopedParmParseArray ()
    {
        m_pp.remove(m_name);
        if (m_had_previous) {
            m_pp.addarr(m_name, m_previous);
        }
    }

private:
    amrex::ParmParse m_pp;
    std::string m_name;
    amrex::Vector<T> m_previous;
    bool m_had_previous = false;
};

amrex::Geometry make_geometry ()
{
    amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(0, 0, 2));
    amrex::RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 3.0)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
    return amrex::Geometry(domain, &real_box, 0, is_periodic.data());
}

void fill_wsm6_state (amrex::MultiFab& cons, amrex::MultiFab& base_state)
{
    constexpr std::array<amrex::Real, 3> rho{1.0, 1.0, 1.0};
    constexpr std::array<amrex::Real, 3> theta{300.0, 300.0, 300.0};
    constexpr std::array<amrex::Real, 3> qv{0.010, 0.008, 0.006};
    constexpr std::array<amrex::Real, 3> qc{0.0010, 0.0005, 0.0000};
    constexpr std::array<amrex::Real, 3> qi{0.0000, 0.0002, 0.0004};
    constexpr std::array<amrex::Real, 3> qr{0.0003, 0.0002, 0.0001};
    constexpr std::array<amrex::Real, 3> qs{0.0000, 0.0001, 0.0002};
    constexpr std::array<amrex::Real, 3> qg{0.0000, 0.0000, 0.0001};
    constexpr std::array<amrex::Real, 3> p0{90000.0, 80000.0, 70000.0};

    cons.setVal(0.0);
    base_state.setVal(0.0);

    for (amrex::MFIter mfi(cons, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& cons_arr = cons.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            cons_arr(i,j,k,Rho_comp)      = rho[k];
            cons_arr(i,j,k,RhoTheta_comp) = rho[k] * theta[k];
            cons_arr(i,j,k,RhoQ1_comp)    = rho[k] * qv[k];
            cons_arr(i,j,k,RhoQ2_comp)    = rho[k] * qc[k];
            cons_arr(i,j,k,RhoQ3_comp)    = rho[k] * qi[k];
            cons_arr(i,j,k,RhoQ4_comp)    = rho[k] * qr[k];
            cons_arr(i,j,k,RhoQ5_comp)    = rho[k] * qs[k];
            cons_arr(i,j,k,RhoQ6_comp)    = rho[k] * qg[k];
        });
    }

    for (amrex::MFIter mfi(base_state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& base_arr = base_state.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            base_arr(i,j,k,BaseState::p0_comp) = p0[k];
        });
    }

    amrex::Gpu::streamSynchronize();
}

void fill_face_velocity (amrex::MultiFab& mf, amrex::Real value)
{
    mf.setVal(value);
    amrex::Gpu::streamSynchronize();
}

std::vector<std::string> read_data_lines (const std::filesystem::path& path)
{
    std::ifstream in(path);
    EXPECT_TRUE(in.good()) << "Failed to open " << path;

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<double> parse_row (const std::string& line)
{
    std::istringstream in(line);
    std::vector<double> values;
    double value = 0.0;
    while (in >> value) {
        values.push_back(value);
    }
    return values;
}

} // namespace

// Motivation: the instrument simulator path should accept the full WSM6
// moisture-state layout (qv/qc/qi/qr/qs/qg in RhoQ1..RhoQ6) without any
// scheme-specific hooks, producing valid ceilometer, Doppler lidar, and MWR
// outputs from the standard conserved-state slots.
TEST(InstrumentSimulators, WSM6StateLayoutProducesOutputs)
{
    const amrex::Geometry geom = make_geometry();
    const amrex::BoxArray ba(geom.Domain());
    const amrex::DistributionMapping dm(ba);

    amrex::MultiFab cons(ba, dm, RhoQ6_comp + 1, 0);
    amrex::MultiFab base_state(ba, dm, BaseState::num_comps, 0);

    auto xface_ba = ba;
    xface_ba.surroundingNodes(0);
    amrex::MultiFab xvel(xface_ba, dm, 1, 0);

    auto yface_ba = ba;
    yface_ba.surroundingNodes(1);
    amrex::MultiFab yvel(yface_ba, dm, 1, 0);

    auto zface_ba = ba;
    zface_ba.surroundingNodes(2);
    amrex::MultiFab zvel(zface_ba, dm, 1, 0);

    fill_wsm6_state(cons, base_state);
    fill_face_velocity(xvel, 1.0);
    fill_face_velocity(yvel, 2.0);
    fill_face_velocity(zvel, 0.5);

    const std::filesystem::path ceil_prefix = "wsm6_instrument_test_ceil_";
    const std::filesystem::path dl_prefix = "wsm6_instrument_test_dl_";
    const std::filesystem::path mwr_prefix = "wsm6_instrument_test_mwr_";
    const std::filesystem::path ceil_file = "wsm6_instrument_test_ceil_00.txt";
    const std::filesystem::path dl_stare_file = "wsm6_instrument_test_dl_00_stare.txt";
    const std::filesystem::path dl_vad_file = "wsm6_instrument_test_dl_00_vad.txt";
    const std::filesystem::path dl_beta_file = "wsm6_instrument_test_dl_00_beta.txt";
    const std::filesystem::path mwr_profiles_file = "wsm6_instrument_test_mwr_00_profiles.txt";
    const std::filesystem::path mwr_column_file = "wsm6_instrument_test_mwr_00_column.txt";

    for (const auto& path : {ceil_file, dl_stare_file, dl_vad_file, dl_beta_file,
                             mwr_profiles_file, mwr_column_file}) {
        std::filesystem::remove(path);
    }

    ScopedParmParseArray<int> ceil_i("erf.ceilometer", "i_loc", amrex::Vector<int>{0});
    ScopedParmParseArray<int> ceil_j("erf.ceilometer", "j_loc", amrex::Vector<int>{0});
    ScopedParmParseValue<int> ceil_interval("erf.ceilometer", "output_interval", 1);
    ScopedParmParseValue<std::string> ceil_output("erf.ceilometer", "output_file", ceil_prefix.string());
    ScopedParmParseValue<std::string> ceil_model("erf.ceilometer", "backscatter_model", std::string("simple"));

    ScopedParmParseArray<int> dl_i("erf.doppler_lidar", "i_loc", amrex::Vector<int>{0});
    ScopedParmParseArray<int> dl_j("erf.doppler_lidar", "j_loc", amrex::Vector<int>{0});
    ScopedParmParseValue<int> dl_interval("erf.doppler_lidar", "output_interval", 1);
    ScopedParmParseValue<std::string> dl_output("erf.doppler_lidar", "output_file", dl_prefix.string());
    ScopedParmParseValue<int> dl_stare("erf.doppler_lidar", "do_vertical_stare", 1);
    ScopedParmParseValue<int> dl_vad("erf.doppler_lidar", "do_vad", 1);
    ScopedParmParseValue<int> dl_backscatter("erf.doppler_lidar", "do_backscatter", 1);

    ScopedParmParseArray<int> mwr_i("erf.mwr", "i_loc", amrex::Vector<int>{0});
    ScopedParmParseArray<int> mwr_j("erf.mwr", "j_loc", amrex::Vector<int>{0});
    ScopedParmParseValue<int> mwr_interval("erf.mwr", "output_interval", 1);
    ScopedParmParseValue<std::string> mwr_output("erf.mwr", "output_file", mwr_prefix.string());
    ScopedParmParseValue<int> mwr_tb("erf.mwr", "do_brightness_temp", 0);

    {
        CeilometerSimulator ceilometer;
        ceilometer.init(geom);
        ceilometer.write(0, 0.0, 0, cons, cons, geom);
    }

    {
        DopplerLidarSimulator doppler;
        doppler.init(geom);
        doppler.write(0, 0.0, 0, cons, xvel, yvel, zvel, cons, geom);
    }

    {
        MWRSimulator mwr;
        mwr.init(geom);
        mwr.write(0, 0.0, 0, cons, base_state, cons, geom);
    }

    ASSERT_TRUE(std::filesystem::exists(ceil_file));
    ASSERT_TRUE(std::filesystem::exists(dl_stare_file));
    ASSERT_TRUE(std::filesystem::exists(dl_vad_file));
    ASSERT_TRUE(std::filesystem::exists(dl_beta_file));
    ASSERT_TRUE(std::filesystem::exists(mwr_profiles_file));
    ASSERT_TRUE(std::filesystem::exists(mwr_column_file));

    const auto ceil_lines = read_data_lines(ceil_file);
    ASSERT_EQ(ceil_lines.size(), 3u);
    const auto ceil_row = parse_row(ceil_lines.front());
    ASSERT_EQ(ceil_row.size(), 8u);
    EXPECT_DOUBLE_EQ(ceil_row[3], 0.0);
    EXPECT_DOUBLE_EQ(ceil_row[4], 0.0);
    EXPECT_GT(ceil_row[6], 0.0);
    EXPECT_GE(ceil_row[7], 0.0);

    const auto dl_stare_lines = read_data_lines(dl_stare_file);
    ASSERT_EQ(dl_stare_lines.size(), 3u);
    const auto dl_stare_row = parse_row(dl_stare_lines.front());
    ASSERT_EQ(dl_stare_row.size(), 8u);
    EXPECT_NEAR(dl_stare_row[6], 0.5, 1.0e-12);
    EXPECT_GT(dl_stare_row[7], 0.0);

    const auto dl_vad_lines = read_data_lines(dl_vad_file);
    ASSERT_EQ(dl_vad_lines.size(), 3u);
    const auto dl_vad_row = parse_row(dl_vad_lines.front());
    ASSERT_EQ(dl_vad_row.size(), 10u);
    EXPECT_NEAR(dl_vad_row[6], 1.0, 1.0e-12);
    EXPECT_NEAR(dl_vad_row[7], 2.0, 1.0e-12);
    EXPECT_NEAR(dl_vad_row[8], std::sqrt(5.0), 1.0e-12);

    const auto dl_beta_lines = read_data_lines(dl_beta_file);
    ASSERT_EQ(dl_beta_lines.size(), 3u);
    const auto dl_beta_row = parse_row(dl_beta_lines.front());
    ASSERT_EQ(dl_beta_row.size(), 7u);
    EXPECT_GT(dl_beta_row[6], 0.0);

    const auto mwr_profiles_lines = read_data_lines(mwr_profiles_file);
    ASSERT_EQ(mwr_profiles_lines.size(), 3u);
    const auto mwr_profile_row = parse_row(mwr_profiles_lines.front());
    ASSERT_EQ(mwr_profile_row.size(), 9u);
    EXPECT_NEAR(mwr_profile_row[7], 10.0, 1.0e-12);
    EXPECT_NEAR(mwr_profile_row[8], 1.0, 1.0e-12);

    const auto mwr_column_lines = read_data_lines(mwr_column_file);
    ASSERT_EQ(mwr_column_lines.size(), 1u);
    const auto mwr_column_row = parse_row(mwr_column_lines.front());
    ASSERT_EQ(mwr_column_row.size(), 7u);
    EXPECT_NEAR(mwr_column_row[5], 1.5, 1.0e-12);
    EXPECT_NEAR(mwr_column_row[6], 0.024, 1.0e-12);

    for (const auto& path : {ceil_file, dl_stare_file, dl_vad_file, dl_beta_file,
                             mwr_profiles_file, mwr_column_file}) {
        std::filesystem::remove(path);
    }
}
