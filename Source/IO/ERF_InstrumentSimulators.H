#ifndef ERF_INSTRUMENT_SIMULATORS_H
#define ERF_INSTRUMENT_SIMULATORS_H

/**
 * ERF Instrument Simulators
 *
 * Provides forward-operator simulators for three ground-based remote sensing instruments:
 *   - CeilometerSimulator   : Vaisala CL-31 (905 nm backscatter lidar)
 *   - DopplerLidarSimulator : Halo Photonics StreamLine XR+ (1550 nm coherent Doppler lidar)
 *   - MWRSimulator          : Radiometrics MP-3000A (K+V band microwave radiometer)
 *
 * Usage (inputs file) — all parameters with their defaults:
 *
 *   # ---- Ceilometer (Vaisala CL-31, 905 nm) ----
 *   erf.ceilometer.do_sim             = true
 *   erf.ceilometer.i_loc              = 32 48      # REQUIRED; one or more 0-based i indices
 *   erf.ceilometer.j_loc              = 32 48      # REQUIRED; must match i_loc count
 *   erf.ceilometer.output_interval    = 10         # write every N timesteps
 *   erf.ceilometer.output_file        = ceil_      # prefix; ceil_00.txt, ceil_01.txt, ...
 *   erf.ceilometer.backscatter_model  = simple     # "simple" (geometric optics) or "mie"
 *   erf.ceilometer.r_eff_default      = 1.0e-5    # m; liquid droplet effective radius fallback
 *   erf.ceilometer.r_eff_ice_default  = 3.0e-5    # m; ice crystal effective radius fallback
 *   erf.ceilometer.beta_thresh        = 3.0e-4    # m-1 sr-1; cloud base detection threshold
 *   erf.ceilometer.move_start_time    = 60. 120.  # optional moving-column intervals [s]
 *   erf.ceilometer.move_stop_time     = 120. 240.
 *   erf.ceilometer.move_speed_x       = -10. 0.   # optional physical speeds [m s-1]
 *   erf.ceilometer.move_speed_y       = 0. -5.
 *
 *   # ---- Doppler Lidar (Halo StreamLine XR+, 1550 nm) ----
 *   erf.doppler_lidar.do_sim              = true
 *   erf.doppler_lidar.i_loc               = 32      # REQUIRED
 *   erf.doppler_lidar.j_loc               = 32      # REQUIRED
 *   erf.doppler_lidar.output_interval     = 10
 *   erf.doppler_lidar.output_file         = dl_
 *   erf.doppler_lidar.backscatter_model   = simple  # "simple" or "mie"
 *   erf.doppler_lidar.r_eff_default       = 1.0e-5  # m; liquid effective radius fallback
 *   erf.doppler_lidar.do_vertical_stare   = true    # output w(z,t) and CNR
 *   erf.doppler_lidar.do_vad              = false   # output u,v,wind_speed,wind_dir via VAD
 *   erf.doppler_lidar.vad_elevation_angle = 75.0    # degrees from horizontal (VAD cone angle)
 *   erf.doppler_lidar.do_backscatter      = false   # output beta_att(z,t) at 1550 nm
 *   erf.doppler_lidar.cnr_model           = normalized  # "normalized" or "physical"
 *   # Physical CNR only (Targ et al. 1991) — ignored when cnr_model=normalized:
 *   erf.doppler_lidar.pulse_energy        = 1.5e-6  # E   [J]   pulse energy
 *   erf.doppler_lidar.optics_efficiency   = 0.1     # eta [-]   optical+detector efficiency
 *   erf.doppler_lidar.beam_radius         = 0.0225  # omega [m] 1/e^2 beam radius at aperture
 *   erf.doppler_lidar.bandwidth           = 50.0e6  # Bw  [Hz]  receiver noise bandwidth
 *   erf.doppler_lidar.focus_range         = 1.0e9   # R   [m]   radius of curvature (1e9=collimated)
 *   erf.doppler_lidar.move_start_time     = 60. 120. # optional moving-column intervals [s]
 *   erf.doppler_lidar.move_stop_time      = 120. 240.
 *   erf.doppler_lidar.move_speed_x        = -10. 0.  # optional physical speeds [m s-1]
 *   erf.doppler_lidar.move_speed_y        = 0. -5.
 *
 *   # ---- MWR (Radiometrics MP-3000A, K+V band, 35 channels) ----
 *   erf.mwr.do_sim             = true
 *   erf.mwr.i_loc              = 32      # REQUIRED
 *   erf.mwr.j_loc              = 32      # REQUIRED
 *   erf.mwr.output_interval    = 10
 *   erf.mwr.output_file        = mwr_
 *   erf.mwr.do_brightness_temp = false   # true = also write simulated TB (35 channels)
 *   erf.mwr.move_start_time    = 60. 120. # optional moving-column intervals [s]
 *   erf.mwr.move_stop_time     = 120. 240.
 *   erf.mwr.move_speed_x       = -10. 0.  # optional physical speeds [m s-1]
 *   erf.mwr.move_speed_y       = 0. -5.
 *
 * See Docs/instrument_simulators/instrument_simulator_design.pdf for full physics derivation.
 */

#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Gpu.H>

#include <ERF_IndexDefines.H>

namespace {
    constexpr amrex::Real ERF_INSTR_PI = static_cast<amrex::Real>(3.14159265358979323846);
}

// ============================================================
//  Shared helpers
// ============================================================

namespace InstrumentSimUtil {

struct SiteLocation {
    int i;
    int j;
    amrex::Real x;
    amrex::Real y;
};

struct MoveSchedule {
    amrex::Vector<amrex::Real> start_time;
    amrex::Vector<amrex::Real> stop_time;
    amrex::Vector<amrex::Real> speed_x;
    amrex::Vector<amrex::Real> speed_y;

    void init (amrex::ParmParse& pp, const std::string& prefix)
    {
        int ns = pp.queryarr("move_start_time", start_time);
        int ne = pp.queryarr("move_stop_time",  stop_time);
        int nx = pp.queryarr("move_speed_x",    speed_x);
        int ny = pp.queryarr("move_speed_y",    speed_y);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            ns == ne, (prefix + ": move_start_time and move_stop_time must have the same number of entries").c_str());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nx == ny, (prefix + ": move_speed_x and move_speed_y must have the same number of entries").c_str());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            ns == nx, (prefix + ": move times and move speeds must have the same number of entries").c_str());

        for (int n = 0; n < ns; ++n) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                stop_time[n] > start_time[n],
                (prefix + ": move_stop_time must be greater than move_start_time").c_str());
            if (n > 0) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    start_time[n] >= stop_time[n-1],
                    (prefix + ": move intervals must not overlap").c_str());
            }
        }
    }

    amrex::Real offset_x (amrex::Real time) const
    {
        amrex::Real offset = 0.0;
        for (int n = 0; n < static_cast<int>(start_time.size()); ++n) {
            if (time > start_time[n]) {
                offset += speed_x[n] * (std::min(time, stop_time[n]) - start_time[n]);
            }
        }
        return offset;
    }

    amrex::Real offset_y (amrex::Real time) const
    {
        amrex::Real offset = 0.0;
        for (int n = 0; n < static_cast<int>(start_time.size()); ++n) {
            if (time > start_time[n]) {
                offset += speed_y[n] * (std::min(time, stop_time[n]) - start_time[n]);
            }
        }
        return offset;
    }

    bool moving () const { return !start_time.empty(); }
};

inline SiteLocation
site_location_at_time (int i0, int j0, amrex::Real time,
                       const amrex::Geometry& geom,
                       const MoveSchedule& move)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);
    const amrex::Real x = geom.ProbLo(0) + (static_cast<amrex::Real>(i0) + 0.5) * dx
                         + move.offset_x(time);
    const amrex::Real y = geom.ProbLo(1) + (static_cast<amrex::Real>(j0) + 0.5) * dy
                         + move.offset_y(time);

    const amrex::Box& dom = geom.Domain();
    int i = static_cast<int>(std::floor((x - geom.ProbLo(0)) / dx));
    int j = static_cast<int>(std::floor((y - geom.ProbLo(1)) / dy));
    i = std::max(dom.smallEnd(0), std::min(i, dom.bigEnd(0)));
    j = std::max(dom.smallEnd(1), std::min(j, dom.bigEnd(1)));

    return {i, j, x, y};
}

/** Extract a full z-column at (iloc, jloc) from a cell-centered MultiFab.
 *  Returns a host vector of size nz (= domain khi - klo + 1).
 *  GPU-safe: uses ParallelFor (runs on device where data lives), then copies
 *  result to host. Uses MPI reduce so the result is valid on IOProcessor.
 */
inline amrex::Vector<amrex::Real>
extract_column (const amrex::MultiFab& mf, int comp,
                int iloc, int jloc,
                int klo, int khi)
{
    int nz = khi - klo + 1;
    // Device-side accumulation buffer (zero-initialised)
    amrex::Gpu::DeviceVector<amrex::Real> col_d(nz, 0.0);
    amrex::Real* col_ptr = col_d.data();

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        if (bx.smallEnd(0) <= iloc && iloc <= bx.bigEnd(0) &&
            bx.smallEnd(1) <= jloc && jloc <= bx.bigEnd(1)) {
            const auto arr = mf.const_array(mfi);
            int k_lo_box = std::max(bx.smallEnd(2), klo);
            int k_hi_box = std::min(bx.bigEnd(2),   khi);
            int nk = k_hi_box - k_lo_box + 1;
            int klo_cap = klo;
            amrex::ParallelFor(nk, [=] AMREX_GPU_DEVICE (int n) noexcept {
                int k = k_lo_box + n;
                col_ptr[k - klo_cap] = arr(iloc, jloc, k, comp);
            });
            amrex::Gpu::streamSynchronize();
        }
    }

    // Copy device → host
    amrex::Vector<amrex::Real> col(nz, 0.0);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     col_d.begin(), col_d.end(), col.begin());

    amrex::ParallelDescriptor::ReduceRealSum(
        col.data(), nz, amrex::ParallelDescriptor::IOProcessorNumber());
    return col;
}

/** Extract a face-centered z-column at (iloc, jloc).
 *  GPU-safe: same pattern as extract_column.
 */
inline amrex::Vector<amrex::Real>
extract_face_column (const amrex::MultiFab& mf, int comp,
                     int iloc, int jloc,
                     int klo, int khi)
{
    int nz = khi - klo + 1;
    amrex::Gpu::DeviceVector<amrex::Real> col_d(nz, 0.0);
    amrex::Real* col_ptr = col_d.data();

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        if (bx.smallEnd(0) <= iloc && iloc <= bx.bigEnd(0) &&
            bx.smallEnd(1) <= jloc && jloc <= bx.bigEnd(1)) {
            const auto arr = mf.const_array(mfi);
            int k_lo_box = std::max(bx.smallEnd(2), klo);
            int k_hi_box = std::min(bx.bigEnd(2),   khi);
            int nk = k_hi_box - k_lo_box + 1;
            int klo_cap = klo;
            amrex::ParallelFor(nk, [=] AMREX_GPU_DEVICE (int n) noexcept {
                int k = k_lo_box + n;
                col_ptr[k - klo_cap] = arr(iloc, jloc, k, comp);
            });
            amrex::Gpu::streamSynchronize();
        }
    }

    amrex::Vector<amrex::Real> col(nz, 0.0);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     col_d.begin(), col_d.end(), col.begin());

    amrex::ParallelDescriptor::ReduceRealSum(
        col.data(), nz, amrex::ParallelDescriptor::IOProcessorNumber());
    return col;
}

/** Compute attenuated backscatter from LWC [kg/m3] and r_eff [m] profiles.
 *  Simple geometric-optics forward model.
 *  sigma_ext(z) = (3/2) * LWC / (rho_w * r_eff)
 *  beta(z)      = sigma_ext / S_lidar
 *  beta_att(z)  = beta(z) * exp(-tau(z))  where tau is two-way optical depth
 */
inline void
compute_beta_att_simple (const amrex::Vector<amrex::Real>& lwc_col,
                         const amrex::Real r_eff,
                         const amrex::Real S_lidar,
                         const amrex::Real dz,
                         amrex::Vector<amrex::Real>& beta_att_col)
{
    constexpr amrex::Real rho_water = 1000.0; // kg m-3
    int nz = static_cast<int>(lwc_col.size());
    beta_att_col.resize(nz, 0.0);

    amrex::Vector<amrex::Real> sigma(nz, 0.0);
    for (int k = 0; k < nz; ++k) {
        amrex::Real lwc = std::max(0.0, lwc_col[k]);
        sigma[k] = (r_eff > 0.0) ? 1.5 * lwc / (rho_water * r_eff) : 0.0;
    }

    // Two-way cumulative optical depth (trapezoid, upward from k=0)
    amrex::Real tau = 0.0;
    for (int k = 0; k < nz; ++k) {
        if (k > 0) {
            tau += (sigma[k] + sigma[k-1]) * dz; // two-way factor absorbed in beta def
        }
        amrex::Real beta_k = sigma[k] / S_lidar;
        beta_att_col[k] = beta_k * std::exp(-tau);
    }
}

/** Compute attenuated backscatter from pre-computed per-level extinction columns.
 *  Used when SBM bin-resolved or SDM particle-resolved sigma_ext is available.
 *  sigma_liq_col[k] and sigma_ice_col[k] are extinction coefficients [m-1].
 *  Two-way optical depth follows the same trapezoid formula as compute_beta_att_simple.
 */
inline void
compute_beta_att_from_sigma (const amrex::Vector<amrex::Real>& sigma_liq_col,
                             const amrex::Vector<amrex::Real>& sigma_ice_col,
                             const amrex::Real S_liq,
                             const amrex::Real S_ice,
                             const amrex::Real dz,
                             amrex::Vector<amrex::Real>& beta_att_col)
{
    int nz = static_cast<int>(sigma_liq_col.size());
    beta_att_col.resize(nz, 0.0);
    amrex::Real tau = 0.0; // two-way cumulative optical depth
    for (int k = 0; k < nz; ++k) {
        amrex::Real sl = std::max(0.0, sigma_liq_col[k]);
        amrex::Real si = std::max(0.0, sigma_ice_col[k]);
        if (k > 0) {
            // Trapezoid × 2 for two-way (matches compute_beta_att_simple convention)
            tau += (sl + si
                    + std::max(0.0, sigma_liq_col[k-1])
                    + std::max(0.0, sigma_ice_col[k-1])) * dz;
        }
        beta_att_col[k] = (sl / S_liq + si / S_ice) * std::exp(-tau);
    }
}

} // namespace InstrumentSimUtil

// ============================================================
//  Pre-computed size-resolved extinction data (SBM / SDM)
// ============================================================

/**
 * Carries size-resolved extinction coefficients for lidar backscatter.
 *   sigma_liq[site][k] : liquid extinction [m-1] at site s, vertical level k
 *   sigma_ice[site][k] : ice    extinction [m-1] at site s, vertical level k
 * When empty() == true (default), the simulators fall back to fixed r_eff bulk model.
 * Filled by ERF.cpp from SBM bin arrays or SDM particle positions/radii.
 */
struct InstrMicrophySizeData {
    amrex::Vector<amrex::Vector<amrex::Real>> sigma_liq; ///< [site][k] m-1
    amrex::Vector<amrex::Vector<amrex::Real>> sigma_ice; ///< [site][k] m-1
    bool empty() const { return sigma_liq.empty(); }
};

// ============================================================
//  1. CeilometerSimulator  (Vaisala CL-31, 905 nm)
// ============================================================

struct CeilometerSimulator
{
    CeilometerSimulator () = default;

    /** Parse input parameters and open output files. */
    void init (const amrex::Geometry& geom)
    {
        amrex::ParmParse pp("erf.ceilometer");

        // locations
        pp.getarr("i_loc", m_iloc);
        pp.getarr("j_loc", m_jloc);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_iloc.size() == m_jloc.size(),
            "erf.ceilometer: i_loc and j_loc must have the same number of entries");
        m_nsites = static_cast<int>(m_iloc.size());

        // validate locations are inside domain
        const amrex::Box& dom = geom.Domain();
        for (int s = 0; s < m_nsites; ++s) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                dom.smallEnd(0) <= m_iloc[s] && m_iloc[s] <= dom.bigEnd(0) &&
                dom.smallEnd(1) <= m_jloc[s] && m_jloc[s] <= dom.bigEnd(1),
                "erf.ceilometer: i_loc/j_loc is outside the domain");
        }

        pp.query("output_interval",    m_output_interval);
        pp.query("output_file",         m_output_prefix);
        pp.query("backscatter_model",   m_backscatter_model);
        pp.query("r_eff_default",       m_r_eff_default);
        pp.query("r_eff_ice_default",   m_r_eff_ice_default);
        pp.query("beta_thresh",         m_beta_thresh);
        m_move.init(pp, "erf.ceilometer");

        // open one file per site
        m_streams.resize(m_nsites);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (int s = 0; s < m_nsites; ++s) {
                std::string fname = m_output_prefix
                    + amrex::Concatenate("", s, 2) + ".txt";
                m_streams[s] = std::make_shared<std::fstream>();
                m_streams[s]->open(fname, std::ios::out);
                if (!m_streams[s]->good()) { amrex::FileOpenFailed(fname); }
                // write header
                *m_streams[s]
                    << "# ERF Ceilometer Simulator (Vaisala CL-31, 905 nm)\n"
                    << "# initial_i_loc=" << m_iloc[s] << "  initial_j_loc=" << m_jloc[s] << "\n"
                    << "# move_enabled=" << (m_move.moving() ? "true" : "false") << "\n"
                    << "# backscatter_model=" << m_backscatter_model << "\n"
                    << "# Col 1: time [s]\n"
                    << "# Col 2: x [m]\n"
                    << "# Col 3: y [m]\n"
                    << "# Col 4: i_loc\n"
                    << "# Col 5: j_loc\n"
                    << "# Col 6: height [m]\n"
                    << "# Col 7: att_backscatter [m-1 sr-1]\n"
                    << "# Col 8: cloud_base_height [m]  (-9999 = no cloud)\n";
                m_streams[s]->flush();
            }
        }
    }

    // Site-location accessors (for ERF.cpp sigma pre-computation)
    int NSites() const { return m_nsites; }
    int ILoc(int s) const { return m_iloc[s]; }
    int JLoc(int s) const { return m_jloc[s]; }
    int ILoc(int s, amrex::Real t, const amrex::Geometry& geom) const
    {
        return InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move).i;
    }
    int JLoc(int s, amrex::Real t, const amrex::Geometry& geom) const
    {
        return InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move).j;
    }
    bool is_write_step(int istep) const { return (istep % m_output_interval) == 0; }

    /** Write ceilometer data for current timestep.
     *  mpdata: optional size-resolved extinction from SBM/SDM (nullptr → fixed r_eff). */
    void write (int lev, amrex::Real t, int istep,
                const amrex::MultiFab& cons_mf,
                const amrex::MultiFab& qmoist_mf,
                const amrex::Geometry& geom,
                const InstrMicrophySizeData* mpdata = nullptr)
    {
        if ((istep % m_output_interval) != 0) { return; }

        const amrex::Box& dom = geom.Domain();
        int klo = dom.smallEnd(2);
        int khi = dom.bigEnd(2);
        int nz  = khi - klo + 1;
        amrex::Real dz = geom.CellSize(2);
        amrex::Real zlo = geom.ProbLo(2) + 0.5 * dz;

        for (int s = 0; s < m_nsites; ++s) {
            auto loc = InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move);
            int iloc = loc.i;
            int jloc = loc.j;

            // Extract rho [kg/m3] and rho*qc [kg/m3]
            auto rho_col = InstrumentSimUtil::extract_column(
                cons_mf, Rho_comp, iloc, jloc, klo, khi);
            auto rhoc_col = InstrumentSimUtil::extract_column(
                cons_mf, RhoQ2_comp, iloc, jloc, klo, khi);
            // Ice component only present for microphysics schemes with ice (e.g. SBM/SDM)
            // Kessler_NoRain has no ice → guard access with nComp() check
            amrex::Vector<amrex::Real> rhoi_col(nz, 0.0);
            if (cons_mf.nComp() > RhoQ3_comp) {
                rhoi_col = InstrumentSimUtil::extract_column(
                    cons_mf, RhoQ3_comp, iloc, jloc, klo, khi);
            }

            if (!amrex::ParallelDescriptor::IOProcessor()) { continue; }

            // LWC [kg/m3]
            amrex::Vector<amrex::Real> lwc_col(nz), iwc_col(nz);
            for (int k = 0; k < nz; ++k) {
                lwc_col[k] = std::max(0.0, rhoc_col[k]);
                iwc_col[k] = std::max(0.0, rhoi_col[k]);
            }

            // Compute attenuated backscatter
            amrex::Vector<amrex::Real> beta_att(nz, 0.0);
            const bool have_size_data = mpdata && !mpdata->empty()
                                        && s < static_cast<int>(mpdata->sigma_liq.size());
            if (have_size_data) {
                // Size-resolved path: SBM bin-summed or SDM particle-summed sigma_ext
                InstrumentSimUtil::compute_beta_att_from_sigma(
                    mpdata->sigma_liq[s], mpdata->sigma_ice[s],
                    m_S_lidar_liq_905, m_S_lidar_ice_905, dz, beta_att);
            } else if (m_backscatter_model == "mie") {
                // Mie model not yet implemented; fall back to simple bulk model
                amrex::Print() << "CeilometerSimulator: Mie model not yet implemented; "
                               << "using simple model.\n";
                InstrumentSimUtil::compute_beta_att_simple(
                    lwc_col, m_r_eff_default, m_S_lidar_liq_905, dz, beta_att);
            } else {
                // Simple geometric-optics bulk model (fixed r_eff for Kessler)
                constexpr amrex::Real rho_water = 1000.0;
                amrex::Real tau = 0.0;
                for (int k = 0; k < nz; ++k) {
                    amrex::Real sl = 1.5 * std::max(0.0, lwc_col[k]) / (rho_water * m_r_eff_default);
                    amrex::Real si = 1.5 * std::max(0.0, iwc_col[k]) / (rho_water * m_r_eff_ice_default);
                    if (k > 0) {
                        amrex::Real sl0 = 1.5 * std::max(0.0, lwc_col[k-1]) / (rho_water * m_r_eff_default);
                        amrex::Real si0 = 1.5 * std::max(0.0, iwc_col[k-1]) / (rho_water * m_r_eff_ice_default);
                        tau += (sl + si + sl0 + si0) * dz;
                    }
                    beta_att[k] = (sl / m_S_lidar_liq_905 + si / m_S_lidar_ice_905)
                                  * std::exp(-tau);
                }
            }

            // Cloud base detection
            amrex::Real z_cb = -9999.0;
            for (int k = 0; k < nz; ++k) {
                if (beta_att[k] > m_beta_thresh) {
                    z_cb = zlo + k * dz;
                    break;
                }
            }

            // Write
            auto& out = *m_streams[s];
            for (int k = 0; k < nz; ++k) {
                amrex::Real z_k = zlo + k * dz;
                out << std::scientific << std::setprecision(6)
                    << std::setw(14) << t
                    << std::setw(14) << loc.x
                    << std::setw(14) << loc.y
                    << std::setw(8)  << iloc
                    << std::setw(8)  << jloc
                    << std::setw(14) << z_k
                    << std::setw(14) << beta_att[k]
                    << std::setw(14) << z_cb
                    << "\n";
            }
            out.flush();
        }
    }

private:
    amrex::Vector<int>    m_iloc, m_jloc;
    int                   m_nsites{0};
    int                   m_output_interval{10};
    std::string           m_output_prefix{"ceil_"};
    std::string           m_backscatter_model{"simple"};
    amrex::Real           m_r_eff_default{1.0e-5};     // m (liquid)
    amrex::Real           m_r_eff_ice_default{3.0e-5}; // m (ice)
    amrex::Real           m_beta_thresh{3.0e-4};       // m-1 sr-1
    InstrumentSimUtil::MoveSchedule m_move;

    // Lidar ratios at 905 nm
    static constexpr amrex::Real m_S_lidar_liq_905{18.8}; // sr
    static constexpr amrex::Real m_S_lidar_ice_905{25.0}; // sr

    amrex::Vector<std::shared_ptr<std::fstream>> m_streams;
};

// ============================================================
//  2. DopplerLidarSimulator  (Halo StreamLine XR+, 1550 nm)
// ============================================================

struct DopplerLidarSimulator
{
    DopplerLidarSimulator () = default;

    // Site-location accessors (for ERF.cpp sigma pre-computation)
    int NSites() const { return m_nsites; }
    int ILoc(int s) const { return m_iloc[s]; }
    int JLoc(int s) const { return m_jloc[s]; }
    int ILoc(int s, amrex::Real t, const amrex::Geometry& geom) const
    {
        return InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move).i;
    }
    int JLoc(int s, amrex::Real t, const amrex::Geometry& geom) const
    {
        return InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move).j;
    }
    bool is_write_step(int istep) const { return (istep % m_output_interval) == 0; }

    void init (const amrex::Geometry& geom)
    {
        amrex::ParmParse pp("erf.doppler_lidar");

        pp.getarr("i_loc", m_iloc);
        pp.getarr("j_loc", m_jloc);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_iloc.size() == m_jloc.size(),
            "erf.doppler_lidar: i_loc and j_loc must have same number of entries");
        m_nsites = static_cast<int>(m_iloc.size());

        const amrex::Box& dom = geom.Domain();
        for (int s = 0; s < m_nsites; ++s) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                dom.smallEnd(0) <= m_iloc[s] && m_iloc[s] <= dom.bigEnd(0) &&
                dom.smallEnd(1) <= m_jloc[s] && m_jloc[s] <= dom.bigEnd(1),
                "erf.doppler_lidar: i_loc/j_loc is outside domain");
        }

        pp.query("output_interval",     m_output_interval);
        pp.query("output_file",          m_output_prefix);
        pp.query("backscatter_model",    m_backscatter_model);
        pp.query("do_vertical_stare",    m_do_stare);
        pp.query("do_vad",               m_do_vad);
        pp.query("vad_elevation_angle",  m_vad_elevation_angle);
        pp.query("do_backscatter",       m_do_backscatter);
        pp.query("r_eff_default",        m_r_eff_default);
        m_move.init(pp, "erf.doppler_lidar");

        // CNR model: "normalized" (SNR = beta_att/r^2, default) or
        //            "physical"   (CNR = F(r)*beta_att, Targ et al. 1991)
        pp.query("cnr_model",            m_cnr_model);
        if (m_cnr_model == "physical") {
            pp.query("pulse_energy",     m_pulse_energy);
            pp.query("optics_efficiency",m_optics_eff);
            pp.query("beam_radius",      m_beam_radius);
            pp.query("bandwidth",        m_bandwidth);
            pp.query("focus_range",      m_focus_range);
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (int s = 0; s < m_nsites; ++s) {
                std::string tag = amrex::Concatenate("", s, 2);
                if (m_do_stare) {
                    std::string cnr_hdr = (m_cnr_model == "physical")
                        ? "# cnr_model=physical  E=" + std::to_string(m_pulse_energy)
                          + " J  eta=" + std::to_string(m_optics_eff)
                          + "  omega=" + std::to_string(m_beam_radius)
                          + " m  Bw=" + std::to_string(m_bandwidth)
                          + " Hz  R=" + std::to_string(m_focus_range) + " m\n"
                        : "# cnr_model=normalized  (CNR = beta_att / r^2)\n";
                    m_stare_streams.push_back(open_stream(
                        m_output_prefix + tag + "_stare.txt",
                        "# ERF Doppler Lidar Simulator (Halo XR+, 1550 nm) -- vertical stare\n"
                        "# initial_i_loc=" + std::to_string(m_iloc[s]) +
                        "  initial_j_loc=" + std::to_string(m_jloc[s]) + "\n"
                        "# move_enabled=" + std::string(m_move.moving() ? "true" : "false") + "\n"
                        + cnr_hdr +
                        "# Col 1: time [s]\n"
                        "# Col 2: x [m]\n"
                        "# Col 3: y [m]\n"
                        "# Col 4: i_loc\n"
                        "# Col 5: j_loc\n"
                        "# Col 6: height [m]\n"
                        "# Col 7: w [m s-1]\n"
                        "# Col 8: CNR [-]\n"));
                }
                if (m_do_vad) {
                    m_vad_streams.push_back(open_stream(
                        m_output_prefix + tag + "_vad.txt",
                        "# ERF Doppler Lidar Simulator -- VAD mode (elev="
                        + std::to_string(m_vad_elevation_angle) + " deg, 36 azimuths)\n"
                        "# initial_i_loc=" + std::to_string(m_iloc[s]) +
                        "  initial_j_loc=" + std::to_string(m_jloc[s]) + "\n"
                        "# move_enabled=" + std::string(m_move.moving() ? "true" : "false") + "\n"
                        "# Col 1: time [s]\n"
                        "# Col 2: x [m]\n"
                        "# Col 3: y [m]\n"
                        "# Col 4: i_loc\n"
                        "# Col 5: j_loc\n"
                        "# Col 6: height [m]\n"
                        "# Col 7: u [m s-1]\n"
                        "# Col 8: v [m s-1]\n"
                        "# Col 9: wind_speed [m s-1]\n"
                        "# Col 10: wind_dir [deg, met convention]\n"));
                }
                if (m_do_backscatter) {
                    m_beta_streams.push_back(open_stream(
                        m_output_prefix + tag + "_beta.txt",
                        "# ERF Doppler Lidar Simulator -- att. backscatter (1550 nm)\n"
                        "# initial_i_loc=" + std::to_string(m_iloc[s]) +
                        "  initial_j_loc=" + std::to_string(m_jloc[s]) + "\n"
                        "# move_enabled=" + std::string(m_move.moving() ? "true" : "false") + "\n"
                        "# Col 1: time [s]\n"
                        "# Col 2: x [m]\n"
                        "# Col 3: y [m]\n"
                        "# Col 4: i_loc\n"
                        "# Col 5: j_loc\n"
                        "# Col 6: height [m]\n"
                        "# Col 7: beta_att [m-1 sr-1]\n"));
                }
            }
        }
    }

    /** Write Doppler lidar data for current timestep.
     *  mpdata: optional size-resolved extinction from SBM/SDM (nullptr → fixed r_eff). */
    void write (int lev, amrex::Real t, int istep,
                const amrex::MultiFab& cons_mf,
                const amrex::MultiFab& xvel_mf,
                const amrex::MultiFab& yvel_mf,
                const amrex::MultiFab& zvel_mf,
                const amrex::MultiFab& qmoist_mf,
                const amrex::Geometry& geom,
                const InstrMicrophySizeData* mpdata = nullptr)
    {
        if ((istep % m_output_interval) != 0) { return; }

        const amrex::Box& dom = geom.Domain();
        int klo = dom.smallEnd(2);
        int khi = dom.bigEnd(2);
        int nz  = khi - klo + 1;
        amrex::Real dz  = geom.CellSize(2);
        amrex::Real zlo = geom.ProbLo(2) + 0.5 * dz;

        int stare_idx = 0, vad_idx = 0, beta_idx = 0;
        for (int s = 0; s < m_nsites; ++s) {
            auto loc = InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move);
            int iloc = loc.i;
            int jloc = loc.j;

            // Extract columns (always on all ranks)
            amrex::Vector<amrex::Real> u_col, v_col, w_col;

            if (m_do_stare || m_do_vad || m_do_backscatter) {
                // w at face (i, j, k+1/2) -- average two faces for cell-center w
                auto w_lo = InstrumentSimUtil::extract_face_column(
                    zvel_mf, 0, iloc, jloc, klo, khi);
                auto w_hi = InstrumentSimUtil::extract_face_column(
                    zvel_mf, 0, iloc, jloc, klo+1, khi+1);
                w_col.resize(nz);
                for (int k = 0; k < nz; ++k) {
                    w_col[k] = 0.5 * (w_lo[k] + w_hi[k]);
                }
            }

            if (m_do_vad) {
                // u at face (i+1/2, j, k) -- use iloc face
                auto u_lo = InstrumentSimUtil::extract_face_column(
                    xvel_mf, 0, iloc, jloc, klo, khi);
                auto u_hi = InstrumentSimUtil::extract_face_column(
                    xvel_mf, 0, iloc+1, jloc, klo, khi);
                u_col.resize(nz);
                for (int k = 0; k < nz; ++k) {
                    u_col[k] = 0.5 * (u_lo[k] + u_hi[k]);
                }

                auto v_lo = InstrumentSimUtil::extract_face_column(
                    yvel_mf, 0, iloc, jloc, klo, khi);
                auto v_hi = InstrumentSimUtil::extract_face_column(
                    yvel_mf, 0, iloc, jloc+1, klo, khi);
                v_col.resize(nz);
                for (int k = 0; k < nz; ++k) {
                    v_col[k] = 0.5 * (v_lo[k] + v_hi[k]);
                }
            }

            amrex::Vector<amrex::Real> beta_att;
            if (m_do_backscatter) {
                const bool have_size_data = mpdata && !mpdata->empty()
                                            && s < static_cast<int>(mpdata->sigma_liq.size());
                if (have_size_data) {
                    if (amrex::ParallelDescriptor::IOProcessor()) {
                        InstrumentSimUtil::compute_beta_att_from_sigma(
                            mpdata->sigma_liq[s], mpdata->sigma_ice[s],
                            m_S_lidar_liq_1550, m_S_lidar_ice_1550, dz, beta_att);
                    }
                } else {
                    auto rhoc_col = InstrumentSimUtil::extract_column(
                        cons_mf, RhoQ2_comp, iloc, jloc, klo, khi);
                    amrex::Vector<amrex::Real> rhoi_col(nz, 0.0);
                    if (cons_mf.nComp() > RhoQ3_comp) {
                        rhoi_col = InstrumentSimUtil::extract_column(
                            cons_mf, RhoQ3_comp, iloc, jloc, klo, khi);
                    }
                    if (amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Vector<amrex::Real> lwc(nz), iwc(nz);
                        for (int k = 0; k < nz; ++k) {
                            lwc[k] = std::max(0.0, rhoc_col[k]);
                            iwc[k] = std::max(0.0, rhoi_col[k]);
                        }
                        compute_beta_att_1550(lwc, iwc, dz, beta_att);
                    }
                }
            }

            if (!amrex::ParallelDescriptor::IOProcessor()) { continue; }

            // Vertical stare output
            if (m_do_stare) {
                auto& out = *m_stare_streams[stare_idx];
                for (int k = 0; k < nz; ++k) {
                    amrex::Real z_k = zlo + k * dz;
                    amrex::Real cnr = 0.0;
                    if (m_do_backscatter && !beta_att.empty()) {
                        if (m_cnr_model == "physical") {
                            cnr = compute_cnr(beta_att[k], z_k);
                        } else {
                            // normalized: CNR = beta_att / r^2
                            cnr = beta_att[k] / (z_k * z_k);
                        }
                    }
                    out << std::scientific << std::setprecision(6)
                        << std::setw(14) << t
                        << std::setw(14) << loc.x
                        << std::setw(14) << loc.y
                        << std::setw(8)  << iloc
                        << std::setw(8)  << jloc
                        << std::setw(14) << z_k
                        << std::setw(14) << w_col[k]
                        << std::setw(14) << cnr
                        << "\n";
                }
                out.flush();
                ++stare_idx;
            }

            // VAD retrieval output
            if (m_do_vad) {
                amrex::Real phi = m_vad_elevation_angle * ERF_INSTR_PI / 180.0;
                amrex::Real cos_phi = std::cos(phi);
                amrex::Real sin_phi = std::sin(phi);
                constexpr int N_az = 36;
                amrex::Real d_psi = 2.0 * ERF_INSTR_PI / N_az;

                auto& out = *m_vad_streams[vad_idx];
                for (int k = 0; k < nz; ++k) {
                    amrex::Real z_k = zlo + k * dz;
                    amrex::Real uk = u_col[k], vk = v_col[k], wk = w_col[k];

                    // Simulate N_az radial velocity observations
                    amrex::Vector<amrex::Real> Vr(N_az);
                    for (int m = 0; m < N_az; ++m) {
                        amrex::Real psi = m * d_psi;
                        Vr[m] = uk * cos_phi * std::cos(psi)
                              + vk * cos_phi * std::sin(psi)
                              + wk * sin_phi;
                    }

                    // Least-squares sinusoidal fit: Vr = A0 + A1*cos(psi) + B1*sin(psi)
                    amrex::Real A0 = 0.0, A1 = 0.0, B1 = 0.0;
                    for (int m = 0; m < N_az; ++m) {
                        amrex::Real psi = m * d_psi;
                        A0 += Vr[m];
                        A1 += Vr[m] * std::cos(psi);
                        B1 += Vr[m] * std::sin(psi);
                    }
                    A0 /= N_az;
                    A1 *= 2.0 / N_az;
                    B1 *= 2.0 / N_az;

                    amrex::Real u_ret = (std::abs(cos_phi) > 1e-8) ? A1 / cos_phi : 0.0;
                    amrex::Real v_ret = (std::abs(cos_phi) > 1e-8) ? B1 / cos_phi : 0.0;
                    amrex::Real wspd = std::sqrt(u_ret*u_ret + v_ret*v_ret);
                    amrex::Real wdir = std::atan2(-u_ret, -v_ret) * 180.0 / ERF_INSTR_PI + 180.0;

                    out << std::scientific << std::setprecision(6)
                        << std::setw(14) << t
                        << std::setw(14) << loc.x
                        << std::setw(14) << loc.y
                        << std::setw(8)  << iloc
                        << std::setw(8)  << jloc
                        << std::setw(14) << z_k
                        << std::setw(14) << u_ret
                        << std::setw(14) << v_ret
                        << std::setw(14) << wspd
                        << std::setw(14) << wdir
                        << "\n";
                }
                out.flush();
                ++vad_idx;
            }

            // Backscatter output
            if (m_do_backscatter) {
                auto& out = *m_beta_streams[beta_idx];
                for (int k = 0; k < nz; ++k) {
                    amrex::Real z_k = zlo + k * dz;
                    out << std::scientific << std::setprecision(6)
                        << std::setw(14) << t
                        << std::setw(14) << loc.x
                        << std::setw(14) << loc.y
                        << std::setw(8)  << iloc
                        << std::setw(8)  << jloc
                        << std::setw(14) << z_k
                        << std::setw(14) << beta_att[k]
                        << "\n";
                }
                out.flush();
                ++beta_idx;
            }
        } // sites
    }

private:
    amrex::Vector<int>    m_iloc, m_jloc;
    int                   m_nsites{0};
    int                   m_output_interval{10};
    std::string           m_output_prefix{"dl_"};
    std::string           m_backscatter_model{"simple"};
    bool                  m_do_stare{true};
    bool                  m_do_vad{false};
    bool                  m_do_backscatter{false};
    amrex::Real           m_vad_elevation_angle{75.0}; // degrees
    amrex::Real           m_r_eff_default{1.0e-5};     // m
    InstrumentSimUtil::MoveSchedule m_move;

    // CNR model selection and instrument parameters (Targ et al. 1991)
    std::string           m_cnr_model{"normalized"};   // "normalized" or "physical"
    amrex::Real           m_pulse_energy{1.5e-6};      // E  [J]   Halo XR+ nominal
    amrex::Real           m_optics_eff{0.1};           // eta [-]  optical/detector efficiency
    amrex::Real           m_beam_radius{0.0225};       // omega [m] 1/e^2 intensity radius
    amrex::Real           m_bandwidth{50.0e6};         // B_w [Hz] receiver bandwidth
    amrex::Real           m_focus_range{1.0e9};        // R [m]   radius of curvature (1e9 = collimated)

    static constexpr amrex::Real m_S_lidar_liq_1550{20.0}; // sr
    static constexpr amrex::Real m_S_lidar_ice_1550{28.0}; // sr
    static constexpr amrex::Real m_r_eff_ice_default{3.0e-5};

    amrex::Vector<std::shared_ptr<std::fstream>> m_stare_streams;
    amrex::Vector<std::shared_ptr<std::fstream>> m_vad_streams;
    amrex::Vector<std::shared_ptr<std::fstream>> m_beta_streams;

    void compute_beta_att_1550 (const amrex::Vector<amrex::Real>& lwc,
                                const amrex::Vector<amrex::Real>& iwc,
                                amrex::Real dz,
                                amrex::Vector<amrex::Real>& beta_att) const
    {
        constexpr amrex::Real rho_w = 1000.0;
        int nz = static_cast<int>(lwc.size());
        beta_att.resize(nz, 0.0);
        amrex::Vector<amrex::Real> sigma(nz, 0.0);
        for (int k = 0; k < nz; ++k) {
            sigma[k] = 1.5 * lwc[k] / (rho_w * m_r_eff_default)
                     + 1.5 * iwc[k] / (rho_w * m_r_eff_ice_default);
        }
        amrex::Real tau = 0.0;
        for (int k = 0; k < nz; ++k) {
            if (k > 0) { tau += (sigma[k] + sigma[k-1]) * dz; }
            amrex::Real beta_k = lwc[k] / (rho_w * m_r_eff_default * m_S_lidar_liq_1550)
                               + iwc[k] / (rho_w * m_r_eff_ice_default * m_S_lidar_ice_1550);
            beta_k *= 1.5;
            beta_att[k] = beta_k * std::exp(-tau);
        }
    }

    /**
     * Compute CNR at range r [m] given attenuated backscatter beta_att [m-1 sr-1].
     *
     * cnr_model = "normalized":
     *   CNR = beta_att / r^2
     *
     * cnr_model = "physical"  (Targ et al. 1991, Eq. 1-2):
     *   CNR(r) = F(r) * beta_att(r)
     *   F(r)   = (E * eta * lambda * pi * omega^2) / (2 * h * Bw * r^2)
     *            * [1 + (1 - r/R)^2 * (pi*omega^2 / (lambda*r))^2]^{-1}
     *
     *   Parameters (user-configurable via inputs file):
     *     E     = pulse_energy      [J]    pulse energy
     *     eta   = optics_efficiency [-]    optical + detector efficiency
     *     omega = beam_radius       [m]    1/e^2 intensity radius of Gaussian beam at aperture
     *     Bw    = bandwidth         [Hz]   receiver noise bandwidth
     *     R     = focus_range       [m]    radius of curvature of wavefront at aperture
     *                                      (R -> infinity for collimated beam)
     *     lambda = 1.55e-6 m (fixed, 1550 nm Halo XR+)
     *     h      = 6.62607e-34 J*s (Planck constant, fixed)
     */
    amrex::Real compute_cnr (amrex::Real beta_att, amrex::Real r) const
    {
        if (r <= 0.0) { return 0.0; }
        if (m_cnr_model == "physical") {
            constexpr amrex::Real lambda  = 1.55e-6;           // m (1550 nm)
            constexpr amrex::Real h_planck = 6.62607015e-34;   // J*s
            // Geometric term (same as normalized, scaled by instrument constants)
            amrex::Real F_geo = (m_pulse_energy * m_optics_eff
                                 * lambda * ERF_INSTR_PI * m_beam_radius * m_beam_radius)
                                / (2.0 * h_planck * m_bandwidth * r * r);
            // Gaussian beam focus correction factor (Targ et al. 1991, denominator of Eq. 2)
            //   [1 + (1 - r/R)^2 * (pi*omega^2 / (lambda*r))^2]^{-1}
            amrex::Real rayleigh_arg  = (ERF_INSTR_PI * m_beam_radius * m_beam_radius)
                                        / (lambda * r);
            amrex::Real focus_factor  = 1.0 - r / m_focus_range;
            amrex::Real focus_corr    = 1.0 / (1.0 + focus_factor * focus_factor
                                                      * rayleigh_arg * rayleigh_arg);
            return F_geo * focus_corr * beta_att;
        } else {
            // normalized: dimensionless proxy, suitable for qualitative comparison
            return beta_att / (r * r);
        }
    }

    std::shared_ptr<std::fstream>
    open_stream (const std::string& fname, const std::string& header) const
    {
        auto ptr = std::make_shared<std::fstream>();
        ptr->open(fname, std::ios::out);
        if (!ptr->good()) { amrex::FileOpenFailed(fname); }
        *ptr << header;
        ptr->flush();
        return ptr;
    }
};

// ============================================================
//  3. MWRSimulator  (Radiometrics MP-3000A, K+V band)
// ============================================================

struct MWRSimulator
{
    MWRSimulator () = default;

    void init (const amrex::Geometry& geom)
    {
        amrex::ParmParse pp("erf.mwr");

        pp.getarr("i_loc", m_iloc);
        pp.getarr("j_loc", m_jloc);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_iloc.size() == m_jloc.size(),
            "erf.mwr: i_loc and j_loc must have same number of entries");
        m_nsites = static_cast<int>(m_iloc.size());

        const amrex::Box& dom = geom.Domain();
        for (int s = 0; s < m_nsites; ++s) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                dom.smallEnd(0) <= m_iloc[s] && m_iloc[s] <= dom.bigEnd(0) &&
                dom.smallEnd(1) <= m_jloc[s] && m_jloc[s] <= dom.bigEnd(1),
                "erf.mwr: i_loc/j_loc is outside domain");
        }

        pp.query("output_interval",    m_output_interval);
        pp.query("output_file",         m_output_prefix);
        pp.query("do_brightness_temp",  m_do_tb);
        m_move.init(pp, "erf.mwr");

        // MP-3000A channel frequencies (35 channels)
        m_freqs = {
            // K-band (22 channels)
            22.235, 22.500, 23.035, 23.835, 25.000, 26.235, 28.000, 30.000,
            31.400, 32.000, 32.400, 33.000, 34.000, 35.000, 36.000, 36.500,
            37.000, 37.500, 38.000, 38.600, 39.500, 40.000,
            // V-band (13 channels)
            51.250, 51.760, 52.280, 52.800, 53.340, 53.860, 54.400,
            54.940, 55.500, 56.020, 56.660, 57.290, 58.000
        };

        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (int s = 0; s < m_nsites; ++s) {
                std::string tag = amrex::Concatenate("", s, 2);

                m_prof_streams.push_back(open_stream(
                    m_output_prefix + tag + "_profiles.txt",
                    "# ERF MWR Simulator (Radiometrics MP-3000A)\n"
                    "# initial_i_loc=" + std::to_string(m_iloc[s]) +
                    "  initial_j_loc=" + std::to_string(m_jloc[s]) + "\n"
                    "# move_enabled=" + std::string(m_move.moving() ? "true" : "false") + "\n"
                    "# Col 1: time [s]\n"
                    "# Col 2: x [m]\n"
                    "# Col 3: y [m]\n"
                    "# Col 4: i_loc\n"
                    "# Col 5: j_loc\n"
                    "# Col 6: height [m]\n"
                    "# Col 7: T [K]\n"
                    "# Col 8: qv [g kg-1]\n"
                    "# Col 9: qc [g kg-1]\n"));

                m_col_streams.push_back(open_stream(
                    m_output_prefix + tag + "_column.txt",
                    "# ERF MWR Simulator -- column integrals\n"
                    "# initial_i_loc=" + std::to_string(m_iloc[s]) +
                    "  initial_j_loc=" + std::to_string(m_jloc[s]) + "\n"
                    "# move_enabled=" + std::string(m_move.moving() ? "true" : "false") + "\n"
                    "# Col 1: time [s]\n"
                    "# Col 2: x [m]\n"
                    "# Col 3: y [m]\n"
                    "# Col 4: i_loc\n"
                    "# Col 5: j_loc\n"
                    "# Col 6: LWP [g m-2]\n"
                    "# Col 7: PWV [mm]\n"));

                if (m_do_tb) {
                    m_tb_streams.push_back(open_stream(
                        m_output_prefix + tag + "_tb.txt",
                        "# ERF MWR Simulator -- simulated brightness temperatures (35 channels)\n"
                        "# initial_i_loc=" + std::to_string(m_iloc[s]) +
                        "  initial_j_loc=" + std::to_string(m_jloc[s]) + "\n"
                        "# move_enabled=" + std::string(m_move.moving() ? "true" : "false") + "\n"
                        "# Col 1: time [s]\n"
                        "# Col 2: x [m]\n"
                        "# Col 3: y [m]\n"
                        "# Col 4: i_loc\n"
                        "# Col 5: j_loc\n"
                        "# Col 6: frequency [GHz]\n"
                        "# Col 7: TB [K]\n"));
                }
            }
        }
    }

    void write (int lev, amrex::Real t, int istep,
                const amrex::MultiFab& cons_mf,
                const amrex::MultiFab& base_state_mf,
                const amrex::MultiFab& qmoist_mf,
                const amrex::Geometry& geom)
    {
        if ((istep % m_output_interval) != 0) { return; }

        const amrex::Box& dom = geom.Domain();
        int klo = dom.smallEnd(2);
        int khi = dom.bigEnd(2);
        int nz  = khi - klo + 1;
        amrex::Real dz  = geom.CellSize(2);
        amrex::Real zlo = geom.ProbLo(2) + 0.5 * dz;

        // Physical constants
        constexpr amrex::Real Rd    = 287.04;   // J kg-1 K-1
        constexpr amrex::Real cp    = 1004.5;   // J kg-1 K-1
        constexpr amrex::Real rdOcp = Rd / cp;
        constexpr amrex::Real p_ref = 1.0e5;    // Pa
        constexpr amrex::Real rho_w = 1000.0;   // kg m-3

        for (int s = 0; s < m_nsites; ++s) {
            auto loc = InstrumentSimUtil::site_location_at_time(m_iloc[s], m_jloc[s], t, geom, m_move);
            int iloc = loc.i;
            int jloc = loc.j;

            // Extract state columns
            auto rho_col    = InstrumentSimUtil::extract_column(
                cons_mf, Rho_comp,       iloc, jloc, klo, khi);
            auto rhoth_col  = InstrumentSimUtil::extract_column(
                cons_mf, RhoTheta_comp,  iloc, jloc, klo, khi);
            auto rhoqv_col  = InstrumentSimUtil::extract_column(
                cons_mf, RhoQ1_comp,     iloc, jloc, klo, khi);
            auto rhoqc_col  = InstrumentSimUtil::extract_column(
                cons_mf, RhoQ2_comp,     iloc, jloc, klo, khi);
            // Base-state pressure p0 (1D, i/j irrelevant but MultiFab is 3D)
            auto p0_col     = InstrumentSimUtil::extract_column(
                base_state_mf, BaseState::p0_comp, iloc, jloc, klo, khi);

            if (!amrex::ParallelDescriptor::IOProcessor()) { continue; }

            // Derived profiles
            amrex::Vector<amrex::Real> T_col(nz), qv_col(nz), qc_col(nz);
            amrex::Real lwp = 0.0, pwv = 0.0;

            for (int k = 0; k < nz; ++k) {
                amrex::Real rho   = rho_col[k];
                amrex::Real rhoth = rhoth_col[k];
                amrex::Real p     = p0_col[k];  // base state pressure [Pa]

                // Potential temperature
                amrex::Real theta = (rho > 0.0) ? rhoth / rho : 300.0;

                // Exner function from base-state pressure
                amrex::Real pi_ex = std::pow(p / p_ref, rdOcp);

                // Absolute temperature
                T_col[k]  = theta * pi_ex;

                // Mixing ratios [kg/kg] -> [g/kg]
                qv_col[k] = (rho > 0.0) ? rhoqv_col[k] / rho * 1000.0 : 0.0;
                qc_col[k] = (rho > 0.0) ? std::max(0.0, rhoqc_col[k] / rho * 1000.0) : 0.0;

                // Column integrals
                lwp += std::max(0.0, rhoqc_col[k]) * dz;   // kg m-2
                pwv += std::max(0.0, rhoqv_col[k]) * dz;   // kg m-2
            }
            lwp *= 1000.0; // g m-2
            pwv /= rho_w;  // mm

            // Profile file
            {
                auto& out = *m_prof_streams[s];
                for (int k = 0; k < nz; ++k) {
                    amrex::Real z_k = zlo + k * dz;
                    out << std::scientific << std::setprecision(6)
                        << std::setw(14) << t
                        << std::setw(14) << loc.x
                        << std::setw(14) << loc.y
                        << std::setw(8)  << iloc
                        << std::setw(8)  << jloc
                        << std::setw(14) << z_k
                        << std::setw(14) << T_col[k]
                        << std::setw(14) << qv_col[k]
                        << std::setw(14) << qc_col[k]
                        << "\n";
                }
                out.flush();
            }

            // Column integrals file
            {
                auto& out = *m_col_streams[s];
                out << std::scientific << std::setprecision(6)
                    << std::setw(14) << t
                    << std::setw(14) << loc.x
                    << std::setw(14) << loc.y
                    << std::setw(8)  << iloc
                    << std::setw(8)  << jloc
                    << std::setw(14) << lwp
                    << std::setw(14) << pwv
                    << "\n";
                out.flush();
            }

            // Brightness temperatures (optional)
            if (m_do_tb && !m_tb_streams.empty()) {
                auto& out = *m_tb_streams[s];
                // Build pressure column from p0 and qv for RT calculation
                amrex::Vector<amrex::Real> p_col(nz), rho_kgm3(nz), qv_kgkg(nz), lwc_kgm3(nz);
                for (int k = 0; k < nz; ++k) {
                    p_col[k]    = p0_col[k];
                    rho_kgm3[k] = rho_col[k];
                    qv_kgkg[k]  = std::max(0.0, rhoqv_col[k] / rho_col[k]);
                    lwc_kgm3[k] = std::max(0.0, rhoqc_col[k]);
                }
                for (amrex::Real freq : m_freqs) {
                    amrex::Real TB = compute_TB(
                        T_col, p_col, rho_kgm3, qv_kgkg, lwc_kgm3, dz, freq);
                    out << std::scientific << std::setprecision(6)
                        << std::setw(14) << t
                        << std::setw(14) << loc.x
                        << std::setw(14) << loc.y
                        << std::setw(8)  << iloc
                        << std::setw(8)  << jloc
                        << std::setw(14) << freq
                        << std::setw(14) << TB
                        << "\n";
                }
                out.flush();
            }
        } // sites
    }

private:
    amrex::Vector<int>    m_iloc, m_jloc;
    int                   m_nsites{0};
    int                   m_output_interval{10};
    std::string           m_output_prefix{"mwr_"};
    bool                  m_do_tb{false};
    InstrumentSimUtil::MoveSchedule m_move;
    amrex::Vector<amrex::Real> m_freqs; // GHz

    amrex::Vector<std::shared_ptr<std::fstream>> m_prof_streams;
    amrex::Vector<std::shared_ptr<std::fstream>> m_col_streams;
    amrex::Vector<std::shared_ptr<std::fstream>> m_tb_streams;

    /** Compute zenith brightness temperature at frequency f [GHz] using
     *  simplified plane-parallel radiative transfer.
     *
     *  Absorption: H2O line at 22.235 GHz (Rosenkranz 1998 simplified),
     *              H2O continuum, O2 complex near 60 GHz (Liebe 1993 simplified),
     *              cloud liquid (Rayleigh, Ellison 2007 permittivity).
     *  RT: TB(f) = T_bg * exp(-tau_tot) + sum_k T_k * [exp(-tau_{k-1}) - exp(-tau_k)]
     */
    amrex::Real compute_TB (
        const amrex::Vector<amrex::Real>& T_K,
        const amrex::Vector<amrex::Real>& p_Pa,
        const amrex::Vector<amrex::Real>& rho,
        const amrex::Vector<amrex::Real>& qv,   // kg/kg
        const amrex::Vector<amrex::Real>& lwc,  // kg/m3
        amrex::Real dz,
        amrex::Real f_GHz) const
    {
        constexpr amrex::Real T_bg = 2.75; // K, CMB
        int nz = static_cast<int>(T_K.size());

        // Cumulative optical depth from surface upward (tau[k] = tau from 0 to k)
        amrex::Vector<amrex::Real> tau(nz + 1, 0.0);
        for (int k = 0; k < nz; ++k) {
            amrex::Real alpha_k = absorption_coeff(T_K[k], p_Pa[k], rho[k], qv[k], lwc[k], f_GHz);
            tau[k + 1] = tau[k] + alpha_k * dz; // Np (nepers)
        }

        // Discrete RT summation (downwelling: from top to surface, then emit upward)
        // For zenith-pointing: integrate from surface upward
        amrex::Real TB = T_bg * std::exp(-tau[nz]);
        for (int k = 0; k < nz; ++k) {
            TB += T_K[k] * (std::exp(-tau[k]) - std::exp(-tau[k + 1]));
        }
        return TB;
    }

    /** Total volume absorption coefficient [Np/m] at frequency f [GHz].
     *  Includes H2O line + continuum, O2 complex, and cloud liquid.
     */
    amrex::Real absorption_coeff (amrex::Real T, amrex::Real p,
                                   amrex::Real rho_air, amrex::Real qv,
                                   amrex::Real lwc, amrex::Real f) const
    {
        // Water vapor density [g/m3]
        amrex::Real rho_v = qv * rho_air * 1000.0;

        amrex::Real alpha = alpha_H2O(T, p, rho_v, f)
                          + alpha_O2(T, p, f)
                          + alpha_liquid(T, lwc, f);
        return alpha;
    }

    /** H2O absorption [Np/m] -- simplified Rosenkranz (1998).
     *  Dominant line at 22.235 GHz + continuum.
     *  f in GHz, T in K, p in Pa, rho_v in g/m3.
     */
    amrex::Real alpha_H2O (amrex::Real T, amrex::Real p,
                            amrex::Real rho_v, amrex::Real f) const
    {
        if (rho_v <= 0.0) { return 0.0; }
        constexpr amrex::Real f0 = 22.235; // GHz, main K-band line
        amrex::Real theta = 300.0 / T;
        amrex::Real p_hPa = p / 100.0;

        // Line strength and pressure-broadened width (Rosenkranz 1998 Table 1)
        amrex::Real S  = 0.01813 * std::exp(2.143 * (1.0 - theta)) * rho_v * theta * theta;
        amrex::Real gamma = 0.002784 * p_hPa * std::pow(theta, 0.626)
                           + 0.02310 * rho_v * std::pow(theta, 0.624);

        // Van Vleck-Weisskopf lineshape [GHz-1]
        amrex::Real df_lo = (f - f0) * (f - f0) + gamma * gamma;
        amrex::Real df_hi = (f + f0) * (f + f0) + gamma * gamma;
        amrex::Real F = (f / f0) * (gamma / df_lo + gamma / df_hi);

        // Continuum (Rosenkranz 1998 Eq. 2)
        amrex::Real cont = 3.57e-7 * p_hPa * theta * theta * rho_v
                         + 4.74e-7 * p_hPa * theta * p_hPa * theta * rho_v;

        // Convert: S * F [Np / (m GHz) * GHz] + cont [Np/m per unit?]
        // Units: S [Np cm-1 GHz-1 (g/m3)-1] * rho_v [g/m3] * F [GHz] -> Np cm-1
        // Following Rosenkranz, multiply by 0.01 for cm->m, multiply by f^2
        amrex::Real alpha_line = S * F / 100.0;     // Np/m approx
        amrex::Real alpha_cont = cont * f * f / 100.0; // Np/m approx

        return alpha_line + alpha_cont;
    }

    /** O2 absorption [Np/m] -- simplified Liebe (1993) resonance sum near 60 GHz.
     *  f in GHz, T in K, p in Pa.
     */
    amrex::Real alpha_O2 (amrex::Real T, amrex::Real p, amrex::Real f) const
    {
        amrex::Real theta = 300.0 / T;
        amrex::Real p_hPa = p / 100.0;

        // O2 mixing ratio = 0.2095, dry air partial pressure
        amrex::Real p_dry = p_hPa;

        // Simplified: represent O2 complex with a few dominant lines
        // Frequencies and strengths from Liebe (1993) Table 1 (subset)
        static const amrex::Real f0_o2[] = {
            50.474, 50.988, 51.503, 52.021, 52.542, 53.067,
            53.597, 54.130, 54.671, 55.221, 55.784, 56.265,
            56.968, 57.612, 58.324, 58.447, 59.591, 60.306,
            60.435, 61.151, 61.800, 62.411, 62.486, 62.998,
            63.568, 64.128, 64.678, 65.224, 65.765, 66.302
        };
        static const amrex::Real a1[] = {
            0.094, 0.246, 0.608, 1.414, 3.102, 6.410,
            12.40, 22.33, 37.19, 57.06, 80.60, 97.42,
            127.5, 152.1, 154.2, 153.6, 129.8, 109.6,
            106.4, 88.24, 68.46, 49.47, 48.81, 33.44,
            22.20, 14.36,  9.14,  5.71,  3.51,  2.12
        };

        amrex::Real alpha = 0.0;
        for (int i = 0; i < 30; ++i) {
            amrex::Real df = f0_o2[i];
            // Pressure-broadened line width
            amrex::Real gamma = 0.00089 * p_dry * std::pow(theta, 0.8);
            amrex::Real df2_lo = (f - df) * (f - df) + gamma * gamma;
            amrex::Real df2_hi = (f + df) * (f + df) + gamma * gamma;
            amrex::Real F = (gamma / df2_lo + gamma / df2_hi);
            // Line strength (proportional to a1 * theta^3 * exp(-a2*(1-theta)))
            amrex::Real S = a1[i] * 1.0e-7 * p_dry * theta * theta * theta;
            alpha += S * F;
        }
        return alpha * 0.01; // convert cm-1 -> m-1 approx
    }

    /** Cloud liquid water absorption [Np/m] -- Rayleigh approximation.
     *  f in GHz, T in K, lwc in kg/m3.
     */
    amrex::Real alpha_liquid (amrex::Real T, amrex::Real lwc, amrex::Real f) const
    {
        if (lwc <= 0.0) { return 0.0; }
        constexpr amrex::Real c_GHz = 2.998e10; // cm/s
        amrex::Real f_Hz = f * 1.0e9;
        amrex::Real omega = 2.0 * M_PI * f_Hz;

        // Ellison (2007) double-Debye permittivity for liquid water
        // Parameters at T in Celsius
        amrex::Real Tc = T - 273.15;
        amrex::Real eps_s = 87.85306 * std::exp(-0.00456992 * Tc);
        amrex::Real eps_1 = 5.7230 + 0.021999 * Tc;
        amrex::Real eps_inf = 3.747;
        amrex::Real tau1_s  = (1.0 / (2.0 * M_PI)) *
            std::exp(19.863 - 0.04441 * Tc) * 1.0e-12; // s
        amrex::Real tau2_s  = (1.0 / (2.0 * M_PI)) *
            std::exp(16.166 - 0.004178 * Tc) * 1.0e-12; // s

        // Complex permittivity (imaginary part only, relevant for absorption)
        amrex::Real chi1 = omega * tau1_s;
        amrex::Real chi2 = omega * tau2_s;
        amrex::Real eps_imag = (eps_s  - eps_1)  * chi1 / (1.0 + chi1 * chi1)
                             + (eps_1  - eps_inf) * chi2 / (1.0 + chi2 * chi2);

        // Rayleigh absorption: alpha = (3*omega/c) * Im(m^2-1)/(m^2+2) * (3/4) * LWC / (4/3 pi r^3 rho_w)
        // Simplified for bulk: alpha = (6*pi*f/c) * (eps''/|eps+2|^2) * (rho_liq/rho_w)
        amrex::Real eps_real = eps_s; // approx for low freq
        amrex::Real denom = (eps_real + 2.0) * (eps_real + 2.0) + eps_imag * eps_imag;
        amrex::Real alpha = 6.0 * ERF_INSTR_PI * f * 1.0e9 / (c_GHz * 1.0e9)
                          * 3.0 * eps_imag / denom
                          * lwc / 1000.0; // lwc kg/m3 / rho_w kg/m3

        return alpha; // Np/m
    }

    std::shared_ptr<std::fstream>
    open_stream (const std::string& fname, const std::string& header) const
    {
        auto ptr = std::make_shared<std::fstream>();
        ptr->open(fname, std::ios::out);
        if (!ptr->good()) { amrex::FileOpenFailed(fname); }
        *ptr << header;
        ptr->flush();
        return ptr;
    }
};

#endif // ERF_INSTRUMENT_SIMULATORS_H
