#include <ERF_SimpleLW.H>

using namespace amrex;

/**
 * Constructor for SimpleLW radiation model
 */
SimpleLW::SimpleLW (const int& lev, SolverChoice& sc)
    : m_lev(lev)
{
    // Parse radiation parameters from inputs file
    ParmParse pp("erf");

    // Radiation call frequency
    pp.query("rad_freq_in_steps", m_rad_freq_in_steps);

    // Simplified LW radiation parameters
    pp.query("simple_lw_F0", m_F0);       // W/m^2
    pp.query("simple_lw_F1", m_F1);       // W/m^2
    pp.query("simple_lw_kappa", m_kappa); // m^2/kg

    // Check if we have moisture
    m_moist = (sc.moisture_type != MoistureType::None);

    if (m_moist) {
        m_qc_idx = sc.moisture_indices.qc;
    }

    // Print configuration
    if (lev == 0) {
        Print() << "SimpleLW Radiation initialized with:" << std::endl;
        Print() << "  F0 (downwelling) = " << m_F0 << " W/m^2" << std::endl;
        Print() << "  F1 (upwelling)   = " << m_F1 << " W/m^2" << std::endl;
        Print() << "  kappa            = " << m_kappa << " m^2/kg" << std::endl;
        Print() << "  Update frequency = " << m_rad_freq_in_steps << " steps" << std::endl;
        if (!m_moist) {
            Print() << "  WARNING: No moisture model - radiation will have no effect!" << std::endl;
        }
    }
}

/**
 * Initialize the radiation model for the given geometry and box array
 */
void
SimpleLW::Init (const Geometry& geom,
                const BoxArray& ba,
                MultiFab* cons_in)
{
    // Simple LW does not require special initialization beyond parameter setup
    amrex::ignore_unused(geom, ba, cons_in);
}

/**
 * Run the simplified longwave radiation calculation
 *
 * The radiative flux is computed as:
 *   F_rad(z) = F0 * exp(-Q(z, inf)) + F1 * exp(-Q(0, z))
 *
 * where Q(a, b) = kappa * integral_a^b (rho * q_l) dz
 *
 * The heating rate is then:
 *   dT/dt = -(1 / (rho * c_p)) * dF/dz
 *
 * We store this as (dtheta/dt) = dT/dt * (theta/T) in qheating_rates
 * But since MakeSources multiplies by rho, we store: dtheta/dt
 */
void
SimpleLW::Run (int& level,
               int& step,
               Real& time,
               const Real& dt,
               const BoxArray& ba,
               Geometry& geom,
               MultiFab* cons_in,
               iMultiFab* lmask,
               MultiFab*  t_surf,
               Vector<MultiFab*>& lsm_input_ptrs,
               Vector<MultiFab*>& lsm_output_ptrs,
               MultiFab* qheating_rates,
               MultiFab* rad_fluxes,
               MultiFab* z_phys,
               MultiFab* lat,
               MultiFab* lon)
{
    amrex::ignore_unused(level, time, dt, ba, lmask, t_surf,
                         lsm_input_ptrs, lsm_output_ptrs,
                         rad_fluxes, lat, lon);

    // Check if we should update radiation this step
    if (step % m_rad_freq_in_steps != 0) {
        return;
    }

    // If no moisture, no cloud LWC, so radiation heating is zero
    if (!m_moist || m_qc_idx < 0) {
        qheating_rates->setVal(0.0);
        return;
    }

    // Get domain information
    const auto dz = geom.CellSize(2);
    const int klo = geom.Domain().smallEnd(2);
    const int khi = geom.Domain().bigEnd(2);

    // Radiation parameters for GPU capture
    const Real F0 = m_F0;
    const Real F1 = m_F1;
    const Real kappa = m_kappa;
    const Real c_p = Cp_d;
    const int qc_idx = m_qc_idx;

    // Loop over all boxes
    for (MFIter mfi(*cons_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        // Get arrays
        const Array4<const Real>& cons_arr = cons_in->const_array(mfi);
        const Array4<Real>& qheat_arr = qheating_rates->array(mfi);

        // Get z coordinates if available (for stretched grids)
        const Array4<const Real>& z_arr = (z_phys) ?
            z_phys->const_array(mfi) : Array4<const Real>{};
        const bool have_terrain = (z_phys != nullptr);

        // For each column, compute the liquid water path and radiative heating
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Get density at this cell
            const Real rho = cons_arr(i, j, k, Rho_comp);

            // Get cloud liquid water mixing ratio (kg/kg)
            // cons stores rho*q, so divide by rho
            const Real qc = cons_arr(i, j, k, qc_idx) / rho;

            // Get cell thickness
            Real dz_cell = dz;
            if (have_terrain) {
                // For terrain-following coords, compute actual cell height
                dz_cell = z_arr(i, j, k+1) - z_arr(i, j, k);
            }

            // Compute Q(0, z) - liquid water path from surface to this level
            // This requires a column integral, so we do it iteratively
            Real Q_below = 0.0;
            for (int kk = klo; kk < k; ++kk) {
                Real rho_kk = cons_arr(i, j, kk, Rho_comp);
                Real qc_kk = cons_arr(i, j, kk, qc_idx) / rho_kk;
                Real dz_kk = dz;
                if (have_terrain) {
                    dz_kk = z_arr(i, j, kk+1) - z_arr(i, j, kk);
                }
                Q_below += kappa * rho_kk * qc_kk * dz_kk;
            }

            // Compute Q(z, inf) - liquid water path from this level to top
            Real Q_above = 0.0;
            for (int kk = k; kk <= khi; ++kk) {
                Real rho_kk = cons_arr(i, j, kk, Rho_comp);
                Real qc_kk = cons_arr(i, j, kk, qc_idx) / rho_kk;
                Real dz_kk = dz;
                if (have_terrain) {
                    dz_kk = z_arr(i, j, kk+1) - z_arr(i, j, kk);
                }
                Q_above += kappa * rho_kk * qc_kk * dz_kk;
            }

            // Compute radiative flux at cell interfaces
            // F(z) = F0 * exp(-Q(z, inf)) + F1 * exp(-Q(0, z))

            // Flux at bottom of this cell (interface k)
            Real Q_below_bot = Q_below;
            Real Q_above_bot = Q_above;
            Real F_bot = F0 * std::exp(-Q_above_bot) + F1 * std::exp(-Q_below_bot);

            // Flux at top of this cell (interface k+1)
            Real Q_below_top = Q_below + kappa * rho * qc * dz_cell;
            Real Q_above_top = Q_above - kappa * rho * qc * dz_cell;
            Real F_top = F0 * std::exp(-Q_above_top) + F1 * std::exp(-Q_below_top);

            // Compute heating rate: dT/dt = -(1/(rho*c_p)) * dF/dz
            // The convention is F positive upward, so dF/dz = (F_top - F_bot) / dz
            Real dFdz = (F_top - F_bot) / dz_cell;
            Real dTdt = -dFdz / (rho * c_p);

            // Convert to potential temperature tendency
            // dtheta/dt = dT/dt * (theta/T)
            // For simplicity, we use the approximation that theta/T ~ 1 for near-surface
            // More accurate: theta = T * (p0/p)^(R/cp)
            // The source term in MakeSources.cpp expects: rho * dtheta/dt
            // But qheating_rates stores dtheta/dt (rho is multiplied in MakeSources)
            Real dtheta_dt = dTdt; // Approximate theta/T ~ 1

            // Store heating rate
            // Component 0: SW heating (not used here, set to 0)
            // Component 1: LW heating
            qheat_arr(i, j, k, 0) = 0.0;     // No shortwave
            qheat_arr(i, j, k, 1) = dtheta_dt;
        });
    }
}
