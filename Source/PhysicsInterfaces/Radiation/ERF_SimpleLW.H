#ifndef ERF_SIMPLE_LW_H
#define ERF_SIMPLE_LW_H

/*
 * Simplified Longwave Radiation Model for Stratocumulus Clouds
 *
 * This implements the longwave radiation parameterization commonly used in
 * LES studies of marine stratocumulus clouds (e.g., DYCOMS-II, VOCALS, ACE-ENA).
 *
 * The radiative flux is computed as:
 *   F_rad(z) = F0 * exp(-Q(z, inf)) + F1 * exp(-Q(0, z))
 *
 * where:
 *   Q(a, b) = kappa * integral_a^b (rho * q_l) dz
 *
 * Default parameters (from ACE-ENA ICMW case):
 *   F0 = 70 W/m^2  (downwelling from above cloud)
 *   F1 = 22 W/m^2  (upwelling from surface)
 *   kappa = 85 m^2/kg (absorption coefficient)
 *
 * The heating rate is computed as:
 *   dT/dt = -(1 / (rho * c_p)) * dF/dz
 *
 * References:
 * - Stevens et al. (2005), MWR, DYCOMS-II RF1
 * - Ackerman et al. (2009), MWR, DYCOMS-II RF2
 * - ACE-ENA ICMW case description (2023)
 */

#include <memory>
#include <iostream>

#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <ERF_RadiationInterface.H>
#include <ERF_Constants.H>
#include <ERF_IndexDefines.H>
#include <ERF_DataStruct.H>

class SimpleLW : public IRadiation
{
public:
    // Constructor
    SimpleLW (const int& lev, SolverChoice& sc);

    // Destructor
    ~SimpleLW () override = default;

    void
    Init (const amrex::Geometry& geom,
          const amrex::BoxArray& ba,
          amrex::MultiFab* cons_in) override;

    void
    Run (int& level,
         int& step,
         double& time,
         const double& dt,
         const amrex::BoxArray& ba,
         amrex::Geometry& geom,
         amrex::MultiFab* cons_in,
         amrex::iMultiFab* lmask,
         amrex::MultiFab*  t_surf,
         amrex::Vector<amrex::MultiFab*>& lsm_input_ptrs,
         amrex::Vector<amrex::MultiFab*>& lsm_output_ptrs,
         amrex::MultiFab* qheating_rates,
         amrex::MultiFab* rad_fluxes,
         amrex::MultiFab* z_phys,
         amrex::MultiFab* lat,
         amrex::MultiFab* lon,
         const bool updated_lsm) override;

    void
    WriteDataLog (const amrex::Real& time) override
    {
        // No-op for simplified radiation
        amrex::ignore_unused(time);
    }

private:
    // Grid level
    int m_lev;

    // Radiation frequency in steps
    int m_rad_freq_in_steps = 1;

    // Radiation parameters (default values from ACE-ENA case)
    amrex::Real m_F0 = 70.0;     // W/m^2, downwelling from above cloud top
    amrex::Real m_F1 = 22.0;     // W/m^2, upwelling from surface
    amrex::Real m_kappa = 85.0;  // m^2/kg, absorption coefficient

    // Index for cloud liquid water in moisture array
    int m_qc_idx = -1;

    // Do we have moisture?
    bool m_moist = false;
};

#endif // ERF_SIMPLE_LW_H
