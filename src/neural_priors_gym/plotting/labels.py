"""TeX label mapping for GW parameter names."""

TEX_LABELS: dict[str, str] = {
    # Masses
    "mass_1_source": r"$m_1^{\rm{src}}\ [M_\odot]$",
    "mass_2_source": r"$m_2^{\rm{src}}\ [M_\odot]$",
    "mass_1": r"$m_1\ [M_\odot]$",
    "mass_2": r"$m_2\ [M_\odot]$",
    "chirp_mass_source": r"$\mathcal{M}_c^{\rm{src}}\ [M_\odot]$",
    "chirp_mass": r"$\mathcal{M}_c\ [M_\odot]$",
    "total_mass": r"$M_{\rm{tot}}\ [M_\odot]$",
    "mass_ratio": r"$q$",
    "symmetric_mass_ratio": r"$\eta$",
    # Tidal parameters
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "delta_lambda_tilde": r"$\delta\tilde{\Lambda}$",
    # Spins
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "chi_eff": r"$\chi_{\rm{eff}}$",
    "chi_p": r"$\chi_p$",
    "theta_jn": r"$\theta_{JN}$",
    "tilt_1": r"$\theta_1$",
    "tilt_2": r"$\theta_2$",
    "phi_12": r"$\phi_{12}$",
    "phi_jl": r"$\phi_{JL}$",
    # Distance and location
    "luminosity_distance": r"$d_L\ [\rm{Mpc}]$",
    "redshift": r"$z$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "geocent_time": r"$t_c\ [\rm{s}]$",
    "psi": r"$\psi$",
    "phase": r"$\phi_c$",
}


def get_tex_label(parameter_name: str) -> str:
    """Return the TeX label for a parameter, falling back to the parameter name."""
    return TEX_LABELS.get(parameter_name, parameter_name)


def get_tex_labels(parameter_names: list[str]) -> list[str]:
    """Return TeX labels for a list of parameters."""
    return [get_tex_label(name) for name in parameter_names]
