########################
Walloon Specific Changes
########################

The Walloon workflow includes several changes to the default PyPSA-Eur:

* **Multi-scenario configuration.** ``config/config.times-pypsa.yaml`` enables wildcard-based 
  Walloon runs and points to ``config/scenarios.walloon.yaml``. The scenario file wires each 
  scenario name to a TIMES demand file (via `sector.times_file`) and to the matching custom 
  potentials, cost set, and aggregated carrier-level minimum and maximum capacities for each scenario.
* **Configuration options to trigger TIMES-adjusted demands.** The TIMES-PyPSA demand interlinkage 
  is activated in the Walloon  configuration by setting `sector.times_demand` to `true` and pointing 
  to a TIMES output file via `sector.times_file`. 
* **Nuclear capacity expansion**: The `electricity.extendable_nuclear_links` is added to the 
  Walloon configuration in ``config/config.walloon.yaml`` to allow new nuclear capacity 
  to be built as extendable links, for the nodes and horizons specified. Additionally, 
  planned nuclear power plants can be added to ``data/custom_powerplants.csv``. 
* **Custom potentials for BEWAL.** The Walloon configuration uses custom potentials 
  for various energy resources, defined in  ``data/walloon/custom_potentials.csv`` and 
  scenario-specific variants: ``custom_potentials_corrige.csv``, 
  ``custom_potentials_imppel.csv``, ``custom_potentials_alternatif.csv``, and 
  ``custom_potentials_alternatif_biolow.csv``. These files set maximum limits for solid biomass 
  (imports, transported, and local production), onshore wind, solar PV, rooftop PV, and selected 
  heat potentials. The relevant CSV is selected via `electricity.walloon_potentials` through 
  ``config/scenarios.walloon.yaml``.
* **Custom cost data.** The Walloon configuration uses updated cost assumptions 
  for specified fuels and technologies. The base file remains 
  ``data/walloon/custom_costs_rc.csv``; additional scenario variants include 
  ``custom_costs_corrige.csv`` and nuclear CAPEX sensitivities 
  ``custom_costs_nuc11500.csv`` and ``custom_costs_nuc13500.csv``. The scenario config 
  selects the appropriate file through `costs.custom_cost_fn`.
* **Aggregated capacity minimum and maximums.** Scenario-specific capacity limits in 
  ``data/walloon/agg_p_nom_minmax_base.csv`` and ``data/walloon/agg_p_nom_minmax_corrige.csv`` 
  refine the minimum/maximum nominal capacities used when enforcing aggregated capacity limits.
* **Custom power plants retirements.** The Walloon (BEWAL) nuclear power plant, Tihange, 
  is now defined in ``data/custom_powerplants.csv`` with as 3 separate units
  (Tihange 1/2/3) to allow the plant to retire its capacity incrementally. 
  The workflow filters out those rows by the current planning horizon so a unit 
  automatically disappears once its retirement year is passed. 
* **Single nuclear representation.** Removed duplication of nuclear representation in 
  model -- before they were represented as both generators and links, now only as links.
* **No new BEWAL nuclear before 2040 and configurable new builds.** ``config/config.walloon.yaml`` 
  contains a Walloon override under ``electricity.extendable_carriers`` that allows nuclear to be
  extendable only for specific planning horizons (e.g. 2040 and 2050). The planning horizon and 
  the carrier list can be configured as needed.

With these adjustments the Walloon run retires the Tihange power plant incrementally 
at their scheduled dates, removes duplicate representation of nuclear, and only allows
new Belgian nuclear capacity when the config explicitly enables it.

