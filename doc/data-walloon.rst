Walloon-specific Data
=====================

* ``config/config.times-pypsa.yaml`` – Walloon multi-scenario configuration enabling wildcards; lists 
  the Walloon scenario names, shared planning horizons, nodes, and references the scenario overrides 
  file.
* ``config/scenarios.walloon.yaml`` – scenario overrides used by the wildcard runs. Each scenario 
  (``scen_base``, ``scen_corrige``, ``scen_nuc13500``, ``scen_nuc11500``, ``scen_imppel``, 
  ``scen_data``) points to a specific TIMES input (``sector.times_file``), Walloon potentials CSV, 
  cost CSV, and aggregated capacity envelope.
* ``data/walloon/custom_costs_rc.csv`` – custom cost assumptions used by the Walloon configuration. 
  Data provided by ICEDD. Additional scenario-specific sets:
  - ``data/walloon/custom_costs_corrige.csv`` – corrected fuel prices and investment costs for the main scenarios.
  - ``data/walloon/custom_costs_nuc11500.csv`` / ``data/walloon/custom_costs_nuc13500.csv`` – cost 
    variants with higher nuclear/SMR CAPEX sensitivities.
* ``data/walloon/wal_2021_existing_capacities_2.csv`` - this contains data on existing generators 
  in Wallonia; data provided by ICEDD.
* ``data/custom_powerplants.csv`` – custom power plant modified to include the Walloon (BEWAL) 
  nuclear power plant Tihange as 3 separate units for incremental retirement. Doel nuclear power 
  plant in Flanders is also split into multiple generators for incremental retirement. Retirement 
  data provided by ICEDD.
* ``data/walloon/custom_potentials.csv`` - custom potentials for the BEWAL region:
  - solid biomass import: maximum amount of biomass that can be imported to BEWAL from outside of 
    the model area (non-Europe) (GWh/an)
  - solid biomass transported: maximum amount of biomass that can be transported from other nodes in 
    the model to BEWAL (GWh/an)
  - solid biomass: maximum amount of local production of solid biomass in BEWAL region (GWh/an)
  - onwind, solar, solar rooftop: maximum potentials for onshore wind, solar PV and rooftop solar PV 
    in BEWAL region (MW)
  Scenario-specific potential variants used by the new scenarios:
  - ``data/walloon/custom_potentials_corrige.csv`` – corrected biomass, solar, and wind potentials 
    (no biomass imports).
  - ``data/walloon/custom_potentials_imppel.csv`` – intermediate biomass import case with modest 
    imports/transport caps.
  - ``data/walloon/custom_potentials_alternatif.csv`` – higher biomass import/transport availability.
  - ``data/walloon/custom_potentials_alternatif_biolow.csv`` – lower biomass import/transport 
    availability.
* ``data/walloon/ntc_2030.csv`` – net transfer capacities (NTCs) between European countries in 2030 (MW).
* ``data/agg_p_nom_minmax.csv`` - minimum and maximum nominal capacities for aggregated generators 
  at the country or bus level. Most values are from TYNDP 2022. Solar-all values for BE and BEWAL 
  are provided by Climact, based on the ELIA ADEXFLEX.
  - ``data/walloon/agg_p_nom_minmax_base.csv`` / ``data/walloon/agg_p_nom_minmax_corrige.csv`` – 
    scenario-specific envelopes for aggregated nominal capacities (e.g., onshore/offshore wind, solar, 
    nuclear) used in the new Walloon scenarios. These also specify exact amounts of nuclear to be built in 
    BEWAL and BEBRU.  
