xtrack-tools
============

Local tools for xtrack/xsuite workflows.

The current tracking helpers use an explicit kick-plane API:

- ``create_initial_conditions(..., kick_plane="x" | "y" | "xy")``
- ``run_tracking_without_ac_dipole(..., use_diagonal_kicks=False)`` to generate separate horizontal and vertical particles from each action-angle pair
- ``process_tracking_data(...)`` to drop the ramp, renumber the first kept turn to ``1``, and keep exactly ``flattop_turns`` turns

Contents
--------

.. toctree::
   :maxdepth: 2

   api
