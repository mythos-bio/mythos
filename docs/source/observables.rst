Observables
===========

Observables are quantities computed from simulation trajectories. During
optimization, observables provide the **measured values** that are compared to
experimental targets — the difference drives gradient-based parameter updates.

.. contents:: On this page
   :local:
   :depth: 2


Observable API
--------------

All observables follow a simple callable interface:

.. code-block:: python

    observable(trajectory) → jnp.ndarray

Given a :class:`~mythos.simulators.io.SimulatorTrajectory`, an observable
returns a JAX array of shape ``(n_states, ...)`` containing the per-frame
measured values.

There are two observable patterns in ``mythos``:

**Single observables** return a single array:

.. code-block:: python

    from mythos.observables.propeller import PropellerTwist

    obs = PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12]]),
    )
    values = obs(trajectory)  # shape: (n_states,)

**Mapped observables** return a dictionary of arrays, keyed by name. These are
useful when a single simulation produces multiple related measurements (e.g.,
distances for every bond type in a Martini topology):

.. code-block:: python

    from mythos.observables.bond_distances import BondDistancesMapped

    obs = BondDistancesMapped(topology=martini_topology)
    values = obs(trajectory)  # {"DMPC_NC3_PO4": array(...), "DMPC_PO4_GL1": array(...), ...}


BaseObservable
^^^^^^^^^^^^^^

The ``BaseObservable`` class is the base for DNA/RNA structural observables. It
requires a ``rigid_body_transform_fn`` callable that converts rigid body
coordinates into nucleotide site positions, and provides helpers for computing
helical axes and duplex quartets.

See :doc:`autoapi/mythos/observables/base/index` for the full API.

Martini observables (``BondDistances``, ``TripletAngles``, membrane observables)
do not inherit from ``BaseObservable`` — they are standalone frozen dataclasses
that operate directly on Cartesian coordinates.


Using Observables in Loss Functions
------------------------------------

Observables are typically paired with loss functions to produce scalar losses
for optimization.

ObservableLossFn
^^^^^^^^^^^^^^^^

The ``ObservableLossFn`` wrapper combines an observable with a loss function
in a single callable. Given a trajectory, it computes the observable value
(as a weighted sum over frames), then evaluates the loss against a target:

.. code-block:: python

    from mythos.losses.observable_wrappers import ObservableLossFn, RootMeanSquaredError
    from mythos.observables.propeller import PropellerTwist

    loss_fn = ObservableLossFn(
        observable=PropellerTwist(
            rigid_body_transform_fn=transform_fn,
            h_bonded_base_pairs=h_bonded_pairs,
        ),
        loss_fn=RootMeanSquaredError(),
        return_observable=True,  # also return the measured value
    )

    loss, measured = loss_fn(trajectory, target, weights)

The ``weights`` array controls how frames are aggregated — typically zeros for
equilibration frames and uniform weights for production frames:

.. code-block:: python

    eq_steps = 2000
    prod_steps = 18000
    weights = jnp.concat([
        jnp.zeros(eq_steps),
        jnp.ones(prod_steps) / prod_steps,
    ])

Available Loss Functions
^^^^^^^^^^^^^^^^^^^^^^^^

``SquaredError``
  :math:`(y_{\text{target}} - y_{\text{actual}})^2`

``RootMeanSquaredError``
  :math:`\sqrt{\frac{1}{N}\sum_i (y_{\text{target},i} - y_{\text{actual},i})^2}`

``l2_loss``
  Standalone function: :math:`\sum_i (y_{\text{actual},i} - y_{\text{target},i})^2`

See :doc:`autoapi/mythos/losses/index` for the full API.


Using Observables with Objectives
----------------------------------

In the :doc:`optimization` framework, observables can also be used inside
``Objective`` and ``DiffTReObjective`` instances. In this context, the
observable is called on the trajectory exposed by a simulator, and the
objective's ``grad_or_loss_fn`` computes gradients from the result.

For DiffTRe-based optimization, observables that support **weighted
evaluation** (via the ``weights`` argument) allow Boltzmann-reweighted loss
computation without re-running the simulation.


WassersteinDistance
^^^^^^^^^^^^^^^^^^^

The ``WassersteinDistance`` observable wraps another observable and computes
the 1D Wasserstein (Earth Mover's) distance between its output distribution
and a fixed reference distribution. This is particularly useful for bottom-up
fitting where you want to match a full distribution shape rather than a single
scalar:

.. code-block:: python

    from mythos.observables.wasserstein import WassersteinDistance

    obs = WassersteinDistance(
        observable=my_bond_observable,
        v_distribution=reference_distribution,  # target distribution array
    )

The mapped variant ``WassersteinDistanceMapped`` operates on multiple named
distributions simultaneously.


.. _observable-catalog:

Available Observables
---------------------

DNA / RNA Structural
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Class
     - Description
     - Module
   * - ``Rise``
     - Helical rise (Å) from quartet midpoint projections
     - :mod:`~mythos.observables.rise`
   * - ``PitchAngle``
     - Average pitch angle (rad); pitch = π / ⟨angle⟩
     - :mod:`~mythos.observables.pitch`
   * - ``Diameter``
     - Helical diameter from backbone distances
     - :mod:`~mythos.observables.diameter`
   * - ``PropellerTwist``
     - Angle between hydrogen-bonded base normals
     - :mod:`~mythos.observables.propeller`
   * - ``TwistXY``
     - Total twist in the X-Y plane
     - :mod:`~mythos.observables.stretch_torsion`
   * - ``ExtensionZ``
     - Extension in the Z direction
     - :mod:`~mythos.observables.stretch_torsion`
   * - ``PersistenceLength``
     - Persistence length from vector autocorrelation + linear fit
     - :mod:`~mythos.observables.persistence_length`
   * - ``RMSE``
     - SVD-aligned RMSE vs. a reference structure
     - :mod:`~mythos.observables.rmse`
   * - ``MeltingTemp``
     - Melting temperature via umbrella sampling + histogram reweighting
     - :mod:`~mythos.observables.melting_temp`

Membrane
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Class
     - Description
     - Module
   * - ``MembraneThickness``
     - Bilayer thickness via leaflet assignment
     - :mod:`~mythos.observables.membrane_thickness`
   * - ``AreaPerLipid``
     - Area per lipid via leaflet assignment
     - :mod:`~mythos.observables.area_per_lipid`
   * - ``MembraneMeltingTemp``
     - Membrane Tm from sigmoid fit to APL vs. temperature
     - :mod:`~mythos.observables.membrane_melting_temp`

Bonded / Angle
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Class
     - Description
     - Module
   * - ``BondDistances`` / ``BondDistancesMapped``
     - Per-bond distances (single or multi-bond mapped variant)
     - :mod:`~mythos.observables.bond_distances`
   * - ``TripletAngles`` / ``TripletAnglesMapped``
     - Angles at central atom in triplets (single or mapped variant)
     - :mod:`~mythos.observables.triplet_angles`

Distance Metrics
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Class
     - Description
     - Module
   * - ``WassersteinDistance`` / ``WassersteinDistanceMapped``
     - 1D Wasserstein (Earth Mover's) distance to a reference distribution
     - :mod:`~mythos.observables.wasserstein`

For the full API reference, see :doc:`autoapi/mythos/observables/index`.
