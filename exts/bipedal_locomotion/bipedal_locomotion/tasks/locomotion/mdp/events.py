from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def prepare_quantity_for_tron(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_radius = 0.127,
):
    asset: Articulation = env.scene[asset_cfg.name]
    env._foot_radius = foot_radius

#sixth modification, remove this fucntion and define it again
# def apply_external_force_torque_stochastic(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     force_range: dict[str, tuple[float, float]],
#     torque_range: dict[str, tuple[float, float]],
#     probability: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """Randomize the external forces and torques applied to the bodies.

#     This function creates a set of random forces and torques sampled from the given ranges. The number of forces
#     and torques is equal to the number of bodies times the number of environments. The forces and torques are
#     applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
#     applied when ``asset.write_data_to_sim()`` is called in the environment.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     # clear the existing forces and torques
#     asset._external_force_b *= 0
#     asset._external_torque_b *= 0

#     # resolve environment ids
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device=asset.device)

#     random_values = torch.rand(env_ids.shape, device=env_ids.device)
#     mask = random_values < probability
#     masked_env_ids = env_ids[mask]

#     if len(masked_env_ids) == 0:
#         return

#     # resolve number of bodies
#     num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

#     # sample random forces and torques
#     size = (len(masked_env_ids), num_bodies, 3)
#     force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#     force_range = torch.tensor(force_range_list, device=asset.device)
#     forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
#     torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#     torque_range = torch.tensor(torque_range_list, device=asset.device)
#     torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
#     # set the forces and torques into the buffers
#     # note: these are only applied when you call: `asset.write_data_to_sim()`
#     asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)

def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize external forces and torques on bodies."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 1. Determine which environment instances get a push
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=asset.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        # If no environments sampled, disable any external wrench for all
        # asset.set_external_force_and_torque(
        #     forces=torch.zeros(0, 3),
        #     torques=torch.zeros(0, 3),
        #     env_ids=None,  # clear
        #     body_ids=None,
        # )
        asset.permanent_wrench_composer.reset()
        return

    # 2. Compute number of bodies
    if isinstance(asset_cfg.body_ids, list):
        body_ids = asset_cfg.body_ids
    else:
        body_ids = list(range(asset.num_bodies))

    num_bodies = len(body_ids)


    #8th modification
    # 3. Sample random forces
    fr_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # force_tensor = torch.tensor(fr_list, device=asset.device)

    # size = (len(masked_env_ids), num_bodies, 3)
    # forces = math_utils.sample_uniform(
    #     force_tensor[:, 0], force_tensor[:, 1], size, asset.device
    # )

    # # 4. Sample random torques
    # tr_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # torque_tensor = torch.tensor(tr_list, device=asset.device)
    # torques = math_utils.sample_uniform(
    #     torque_tensor[:, 0], torque_tensor[:, 1], size, asset.device
    # )

    # # 5. Apply via supported API
    # asset.set_external_force_and_torque(
    #     forces=forces,
    #     torques=torques,
    #     env_ids=masked_env_ids,
    #     body_ids=body_ids,
    # )

    # Add these lines:

    #     # 3. Sample random forces for each env & body (shape: N_envs x num_bodies x 3)
    # # 3. Sample random forces for each env & body (shape: N_envs x num_bodies x 3)
    # fr_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # force_min = torch.tensor([fr[0] for fr in fr_list], device=asset.device)
    # force_max = torch.tensor([fr[1] for fr in fr_list], device=asset.device)

    # # Expand min/max to broadcast over (N_envs x num_bodies)
    # force_min = force_min.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)
    # force_max = force_max.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)

    # forces = math_utils.sample_uniform(
    #     force_min, force_max, (len(masked_env_ids), num_bodies, 3), asset.device
    # )

    # # 4. Sample random torques similarly
    # tr_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    # torque_min = torch.tensor([tr[0] for tr in tr_list], device=asset.device)
    # torque_max = torch.tensor([tr[1] for tr in tr_list], device=asset.device)

    # torque_min = torque_min.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)
    # torque_max = torque_max.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)

    # torques = math_utils.sample_uniform(
    #     torque_min, torque_max, (len(masked_env_ids), num_bodies, 3), asset.device
    # )

    # # 5. Apply via supported API with correct shape
    # asset.set_external_force_and_torque(
    #     forces=forces,
    #     torques=torques,
    #     env_ids=masked_env_ids,
    #     body_ids=body_ids,
    # )

    # # 6. Write to simulation
    # asset.write_data_to_sim()


        # 3. Prepare batched ranges for external forces
    force_min = torch.tensor(
        [force_range.get(k, (0.0, 0.0))[0] for k in ["x", "y", "z"]],
        device=asset.device,
    )
    force_max = torch.tensor(
        [force_range.get(k, (0.0, 0.0))[1] for k in ["x", "y", "z"]],
        device=asset.device,
    )

    # reshape to (1, 1, 3) then expand to (N_envs, N_bodies, 3)
    force_min = force_min.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)
    force_max = force_max.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)

    forces = math_utils.sample_uniform(
        force_min, force_max, (len(masked_env_ids), num_bodies, 3), asset.device
    )

    # 4. Prepare batched ranges for torques
    torque_min = torch.tensor(
        [torque_range.get(k, (0.0, 0.0))[0] for k in ["x", "y", "z"]],
        device=asset.device,
    )
    torque_max = torch.tensor(
        [torque_range.get(k, (0.0, 0.0))[1] for k in ["x", "y", "z"]],
        device=asset.device,
    )

    torque_min = torque_min.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)
    torque_max = torque_max.view(1, 1, 3).expand(len(masked_env_ids), num_bodies, 3)

    torques = math_utils.sample_uniform(
        torque_min, torque_max, (len(masked_env_ids), num_bodies, 3), asset.device
    )

    # 5. Call the correct API with batched forces/torques
    # asset.set_external_force_and_torque(
    #     forces=forces,
    #     torques=torques,
    #     env_ids=masked_env_ids,
    #     body_ids=body_ids,
    # )

    # # 6. Write them into the simulation
    # asset.write_data_to_sim()

    # Clear previous forces first
    asset.permanent_wrench_composer.reset()

    # Apply forces using new API
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=masked_env_ids,
        is_global=False,
    )

def randomize_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertias of the bodies (num_assets, num_bodies)
    inertias = asset.root_physx_view.get_inertias().clone()
    masses = asset.root_physx_view.get_masses().clone()

    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias *= scale.unsqueeze(-1)

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the center of mass (COM) of the bodies by adding, scaling, or setting random values for each dimension.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()

    # Apply randomization to each dimension separately
    for dim in range(3):  # 0=x, 1=y, 2=z
        coms[..., dim] = _randomize_prop_by_op(
            coms[..., dim],
            com_distribution_params[dim],
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )

    asset.root_physx_view.set_coms(coms, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data
