"""Tests for disjoint MEC validation and held-out evaluation splits."""

from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import MECEnv
from onpolicy.scripts.eval.eval_mec import parse_args


def test_default_validation_and_test_episode_seeds_are_disjoint():
    args = get_config().parse_args([])
    validation = {
        args.eval_seed + batch * args.n_eval_rollout_threads * 1000
        + rank * 1000
        for batch in range(
            (args.eval_episodes + args.n_eval_rollout_threads - 1)
            // args.n_eval_rollout_threads
        )
        for rank in range(args.n_eval_rollout_threads)
    }
    validation = set(sorted(validation)[: args.eval_episodes])
    held_out = {
        args.test_seed + args.test_seed_stride * index
        for index in range(args.test_episodes)
    }

    assert args.eval_seed == 1000
    assert args.test_seed == 100000
    assert validation.isdisjoint(held_out)


def test_standalone_evaluation_honors_episode_horizon_override():
    args = parse_args(
        [
            "--env_name",
            "MEC",
            "--mec_scenario",
            "v6_hap_loadbearing",
            "--mec_episode_horizon",
            "350",
        ],
        get_config(),
    )
    env = MECEnv(args)

    assert env.env.horizon == 350
