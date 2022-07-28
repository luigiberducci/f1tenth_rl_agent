import pathlib

import numpy as np
import stable_baselines3.sac.policies
import torch

import argparse

from rl_node.policies import SACMlpPolicy


def main(args):
    model_file = args.model_file
    output_dir = args.output_dir
    actor_class = getattr(stable_baselines3, args.actor_class)
    print(f"[info] sb3 actor: filepath: {model_file}, actor class: {actor_class}")


    # conversion
    sb3_model = actor_class.load(model_file, device="cpu")
    torch_model = SACMlpPolicy.from_actor(actor=sb3_model.policy.actor)
    torch_model.to("cpu")
    check = sanity_check(sb3_model, torch_model, n_repeats=10, debug=args.debug)

    # save model
    out_filepath = output_dir / f"torch_{model_file.stem}.pt"
    model_scripted = torch.jit.script(torch_model)
    model_scripted.save(out_filepath)
    print(f"[info] torch model saved in {out_filepath}")


def sanity_check(sb3_model, torch_model, n_repeats, debug=False):
    torch_model.requires_grad_(False)
    observation_size = sb3_model.observation_space.shape[0]

    for _ in range(n_repeats):
        dummy_input = torch.randn(1, observation_size).to("cpu")
        dummy_input = torch.clip(dummy_input, -1, +1)

        new_output = torch_model(dummy_input).numpy()
        original_output, _ = sb3_model.predict(dummy_input, deterministic=True)

        if debug:
            print(f"dummy input: {dummy_input}")
            print(f"new output: {new_output}")
            print(f"original output: {original_output}")

        assert np.mean(new_output - original_output) <= 0.0000001, "failed matching output"
    return True


if __name__ == "__main__":
    available_actors = ["SAC"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=pathlib.Path, help="sb3 model", required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, help="where to store the torch model", default=".")
    parser.add_argument("--actor_class", choices=available_actors, help="sb3 class defining the actor", default="SAC")
    parser.add_argument("-debug", action="store_true", help="print out results of sanity check")
    args = parser.parse_args()

    main(args)
