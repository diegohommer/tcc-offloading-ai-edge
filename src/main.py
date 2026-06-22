"""Main execution script for the PON LLM Offloading Simulation.

This script orchestrates the discrete-event simulation by loading the hardware
configurations, background traffic, and LLM workload datasets. It generates
inference requests based on a Poisson arrival process and routes them according
to the selected heuristic policy.

Scientific Justification for Request Generation:
User requests for interactive services (such as LLM inference) are modeled
using a Poisson arrival process (exponential inter-arrival times). This is the
standard foundation of M/M/c queueing theory, accurately reflecting the
stochastic, independent nature of human-generated traffic in access networks.
Source: "Data Networks" (Bertsekas and Gallager, 1992)
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# Add the 'src' directory to the Python path to allow direct execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator.pon_env import PONSimulator, LLMRequest
import heuristics.policies as policies

# Configure logging to show essential simulation progress
logging.basicConfig(level=logging.WARNING, format="%(message)s")


def request_generator(env, simulator, df_workload, policy_func, total_requests, arrival_rate_ms):
    """Generates LLM inference requests and injects them into the simulation.

    Args:
        env (simpy.Environment): The SimPy environment.
        simulator (PONSimulator): The initialized PON environment.
        df_workload (pd.DataFrame): Dataset containing prompt and generation token distributions.
        policy_func (Callable): The heuristic function used to decide the offload target.
        total_requests (int): The maximum number of requests to simulate.
        arrival_rate_ms (float): The average time (ms) between incoming requests.

    Yields:
        simpy.events.Timeout: The simulated delay between arriving requests.
    """
    for i in range(total_requests):
        # 1. Wait for the next user request (Exponential Inter-Arrival Time)
        inter_arrival_time = np.random.exponential(scale=arrival_rate_ms)
        yield env.timeout(inter_arrival_time)

        # 2. Sample a real interaction from the LMSYS dataset
        sample = df_workload.sample(1).iloc[0]
        request = LLMRequest(
            req_id=i,
            prompt_tokens=int(sample["prompt_tokens"]),
            generation_tokens=int(sample["generation_tokens"]),
            env=env,
        )

        # 3. Determine current network congestion
        # Note: In a fully coupled implementation, this would dynamically read the
        # T-CONT queue length from the background traffic simulation.
        # For now, we simulate a variable congestion factor (1.0 = Empty, 3.0 = Heavy Load).
        current_congestion = np.random.uniform(1.0, 3.0)

        # 4. Ask the "Brain" (Heuristic Policy) where to process this request
        target_tier = policy_func(request, simulator)

        # 5. Hand the request to the Simulator Engine
        env.process(simulator.handle_request(request, target_tier, current_congestion))


def main():
    """Parses CLI arguments, sets up data, and runs the simulation."""
    parser = argparse.ArgumentParser(
        description="Run the PON LLM Offloading Simulation Environment."
    )

    # CLI Arguments
    parser.add_argument(
        "--policy",
        "-p",
        type=str,
        choices=["user", "cloud", "edge", "static", "dynamic"],
        default="cloud",
        help="Selects the offloading heuristic policy to test.",
    )
    parser.add_argument(
        "--requests",
        "-r",
        type=int,
        default=500,
        help="Total number of LLM inference requests to simulate.",
    )
    parser.add_argument(
        "--arrival_rate",
        "-a",
        type=float,
        default=250.0,
        help="Average time between requests in milliseconds (Poisson lambda).",
    )
    parser.add_argument(
        "--hw_config",
        "-c",
        type=str,
        default="data/hardware_config.json",
        help="Path to the hardware configuration JSON file.",
    )
    parser.add_argument(
        "--workload",
        "-w",
        type=str,
        default="data/processed/lmsys_token_distribution.parquet",
        help="Path to the token distribution dataset.",
    )

    args = parser.parse_args()

    print(f"--- Starting PON Inference Simulation ---")
    print(f"Policy: {args.policy.upper()}")
    print(f"Total Requests: {args.requests}")
    print(f"Average Arrival Rate: 1 request every {args.arrival_rate} ms\n")

    # 1. Map CLI argument to the actual policy function
    policy_map = {
        "user": policies.baseline_user_only,
        "cloud": policies.baseline_cloud_only,
        "edge": policies.baseline_edge_only,
        "static": policies.baseline_static_limit,
        "dynamic": policies.proposed_dynamic_algorithm,
    }
    selected_policy = policy_map[args.policy]

    # 2. Resolve Paths and Load Data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    hw_path = os.path.join(base_dir, args.hw_config)
    workload_path = os.path.join(base_dir, args.workload)

    if not os.path.exists(hw_path) or not os.path.exists(workload_path):
        print("ERROR: Data files not found. Have you run the generation scripts inside 'utils/'?")
        sys.exit(1)

    df_workload = pd.read_parquet(workload_path)

    # 3. Initialize Simulator Environment
    simulator = PONSimulator(hardware_config_path=hw_path)

    # 4. Inject the workload generator into the SimPy environment
    simulator.env.process(
        request_generator(
            env=simulator.env,
            simulator=simulator,
            df_workload=df_workload,
            policy_func=selected_policy,
            total_requests=args.requests,
            arrival_rate_ms=args.arrival_rate,
        )
    )

    # 5. Run the simulation
    # We let it run until all events are processed (until=None)
    simulator.env.run()

    # 6. Process and Save Results
    results_data = []
    for req in simulator.completed_requests:
        results_data.append(
            {
                "req_id": req.req_id,
                "prompt_tokens": req.prompt_tokens,
                "generation_tokens": req.generation_tokens,
                "offload_target": req.offload_target,
                "total_latency_ms": req.total_latency,
            }
        )

    df_results = pd.DataFrame(results_data)

    # Calculate simple statistics
    avg_latency = df_results["total_latency_ms"].mean()
    target_counts = df_results["offload_target"].value_counts().to_dict()

    print("\n--- Simulation Complete ---")
    print(f"Total Requests Processed: {len(df_results)}")
    print(f"Average End-to-End Latency: {avg_latency:.2f} ms")
    print(f"Offloading Distribution: {target_counts}")

    # Ensure results directory exists and save
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"sim_results_{args.policy}.csv")
    df_results.to_csv(output_file, index=False)

    print(f"Detailed logs saved to: {output_file}")


if __name__ == "__main__":
    main()
