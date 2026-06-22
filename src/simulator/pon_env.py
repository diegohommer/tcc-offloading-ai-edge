"""Discrete-event simulation engine for LLM inference over PON architectures.

This module models the end-to-end latency of processing Large Language Model (LLM)
requests across different network tiers (Edge, Fog, Cloud) using SimPy. It explicitly
models both the access network congestion (XGS-PON upstream) and the specific
autoregressive compute latency of modern LLMs (TTFT and TPS).
"""

import json
import logging
from typing import Dict, Any

import simpy

# Configure logging for the simulator output
logging.basicConfig(level=logging.INFO, format="%(message)s")


class LLMRequest:
    """Represents a single user prompt sent to the LLM.

    Attributes:
        req_id (int): Unique identifier for the request.
        prompt_tokens (int): Size of the input prompt (simulating context length).
        generation_tokens (int): Expected size of the generated output.
        arrival_time (float): Simulation time (ms) when the request was generated.
        completion_time (float): Simulation time (ms) when the response is fully received.
        offload_target (str): The tier where the request was processed ('edge', 'fog', 'cloud').
        total_latency (float): The total end-to-end latency in milliseconds.
    """

    def __init__(
        self, req_id: int, prompt_tokens: int, generation_tokens: int, env: simpy.Environment
    ):
        """Initializes a new LLM request."""
        self.req_id = req_id
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.arrival_time = env.now
        self.completion_time = 0.0
        self.offload_target = None
        self.total_latency = 0.0


class ComputeNode:
    """Simulates an LLM inference server at a specific tier (Edge, Fog, Cloud).

    Scientific Justification for Compute Delay Model:
    The latency of autoregressive LLMs is split into two distinct phases:
    the prefill phase (processing the prompt) and the decode phase (generating tokens).
    Our model uses Time-To-First-Token (TTFT) to encapsulate the prefill phase,
    and Tokens-Per-Second (TPS) for the decode phase.
    Source: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" (Kwon et al., SOSP 2023)
    URL: https://arxiv.org/abs/2309.06180

    Attributes:
        env (simpy.Environment): The core simulation environment.
        tier_name (str): Tier identifier ('edge', 'fog', 'cloud').
        ttft_ms (float): Base Time-To-First-Token latency in milliseconds.
        tps (float): Token generation rate (Tokens-Per-Second).
        resource (simpy.Resource): Hardware queueing mechanism for concurrent requests.
    """

    def __init__(self, env: simpy.Environment, tier_name: str, config: Dict[str, Any]):
        """Initializes the compute node with hardware capabilities from the config."""
        self.env = env
        self.tier_name = tier_name
        self.ttft_ms = float(config["time_to_first_token_ms"])
        self.tps = float(config["tokens_per_second"])

        # Capacity modeling based on Queueing Theory:
        # Edge (Jetson) acts as a strict M/M/1 single-server queue.
        # Cloud acts as an M/M/inf queue, assuming elastic parallel GPU clusters.
        if tier_name in ["user", "edge"]:
            capacity = 1
        elif tier_name == "fog":
            capacity = 10
        else:
            capacity = simpy.core.Infinity

        self.resource = simpy.Resource(env, capacity=capacity)

    def process(self, request: LLMRequest):
        """Simulates the hardware processing time of the LLM request.

        Args:
            request (LLMRequest): The inference request to process.

        Yields:
            simpy.events.Timeout: A timeout event representing the compute duration.
        """
        generation_delay_ms = (request.generation_tokens / self.tps) * 1000.0
        total_compute_ms = self.ttft_ms + generation_delay_ms

        yield self.env.timeout(total_compute_ms)


class NetworkLink:
    """Simulates the XGS-PON upstream link and propagation delay.

    Scientific Justification for Network Delay Model:
    In TWDM-PONs, upstream delay consists of optical propagation time, frame
    transmission time, and Dynamic Bandwidth Allocation (DBA) queueing delay.
    When offloading to the Cloud, the PON uplink must carry the entire prompt
    payload, subjecting it to T-CONT queueing behind bursty background traffic.
    Source: "Ethernet PON (ePON): Design and analysis of an optical access network" (Kramer et al., 2002)
    URL: https://ieeexplore.ieee.org/document/993309

    Attributes:
        env (simpy.Environment): The core simulation environment.
        config (Dict[str, Any]): Dictionary containing network RTT configurations per tier.
    """

    def __init__(self, env: simpy.Environment, config: Dict[str, Any]):
        """Initializes the network link simulator."""
        self.env = env
        self.config = config

    def transfer(
        self, request: LLMRequest, target_tier: str, current_congestion_factor: float = 1.0
    ):
        """Simulates the time taken to send the prompt to the target tier.

        Args:
            request (LLMRequest): The inference request being transmitted.
            target_tier (str): Destination tier ('edge', 'fog', 'cloud').
            current_congestion_factor (float): Multiplier representing PON uplink DBA queue length.

        Yields:
            simpy.events.Timeout: A timeout event representing the network transfer delay.
        """
        base_rtt = float(self.config[target_tier]["network_rtt_ms"])

        if target_tier in ["fog", "cloud"]:
            # Assume 1 token ~ 4 bytes. Calculate payload transmission delay over 10 Gbps uplink.
            # 10 Gbps = 1,250,000 bytes per millisecond.
            payload_bytes = request.prompt_tokens * 4
            transmission_delay = payload_bytes / 1250000.0

            # Upstream delay = One-way propagation + (Transmission delay * Congestion penalty)
            total_network_delay = (base_rtt / 2.0) + (
                transmission_delay * current_congestion_factor
            )
        else:
            # Local edge processing entirely avoids the shared PON uplink
            total_network_delay = base_rtt / 2.0

        yield self.env.timeout(total_network_delay)


class PONSimulator:
    """The main orchestration engine for the LLM offloading simulation.

    Attributes:
        env (simpy.Environment): The core simulation environment.
        hw_config (Dict[str, Any]): Loaded hardware and network tier configurations.
        network (NetworkLink): The simulated PON network link.
        nodes (Dict[str, ComputeNode]): Mapping of tier names to ComputeNode instances.
        completed_requests (list): List of LLMRequests that have finished processing.
    """

    def __init__(self, hardware_config_path: str):
        """Initializes the simulator and provisions the physical infrastructure.

        Args:
            hardware_config_path (str): Path to the hardware JSON configuration file.
        """
        self.env = simpy.Environment()

        with open(hardware_config_path, "r") as f:
            self.hw_config = json.load(f)["tiers"]

        self.network = NetworkLink(self.env, self.hw_config)
        self.nodes = {
            "user": ComputeNode(self.env, "user", self.hw_config["user"]),
            "edge": ComputeNode(self.env, "edge", self.hw_config["edge"]),
            "fog": ComputeNode(self.env, "fog", self.hw_config["fog"]),
            "cloud": ComputeNode(self.env, "cloud", self.hw_config["cloud"]),
        }

        self.completed_requests = []

    def handle_request(self, request: LLMRequest, target_tier: str, congestion: float):
        """Simulates the entire lifecycle of an LLM request: Network -> Compute -> Network.

        Args:
            request (LLMRequest): The inference request to process.
            target_tier (str): The assigned processing tier ('edge', 'fog', 'cloud').
            congestion (float): Current network congestion factor at the time of request.

        Yields:
            simpy.events.Process: Yields process steps for the SimPy environment to execute.
        """
        request.offload_target = target_tier

        # 1. Network Upstream: Transfer prompt to the target hardware
        yield self.env.process(self.network.transfer(request, target_tier, congestion))

        # 2. Compute: Wait for hardware availability, then process
        target_node = self.nodes[target_tier]
        with target_node.resource.request() as hardware_req:
            yield hardware_req
            yield self.env.process(target_node.process(request))

        # 3. Network Downstream: Return generated tokens (assuming uncongested downstream)
        yield self.env.timeout(float(self.hw_config[target_tier]["network_rtt_ms"]) / 2.0)

        # 4. Finalize and record metrics
        request.completion_time = self.env.now
        request.total_latency = request.completion_time - request.arrival_time
        self.completed_requests.append(request)

        logging.info(
            f"Req {request.req_id} finished at {target_tier.upper()} | Latency: {request.total_latency:.2f} ms"
        )

    def run(self, until: int):
        """Starts the simulation clock.

        Args:
            until (int): The maximum simulation time (in milliseconds) to run.
        """
        self.env.run(until=until)
