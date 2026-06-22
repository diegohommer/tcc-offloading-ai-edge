"""Offloading policies and heuristics for LLM inference in PONs.

This module defines the decision-making algorithms that determine where a
specific LLMRequest should be processed (Edge, Fog, or Cloud).

For the baselines, the logic is either static or purely reactive. For the
proposed dynamic algorithm, it evaluates both the prompt weight and the
current state of the PON and hardware queues.
"""


def baseline_cloud_only(request, simulator_state) -> str:
    """Baseline 1: Forwards 100% of requests to the Cloud Data Center.

    This represents the standard current paradigm where all AI processing
    is centralized, ignoring PON upstream congestion.

    Args:
        request: The current LLMRequest object.
        simulator_state: The current instance of PONSimulator.

    Returns:
        str: Always returns 'cloud'.
    """
    return "cloud"


def baseline_edge_only(request, simulator_state) -> str:
    """Baseline 2: Attempts to process 100% of requests at the Edge (ONU).

    This represents a strict local-execution paradigm. If the Edge GPU is busy,
    the request will wait in the local hardware queue (FCFS) indefinitely.

    Args:
        request: The current LLMRequest object.
        simulator_state: The current instance of PONSimulator.

    Returns:
        str: Always returns 'edge'.
    """
    return "edge"


def baseline_user_only(request, simulator_state) -> str:
    """Baseline 0: Attempts to process 100% of requests directly on the User Device.

    This represents completely local execution (e.g., a smartphone NPU).
    It has zero network latency but extremely low tokens-per-second, leading
    to massive processing bottlenecks.
    """
    return "user"


def baseline_static_limit(request, simulator_state) -> str:
    """Baseline 3: A reactive, static threshold-based offloading policy.

    This policy checks the Edge node's current hardware queue. If the Edge GPU
    is currently processing a request (capacity full), it blindly offloads the
    new request to the Cloud to avoid local queueing delays.

    Args:
        request: The current LLMRequest object.
        simulator_state: The current instance of PONSimulator.

    Returns:
        str: 'edge' if local GPU is free, otherwise 'cloud'.
    """
    edge_node = simulator_state.nodes["edge"]

    # In SimPy, resource.count is the number of active users (max 1 for Edge)
    # resource.queue contains the pending requests.
    if edge_node.resource.count == 0 and len(edge_node.resource.queue) == 0:
        return "edge"
    else:
        # If Edge is busy, offload to Cloud blindly
        return "cloud"


def proposed_dynamic_algorithm(request, simulator_state) -> str:
    """Proposed TCC Algorithm: Dynamic context-aware offloading.

    (BASE TESTING VERSION)
    Currently implements a placeholder logic: offloads heavy prompts to the
    Cloud and keeps light prompts at the Edge, while checking the Fog tier.

    Future Implementation:
    Will calculate expected latency for all 3 tiers based on prompt size,
    current T-CONT DBA congestion (PON uplink), and GPU queue lengths,
    selecting the tier with the minimum `Expected Total Latency`.

    Args:
        request: The current LLMRequest object.
        simulator_state: The current instance of PONSimulator.

    Returns:
        str: The optimal tier ('edge', 'fog', or 'cloud').
    """
    # TODO: Implement the full mathematical optimization equation here.

    # Simple placeholder logic for initial framework testing:
    # If it's a massive prompt (> 1024 tokens), we assume the Edge would take
    # too long to process it, so we offload it.
    if request.prompt_tokens > 1024:
        # Let's try the Fog first if it's not heavily congested
        fog_node = simulator_state.nodes["fog"]
        if len(fog_node.resource.queue) < 5:
            return "fog"
        else:
            return "cloud"
    else:
        # For small prompts, prefer Edge unless completely overwhelmed
        edge_node = simulator_state.nodes["edge"]
        if len(edge_node.resource.queue) < 2:
            return "edge"
        else:
            return "cloud"
