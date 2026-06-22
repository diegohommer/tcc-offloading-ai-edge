"""Generates synthetic PON background traffic based on standard distributions.

This module creates a simulated log of network packets representing the
background traffic of 64 ONUs competing for upstream bandwidth in a TWDM-PON.
It strictly separates traffic into Expedited Forwarding (EF), Assured
Forwarding (AF), and Best Effort (BE) classes.

Scientific Justification for Traffic Modeling:
1. EF Traffic (Poisson): EF traffic represents delay-sensitive, predictable
   streams such as voice or control signaling, requiring guaranteed bandwidth.
   In PON Dynamic Bandwidth Allocation (DBA) literature, it is standard practice
   to model this using a Poisson process with exponential inter-arrival times.
   Source: "A New Predictive Dynamic Priority Scheduling in EPONs" (De et al., 2010)
   URL: https://web.iitd.ac.in/~swadesd/res/pubs/JNL/EPON-OSN2010.pdf

2. AF and BE Traffic (Self-Similar / Pareto): AF and BE represent standard
   Ethernet data (e.g., video, which requires minimum bandwidth guarantees,
   and general background traffic). Seminal networking research proved that
   aggregate Ethernet LAN traffic is statistically self-similar and bursty
   across time scales. Therefore, traditional Poisson models are unable to
   capture this fractal behavior. Instead, a heavy-tailed distribution like
   Pareto is used to approximate the Hurst parameter (typically between 0.5 and 1).
   Source: "On the Self-Similar Nature of Ethernet Traffic" (Leland et al., 1994)
   URL: http://www.cse.cuhk.edu.hk/~cslui/CSC5480/selfsim-eth.pdf
"""

import os
import numpy as np
import pandas as pd


def generate_traffic(num_packets: int = 100000, target_load_mbps: int = 800) -> pd.DataFrame:
    """Generates a DataFrame of simulated network packets.

    Args:
        num_packets (int): The total number of packets to simulate.
        target_load_mbps (int): The target total upstream load to scale
            packet arrival timestamps.

    Returns:
        pd.DataFrame: A dataframe containing packet arrival times (ms),
        source ONU IDs, QoS classes, and payload sizes (bytes).
    """
    print(f"Generating {num_packets} packets of background traffic...")

    classes = ["EF", "AF", "BE"]
    probabilities = [0.10, 0.30, 0.60]

    packet_classes = np.random.choice(classes, size=num_packets, p=probabilities)

    packet_sizes = np.zeros(num_packets, dtype=int)

    ef_mask = packet_classes == "EF"
    af_be_mask = (packet_classes == "AF") | (packet_classes == "BE")

    packet_sizes[ef_mask] = 70
    packet_sizes[af_be_mask] = np.random.randint(64, 1519, size=np.sum(af_be_mask))

    iats = np.zeros(num_packets)

    target_bytes_per_ms = (target_load_mbps * 1000000) / 8 / 1000
    avg_packet_size = np.mean(packet_sizes)
    base_iat_ms = avg_packet_size / target_bytes_per_ms

    iats[ef_mask] = np.random.exponential(scale=base_iat_ms, size=np.sum(ef_mask))

    alpha = 1.3
    pareto_mode = base_iat_ms * (alpha - 1) / alpha
    iats[af_be_mask] = (np.random.pareto(alpha, size=np.sum(af_be_mask)) + 1) * pareto_mode

    arrival_times_ms = np.cumsum(iats)

    source_onus = np.random.randint(1, 65, size=num_packets)

    df = pd.DataFrame(
        {
            "arrival_time_ms": arrival_times_ms,
            "onu_id": source_onus,
            "qos_class": packet_classes,
            "size_bytes": packet_sizes,
        }
    )

    return df


def main():
    """Executes the traffic generation pipeline and saves the output to a Parquet file."""
    df_traffic = generate_traffic(num_packets=500000, target_load_mbps=800)

    print("\nTraffic Statistics:")
    print(df_traffic["qos_class"].value_counts(normalize=True) * 100)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "../../data/processed"))
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "background_traffic.parquet")
    df_traffic.to_parquet(output_path, index=False)
    print(f"\n✅ Success! Traffic simulation saved to: {output_path}")


if __name__ == "__main__":
    main()
