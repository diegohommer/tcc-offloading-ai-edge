# TCC: Dynamic Offloading for AI Inference at the Edge over PON Networks

**Repository:** `tcc-offloading-ia-edge`

This repository contains the source code, simulation scripts, and data analysis for the Bachelor's Thesis (TCC) focused on developing a dynamic balancing and offloading heuristic for Large Language Models (LLMs) inference in Passive Optical Networks (PON).

## 📌 The Problem
The central problem arises from the conflict between the high bandwidth and latency demands of LLMs and the architecture of PON networks.
1. **Cloud Bottleneck:** Sending all user AI requests to the Cloud generates a massive uplink bottleneck in the primary network and suffers from high round-trip time (RTT) latency.
2. **Edge Limitations:** Attempting to process everything at the edge (Edge OLT) is hindered by hardware limitations (GPU/VRAM).

**Objective:** To create a dynamic decision algorithm based on the estimated weight of the request (tokens) and PON congestion. The goal is the minimization of total latency and intelligent bandwidth optimization.

---

## 📂 Directory Structure

```text
tcc-offloading-ia-edge/
├── data/
│   ├── raw/                # Raw datasets (ShareGPT, LMSYS)
│   └── processed/          # Extracted distributions (input/output tokens)
├── src/
│   ├── simulator/          # PON network simulator source code (XGS-PON)
│   ├── heuristics/         # Implementation of our algorithm and baselines
│   └── utils/              # Log parser and background traffic generator
├── notebooks/              # Jupyter Notebooks for exploratory analysis and charts
├── results/                # Simulation logs and latency/throughput metrics
└── README.md               # Main repository documentation
```

---

## 📊 Data Sources and Simulation Parameters

To ensure the simulation is grounded in reality, it is fed with parameters extracted from widely validated public databases and benchmarks:

* **AI Workload:** ShareGPT / LMSYS Chatbot Arena. Extracts the distribution of input tokens (prompt) and output tokens (generation).
* **Edge Performance:** MLPerf Inference / Llama.cpp GitHub. Extracts Time-To-First-Token (TTFT) and Tokens-Per-Second (TPS) latency on hardware like NVIDIA Jetson Orin.
* **Network Traffic:** MAWI Working Group (PCAP traces) / ITU-T G.9807.1 Standard. Extracts background traffic patterns, frame size, and bursts in XGS-PON.

---

## ⚙️ Concurrency and Baselines

To prove the effectiveness of the proposed offloading heuristic, the simulated environment will compare it against three established traditional approaches:

1. **Baseline 1 (All-in-Cloud):** Forwards 100% of the requests directly to the Cloud Data Center. This point evaluates the RTT degradation and saturation of the PON network's uplink channel.
2. **Baseline 2 (Edge-Only / FCFS):** Processes everything at the edge (Edge OLT) on a first-come, first-served basis until VRAM is exhausted. This point evaluates the massive queue formation and the local waiting time bottleneck.
3. **Baseline 3 (Static Limit):** A fixed reactive heuristic (e.g., processes at the edge until reaching 80% GPU occupancy). This point evaluates the lack of sensitivity to the dynamic context of the fiber network traffic.
4. **Proposed Algorithm:** Dynamic decision based on the estimated weight of the request (tokens) and PON congestion. Aims for the minimization of total latency and intelligent bandwidth optimization.

---

## 🚀 How to Run (In Development)

Instructions on how to set up the virtual environment, install dependencies, and run the simulation scripts will be added as development progresses.

---
**Author:** Diego Amorim
