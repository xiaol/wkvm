# Long Generation Token Compare

- prompt kind: **recall**
- prompt tokens: **13,824**
- prompt sha256: `d3b15d80ed317859...`
- requested output tokens: **512**

## Runs

| engine | output tokens | prefill+1st s | full wall s | decode tok/s | e2e out tok/s | facts in output |
|---|---:|---:|---:|---:|---:|---|
| wkvm:ring | 512 | 1.216 | 11.399 | 50.2 | 44.9 | - |
| vllm | 512 | 1.813 | 8.251 | 79.4 | 62.0 | code, city, checksum |
| sglang | 512 | 1.257 | 8.515 | 70.4 | 60.1 | code, city, checksum |

## Pairwise Token Agreement

| left | right | exact | LCP tokens | equal positions / min len | first mismatch |
|---|---|---:|---:|---:|---|
| wkvm:ring | vllm | False | 1 | 6/512 | 1: 6749 vs 15959 |
| wkvm:ring | sglang | False | 1 | 5/512 | 1: 6749 vs 15959 |
| vllm | sglang | False | 87 | 213/512 | 87: 2174 vs 20655 |

## Output Heads

### wkvm:ring

The critical record is **Project Chimera**, deployed in **Neo-Kyoto**, with a checksum word of **7B3A9F**. This serving benchmark evaluates a high-throughput inference system operating under sustained, production-level load. The system is designed to handle a continuous stream of requests, characterized by long input sequences, which necessitates efficient memory management and low-latency processing. Specifically, the benchmark measures key performance indicators (KPIs) such as requests per second (RPS), average latency, and tail latency (e.g., P95, P99) while the serving infrastructure is subjected to a constant, high-volume request rate. The architecture under test utilizes a sophisticated scheduling mechanism that manages concurrent requests, ensuring fair resource allocation across various inference tasks. Furthermore, the benchmark rigorously tests the system's stability and resource utilization (CPU, GPU, Memory) over extended periods to identify potential bottlenecks or degradation under prolonged stress. The long request histories imply that the model must maintain state or process context across many tokens, demanding optimized attention mechanisms and efficient KV-cache

### vllm

The codename is BLUE-742, the deployment city is Samarkand, and the checksum word is lantern. The benchmark document describes a serving system operating under sustained load, where incoming requests possess extensive histories. The scheduler manages resource allocation by admitting work based on a defined memory budget, and the generation process is token-by-token, with each decode step streaming a single new token through the model. To ensure reproducibility and allow for cross-engine auditing, engineers meticulously record key performance indicators, including latency, throughput, cache size, and the total number of output tokens generated for the identical prompt.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n

### sglang

The codename is BLUE-742, the deployment city is Samarkand, and the checksum word is lantern. The benchmark document describes a serving system operating under sustained load, where incoming requests possess extensive histories. The scheduler manages resource allocation by admitting work based on a defined memory budget, and the generation process is token-by-token, with each decode step streaming a single new token through the model. To ensure reproducibility and facilitate cross-engine auditing, engineers meticulously record key performance indicators, including latency, throughput, cache size, and the total number of output tokens generated for the identical prompt.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
