# Long Generation Token Compare

- prompt kind: **recall**
- prompt tokens: **16,384**
- prompt sha256: `42dce83e063e2058...`
- requested output tokens: **512**

## Runs

| engine | output tokens | prefill+1st s | full wall s | decode tok/s | e2e out tok/s | facts in output |
|---|---:|---:|---:|---:|---:|---|
| wkvm:ring | 512 | 1.487 | 11.332 | 51.9 | 45.2 | - |
| vllm | 512 | 2.064 | 8.607 | 78.1 | 59.5 | code, city, checksum |

## Pairwise Token Agreement

| left | right | exact | LCP tokens | equal positions / min len | first mismatch |
|---|---|---:|---:|---:|---|
| wkvm:ring | vllm | False | 1 | 7/512 | 1: 6749 vs 15959 |

## Output Heads

### wkvm:ring

The critical record is **Project Chimera**, deployed in **Neo-Kyoto**, with a checksum word of **$\Sigma_{901}$**. This serving benchmark evaluates a high-throughput inference system operating under sustained, heavy load. The system is designed to handle a continuous stream of requests, characterized by their significant input length, which necessitates efficient memory management and processing pipelines. Specifically, the benchmark measures key performance indicators (KPIs) such as latency, throughput (requests per second), and resource utilization (CPU/GPU load) while the serving system is subjected to a constant influx of data. The long request histories imply that the underlying model inference must be highly optimized for sequential processing and state retention, often involving complex attention mechanisms or recurrent structures. The evaluation focuses on the system's stability and predictable performance envelope when pushed to its operational limits, ensuring that the serving infrastructure can maintain service quality under real-world, high-demand scenarios.thought\nHere's a thinking process to arrive at the desired answer:\n\n1.  **Analyze the Request:**\n    *   **Inp

### vllm

The codename is BLUE-742, the deployment city is Samarkand, and the checksum word is lantern. The benchmark document details a serving system operating under sustained load, where incoming requests possess extensive histories. The scheduler manages resource allocation by admitting work based on a defined memory budget. Furthermore, the generation process is token-by-token streaming, with each decode step producing a single new token from the model. To ensure reproducibility and facilitate cross-engine auditing, engineers meticulously record key performance indicators, including latency, throughput, cache size, and the total number of output tokens generated for the identical prompt.
