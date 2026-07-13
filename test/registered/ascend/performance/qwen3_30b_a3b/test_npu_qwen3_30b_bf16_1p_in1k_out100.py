import unittest
from pathlib import Path

import openai
import pandas as pd

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_MODEL_PATH,
    QWEN3_0_6B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_USE_MAX_DP_ATT": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    168,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--tp-size",
    2,
    "--enable-dp-attention",
    "--dp-size",
    1,
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    "--dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen",
    "--dump-kv-path",
    "/root/.cache/KVTC/openmath_dump"
]

import asyncio
from openai import AsyncOpenAI


async def run_requests(dataset_name, client, requests, client_concurrency):
    concurrency_semaphore = asyncio.Semaphore(client_concurrency)

    async def send_request(dataset_name, client, entry):
        prompt_id, prompt = entry

        async with concurrency_semaphore:
            print(f"Send {prompt_id=}")
            response = await client.chat.completions.create(
                model="Qwen3-30B-A3B",
                messages=[{"role": "user", "content": prompt["problem"]}],
                temperature=0,
            )

            print(f"{dataset_name} prompt {prompt_id} {response.usage.total_tokens=}")

    tasks = [send_request(dataset_name, client, r) for r in requests]

    return await asyncio.gather(*tasks)


class TestKVTCQwen30B_dump_openmath(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    openmath_parts = 144
    remote_address = [
            f"https://huggingface.co/datasets/nvidia/OpenMathReasoning/resolve/main/data/cot-00{i:03d}-of-00144.parquet"
            for i in range(openmath_parts)
            ]
    kvtc_dataset_name = "openmath"
    client_concurrency = 16

    def test_kvtc_qwen3_30b_dump_openmath(self):
        selector_dir = Path(__file__).resolve().parent / "datasets"
        selector_paths = (
            selector_dir / "low_token_openmath.txt",
            selector_dir / "high_token_openmath.txt",
        )
        selected_ids = {
            int(line.strip())
            for selector_path in selector_paths
            for line in selector_path.read_text().splitlines()
            if line.strip()
        }
        self.assertTrue(selected_ids, "OpenMath prompt selector files are empty")

        self.assertEqual(
                len(prompts), selected_ids,
                "Some selected OpenMath prompt IDs were not found in the parquet dataset",
        )

        dfs = []
        for i in range(self.openmath_parts):
            f = self.dataset_path / f"{self.kvtc_dataset_name}_{i}"

            df = pd.read_parquet(f)
            dfs.append(df)

        openmath_dataset = pd.concat(dfs, ignore_index=True)

        client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="None")
        submitted_ids = set()

        prompts = [
                (prompt_id, entry)
                for prompt_id, entry in openmath_dataset.iterrows()
                if prompt_id in selected_ids
                ]

        asyncio.run(run_requests(self.kvtc_dataset_name, client, prompts, self.client_concurrency))


class TestKVTCQwen30B_dump_fineweb(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    kvtc_remote_address = ["https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2025-26/000_00000.parquet"]
    kvtc_dataset_name = "fineweb"
    client_concurrency = 16

    def test_kvtc_qwen3_30b_dump_fineweb(self):
        selector_dir = Path(__file__).resolve().parent / "datasets"
        selector_paths = (
            selector_dir / "low_token_fineweb.txt",
            selector_dir / "high_token_fineweb.txt",
        )
        selected_ids = {
            int(line.strip())
            for selector_path in selector_paths
            for line in selector_path.read_text().splitlines()
            if line.strip()
        }
        self.assertTrue(selected_ids, "OpenMath prompt selector files are empty")

        self.assertEqual(
                len(prompts), selected_ids,
                "Some selected OpenMath prompt IDs were not found in the parquet dataset",
        )

        openmath_dataset = pd.read_parquet(self.dataset_path / f"{self.kvtc_dataset_name}_{0}")

        client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="None")
        submitted_ids = set()

        prompts = [
                (prompt_id, entry)
                for prompt_id, entry in openmath_dataset.iterrows()
                if prompt_id in selected_ids
                ]

        asyncio.run(run_requests(self.kvtc_dataset_name, client, prompts, self.client_concurrency))

if __name__ == "__main__":
    unittest.main()
