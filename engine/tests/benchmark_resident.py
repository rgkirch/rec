"""Measure Rec prefill and fixed-token resident decode separately on M1 Max."""

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx

from rec_mlx.kernels import build_decoder, device
from rec_mlx.quantized import QATModel


def measure(engine, prompt, warmup_steps, decode_steps):
    session = engine.make_session(max_new_tokens=warmup_steps + decode_steps)
    mx.synchronize()
    started = time.perf_counter()
    logits = session.prefill(prompt)
    mx.eval(logits)
    mx.synchronize()
    prefill_seconds = time.perf_counter() - started

    token = 150006
    for _ in range(warmup_steps):
        engine.forward(session._state, token)
    mx.synchronize()

    started = time.perf_counter()
    for _ in range(decode_steps):
        engine.forward(session._state, token)
    mx.synchronize()
    decode_seconds = time.perf_counter() - started
    return {
        "prompt_tokens": len(prompt),
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": len(prompt) / prefill_seconds,
        "decode_tokens": decode_steps,
        "decode_seconds": decode_seconds,
        "decode_tok_s": decode_steps / decode_seconds,
        "warmup_decode_tokens": warmup_steps,
        "decode_input_token": token,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(8, 12, 16, 32), required=True)
    parser.add_argument("--mlp-workers", type=int, choices=(8, 32, 64))
    parser.add_argument("--head-workers", type=int)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--trace-attach-delay",
        type=float,
        default=0,
        help="Pause after model setup so xctrace can attach before the benchmark",
    )
    args = parser.parse_args()
    if (
        args.prompt_tokens < 1
        or args.warmup_steps < 0
        or args.decode_steps < 1
        or args.repeats < 1
        or args.trace_attach_delay < 0
    ):
        parser.error(
            "prompt/decode lengths and repeats must be positive; warmup and attach delay may be zero"
        )

    base_profile = device.profile_for_device(mx.device_info())
    if base_profile.chip != "M1 Max":
        parser.error("this benchmark is reserved for the M1 Max experiment")
    mlp_workers = base_profile.mlp_workers if args.mlp_workers is None else args.mlp_workers
    head_workers = base_profile.head_workers if args.head_workers is None else args.head_workers
    if not 1 <= mlp_workers <= args.workers:
        parser.error("--mlp-workers must fit the resident worker pool")
    if not 1 <= head_workers <= args.workers:
        parser.error("--head-workers must fit the resident worker pool")
    profile = replace(
        base_profile,
        workers=args.workers,
        mlp_workers=mlp_workers,
        head_workers=head_workers,
    )
    device.profile_for_device = lambda _info: profile

    mx.set_cache_limit(256 * 1024**2)
    model = QATModel(args.model)
    engine, _resident = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    vocab_size = int(model.config["vocab_size"])
    prompt = [(index * 104729 + 1) % vocab_size for index in range(args.prompt_tokens)]

    if args.trace_attach_delay:
        print(json.dumps({"trace_attach_ready": True, "pid": os.getpid()}), flush=True)
        time.sleep(args.trace_attach_delay)

    # Compile and warm the measured path once, then report only subsequent runs.
    measure(engine, prompt, args.warmup_steps, args.decode_steps)
    runs = []
    for repeat in range(args.repeats):
        result = measure(engine, prompt, args.warmup_steps, args.decode_steps)
        result["repeat"] = repeat + 1
        runs.append(result)
        print(json.dumps({"run": result}), flush=True)

    print(
        json.dumps(
            {
                "device": mx.device_info(),
                "profile": asdict(profile),
                "prompt_tokens": args.prompt_tokens,
                "decode_tokens": args.decode_steps,
                "prefill_tok_s_median": statistics.median(row["prefill_tok_s"] for row in runs),
                "decode_tok_s_median": statistics.median(row["decode_tok_s"] for row in runs),
                "measurement": "fixed-token resident decode; no sampling included",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
