"""Compare resident decoding with scalar and native MLX decoding on the Flint checkpoint."""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from rec_mlx.engine import Engine, Sampling
from rec_mlx.kernels import build_decoder, megakernel
from rec_mlx.kernels.masked_sampling import FilteredResidentSampler
from rec_mlx.quantized import QATModel


def validate_stopping(engine, resident, specification, tokenizer, prompt):
    model = engine.model
    original_eos, original_filtered = model.eos_tokens, resident.filtered
    eos_cases, cancellation_cases = [], []
    modes = ((0, 1, 0), (1, 1, 0), (0.6, 0.95, 20))
    try:
        for temperature, top_p, top_k in modes:
            settings = dict(temperature=temperature, top_p=top_p, top_k=top_k)
            sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

            def pick(logits, index):
                flat = logits.reshape(-1).astype(mx.float32)
                if temperature == 0:
                    return int(mx.argmax(flat).item())
                if (temperature, top_p, top_k) == (1, 1, 0):
                    return int(mx.random.categorical(flat, key=mx.random.key(42 + index)).item())
                return int(sample(flat - mx.logsumexp(flat)).item())

            discovery = engine.make_session(max_new_tokens=33)
            logits = discovery.prefill(prompt)
            mx.random.seed(42)
            tokens, keys = [pick(logits, 0)], [mx.random.state[0]]
            for index in range(32):
                tokens.append(pick(discovery.forward_token(tokens[-1]), index + 1))
                keys.append(mx.random.state[0])
            del discovery
            starts = [
                begin
                for begin in range(25)
                if all(tokens[begin + stop] not in tokens[: begin + stop] for stop in (1, 4, 8))
            ]
            assert starts, "Prompt did not produce unique early/middle/final EOS fixtures"
            begin = starts[0]
            for stop in (1, 4, 8):
                reference = engine.make_session(max_new_tokens=33)
                candidate = engine.make_session(max_new_tokens=33)
                reference.prefill(prompt)
                candidate.prefill(prompt)
                for token in tokens[:begin]:
                    reference.forward_token(token)
                    candidate.forward_token(token)
                mx.random.state[0] = keys[begin]
                expected = [
                    pick(reference.forward_token(tokens[begin + i]), begin + i + 1)
                    for i in range(stop)
                ]
                expected_key = mx.random.state[0]
                model.eos_tokens = [expected[-1]]
                # Each compiled graph must capture this fixture's EOS, not a previous one.
                resident.filtered = FilteredResidentSampler(resident.module, engine, specification)
                mx.random.state[0] = keys[begin]
                actual = resident.block(candidate._state, tokens[begin], 8, 43 + begin, **settings)
                assert actual == expected and len(actual) == stop, "EOS truncation mismatch"
                assert mx.array_equal(mx.random.state[0], expected_key).item(), "EOS RNG mismatch"
                assert candidate._state["position"] == reference._state["position"]
                assert same_kv(candidate._state["kv"], reference._state["kv"]), (
                    "EOS wrote beyond the retained prefix"
                )
                eos_cases.append(dict(**settings, stop=stop, packet_start=begin))
                del reference, candidate
    finally:
        model.eos_tokens, resident.filtered = original_eos, original_filtered

    api = Engine.__new__(Engine)
    api._prefix_preparation = None
    api.mx, api.runtime, api.resident_sampler = mx, engine, resident
    api.tokenizer, api.context_limit, api.eos = tokenizer, model.context_limit, set(original_eos)
    mx.eval(model.native.parameters())
    for temperature, top_p, top_k in modes:
        api._prompt_cache = None
        sampling = Sampling(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=17, seed=42
        )
        emitted = []
        for event in api.generate_tokens(prompt, sampling, should_stop=lambda: len(emitted) == 2):
            if "token" in event:
                emitted.append(event["token"])
            else:
                cancelled = event
        assert cancelled["finish_reason"] == "stop" and len(emitted) == 2
        assert cancelled["metrics"]["computed_tokens"] == 9, "Cancellation did not discard a packet"
        cached = list(api.generate_tokens(prompt, sampling))[-1]["metrics"]
        api._prompt_cache = None
        cold = list(api.generate_tokens(prompt, sampling))[-1]["metrics"]
        assert cached["cached_prompt_tokens"] == len(prompt) and cold["cached_prompt_tokens"] == 0
        assert cached["token_ids"] == cold["token_ids"], (
            "Cancelled packet contaminated prefix reuse"
        )
        cancellation_cases.append(dict(temperature=temperature, top_p=top_p, top_k=top_k))
    return dict(eos=eos_cases, cancellation=cancellation_cases)


def same_kv(a, b):
    return all(mx.array_equal(x, y).item() for x, y in zip(a.banks, b.banks, strict=True))


def validate_sustained_native(engine, model, steps, repeats):
    mx.eval(model.native.parameters())
    results = []
    token = 150006
    prompt = [1]
    for repeat in range(repeats):
        session = engine.make_session(max_new_tokens=steps + 1)
        native_session = engine.make_session(max_new_tokens=steps + 1)
        logits = session.prefill(prompt)
        native_prefill_logits = native_session.prefill(prompt)
        assert mx.array_equal(logits, native_prefill_logits).item(), "Prefill logits differ"
        assert same_kv(session._state["kv"], native_session._state["kv"]), "Prefill KV differs"
        native_cache = native_session.prompt_cache.layers
        for row, cache in zip(engine.metadata[:49], native_cache, strict=True):
            if not row[1]:
                cache.grow(len(prompt) + steps + 1)

        for step in range(steps):
            logits, trace = engine.forward(session._state, token)
            h = model.native.model.embed_tokens(mx.array([[token]], mx.int32))
            native_trace = [h.reshape(-1)]
            for layer, cache in zip(model.native.layers, native_cache, strict=True):
                h = layer(h, None, cache)
                native_trace.append(h.reshape(-1))
            native_logits = model.native.lm_head(model.native.model.norm(h)).reshape(-1)
            native_trace = mx.stack(native_trace)
            trace_equal = mx.array_equal(native_trace, trace).item()
            logits_equal = mx.array_equal(native_logits, logits.reshape(-1)).item()
            if not trace_equal or not logits_equal:
                detail = dict(
                    repeat=repeat + 1,
                    step=step + 1,
                    trace_equal=trace_equal,
                    logits_equal=logits_equal,
                    native_trace_shape=list(native_trace.shape),
                    rec_trace_shape=list(trace.shape),
                )
                if native_trace.shape == trace.shape and native_trace.ndim == 2:
                    difference = mx.abs(
                        native_trace.astype(mx.float32) - trace.astype(mx.float32)
                    )
                    unequal = mx.sum(native_trace != trace, axis=1).tolist()
                    maximum = mx.max(difference, axis=1).tolist()
                    first = next((index for index, count in enumerate(unequal) if count), None)
                    detail.update(
                        unequal_per_trace_row=unequal,
                        max_abs_per_trace_row=maximum,
                        first_divergent_trace_row=first,
                        first_divergent_model_layer=None if first is None else first - 1,
                    )
                if not logits_equal:
                    detail["logits_unequal"] = int(
                        mx.sum(native_logits != logits.reshape(-1)).item()
                    )
                    detail["logits_max_abs"] = float(
                        mx.max(
                            mx.abs(
                                native_logits.astype(mx.float32)
                                - logits.reshape(-1).astype(mx.float32)
                            )
                        ).item()
                    )
                print(json.dumps(detail), flush=True)
                raise AssertionError("Native parity mismatch")

        row = dict(repeat=repeat + 1, steps=steps, native_layer_checks=steps)
        results.append(row)
        print(json.dumps(row), flush=True)
        del session, native_session, native_cache
        mx.clear_cache()
    return results


def replace_shared_qkv_with_single_owner(
    source, workers, *, logbase, eps, use_block=False
):
    start = source.index("uint kv_team=group/")
    end = source.index("bool rounded=", start)
    projection = (
        "for(uint block=simdgroup_index_in_threadgroup;block<32;block+=8)"
        "q4_local_block<2048>(m,n,out,lane,block*4);"
        if use_block
        else "q4_local<2048>(m,n,out,lane,simdgroup_index_in_threadgroup,8);"
    )
    replacement = f"""for(uint work=group;work<40;work+={workers}) {{
            uint head=work;
            uint local=head<32?head:head<36?head-32:head-36;
            Matrix m=weight_matrix(bank,d+(head<32?10:head<36?15:20));
            m.w+=size_t(local)*128*1024;
            m.s+=local*128*32;
            m.b+=local*128*32;
            m.n=128;
            device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
            {projection}
            threadgroup_barrier(mem_flags::mem_device);
            if(simdgroup_index_in_threadgroup==0 && head<36)
                head_norm(out,(const device bfloat16_t*)(bank+d[head<32?5:6]),1,!global,uint(position),{logbase}f,{eps}f,lane,0);
            threadgroup_barrier(mem_flags::mem_device);
            if(head>=32) {{
                KVView target=head<36?keys:vals;
                for(uint dim=tid;dim<128;dim+=256)target.store(local,insert,dim,out[dim]);
            }}
        }}
        SYNC();
        """
    return source[:start] + replacement + source[end:]


def instrument_stage_kernel(
    specification,
    metal,
    *,
    workers,
    logbase,
    eps,
    single_owner_qkv=False,
    single_owner_block=False,
):
    options = dict(specification)
    source = options["source"]
    if single_owner_qkv or single_owner_block:
        source = replace_shared_qkv_with_single_owner(
            source,
            workers,
            logbase=logbase,
            eps=eps,
            use_block=single_owner_block,
        )
    width = 6144

    def store_vector(slot, name, size):
        return (
            f"if(group==0&&layer==0){{for(uint capture_i=tid;capture_i<{size};capture_i+=256)"
            f"stage_trace[{slot}u*{width}u+capture_i]={name}[capture_i];}}"
        )

    def store_hidden(slot):
        return (
            f"if(group==0&&layer==0){{for(uint capture_j=0;capture_j<8;++capture_j){{"
            "uint capture_i=(tid+(capture_j/4)*256)*4+capture_j%4;"
            f"stage_trace[{slot}u*{width}u+capture_i]=h[capture_j];}}}}"
        )

    def insert_after_call(text, prefix, addition):
        if text.count(prefix) != 1:
            raise ValueError(f"Expected one stage anchor, found {text.count(prefix)}: {prefix}")
        start = text.index(prefix)
        end = text.index(");", start) + 2
        return text[:end] + addition + text[end:]

    source = insert_after_call(
        source,
        "rms_256_w4(h,(const device bfloat16_t*)(bank+d[3]),n,",
        store_vector(0, "n", 2048),
    )
    source = insert_after_call(
        source,
        "rms_256_w4(h,(const device bfloat16_t*)(bank+d[4]),n,",
        store_vector(3, "n", 2048),
    )

    o_start = source.index("q4<4096>(weight_matrix(bank,d+25),a,o,lane,sg,")
    o_sync = source.index("SYNC();", o_start) + len("SYNC();")
    a_sync = source.rfind("SYNC();", 0, o_start) + len("SYNC();")
    source = (
        source[:a_sync]
        + store_vector(9, "a", 4096)
        + store_vector(10, "q", 4096)
        + store_vector(11, "k", 512)
        + store_vector(12, "v", 512)
        + source[a_sync:]
    )
    o_start = source.index("q4<4096>(weight_matrix(bank,d+25),a,o,lane,sg,")
    o_sync = source.index("SYNC();", o_start) + len("SYNC();")
    source = source[:o_sync] + store_vector(1, "o", 2048) + source[o_sync:]

    residual = (
        "for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;"
        "h[j]=bfloat16_t(float(h[j])+float(o[i]));}\n"
    )
    if source.count(residual) != 1:
        raise ValueError(f"Expected one attention-residual anchor, found {source.count(residual)}")
    source = source.replace(residual, residual + store_hidden(2), 1)

    gate_up_sync = "SYNC();\n            for(uint i=index;i<gm.n;"
    if source.count(gate_up_sync) != 1:
        raise ValueError(f"Expected one dense gate/up anchor, found {source.count(gate_up_sync)}")
    source = source.replace(
        gate_up_sync,
        "SYNC();\n            "
        + store_vector(4, "gate", 6144)
        + store_vector(5, "up", 6144)
        + "for(uint i=index;i<gm.n;",
        1,
    )

    down_start = source.index("q4<6144>(dm,act,down,lane,sg,")
    activation_sync = source.rfind("SYNC();", 0, down_start) + len("SYNC();")
    source = source[:activation_sync] + store_vector(6, "act", 6144) + source[activation_sync:]
    down_sync = source.index("SYNC();", down_start) + len("SYNC();")
    source = source[:down_sync] + store_vector(7, "down", 2048) + source[down_sync:]

    trace_store = "if(group==0)trace[(layer+1)*2048+i]=h[j];"
    if source.count(trace_store) != 2:
        raise ValueError(f"Expected two layer-output anchors, found {source.count(trace_store)}")
    source = source.replace(
        trace_store,
        "if(group==0&&layer==0)stage_trace[8u*6144u+i]=h[j];" + trace_store,
    )

    options["source"] = source
    options["output_names"] = [*options["output_names"], "stage_trace"]
    return metal(name="flint_decode_stage_probe", **options)


def make_single_head_projection_kernel(specification, metal):
    source = r"""
    uint tid=thread_index_in_threadgroup;
    uint lane=thread_index_in_simdgroup;
    uint simd=simdgroup_index_in_threadgroup;
    uint head=threadgroup_position_in_grid.x;
    threadgroup bfloat16_t n[2048];
    for(uint i=tid;i<2048;i+=256)n[i]=normalized[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint* d=metadata;
    Matrix m=weight_matrix(bank0,d+(head<32?10:head<36?15:20));
    uint local=head<32?head:head<36?head-32:head-36;
    m.w+=size_t(local)*128*1024;
    m.s+=local*128*32;
    m.b+=local*128*32;
    m.n=128;
    device bfloat16_t* out=projection+head*128;
    for(uint block=simd;block<32;block+=8)
        q4_local_block<2048>(m,n,out,lane,block*4);
    threadgroup_barrier(mem_flags::mem_device);
    if(simd==0&&head<36)
        head_norm(out,(const device bfloat16_t*)(bank0+d[head<32?5:6]),1,false,0u,0.0f,1e-6f,lane,0);
    threadgroup_barrier(mem_flags::mem_device);
    """
    options = dict(specification)
    options.update(
        input_names=["bank0", "metadata", "normalized"],
        output_names=["projection"],
        source=source,
    )
    return metal(name="flint_independent_head_projection", **options)


def validate_layer0_stages(engine, model, captured_stages, projection_kernel):
    prompt, token = [1], 150006
    session = engine.make_session(max_new_tokens=2)
    native_session = engine.make_session(max_new_tokens=2)
    session.prefill(prompt)
    native_session.prefill(prompt)
    assert same_kv(session._state["kv"], native_session._state["kv"]), "Prefill KV differs"
    native_cache = native_session.prompt_cache.layers
    for row, cache in zip(engine.metadata[:49], native_cache, strict=True):
        if not row[1]:
            cache.grow(len(prompt) + 2)

    engine.forward(session._state, token)
    stages = captured_stages["values"]
    mx.eval(stages)
    layer = model.native.layers[0]
    hidden = model.native.model.embed_tokens(mx.array([[token]], mx.int32))
    input_norm = layer.input_layernorm(hidden)
    attention_module = layer.self_attn
    batch, length, _hidden = input_norm.shape
    raw_query = attention_module.q_proj(input_norm).reshape(
        batch, length, attention_module.n_heads, attention_module.head_dim
    )
    raw_keys = attention_module.k_proj(input_norm).reshape(
        batch, length, attention_module.n_kv, attention_module.head_dim
    )
    raw_values = attention_module.v_proj(input_norm).reshape(
        batch, length, attention_module.n_kv, attention_module.head_dim
    )
    query = attention_module.q_norm(raw_query).transpose(0, 2, 1, 3)
    new_keys = attention_module.k_norm(raw_keys).transpose(0, 2, 1, 3)
    new_values = raw_values.transpose(0, 2, 1, 3)
    keys, values = native_cache[0].update_and_fetch(new_keys, new_values)
    attention_core = mx.fast.scaled_dot_product_attention(
        query,
        keys,
        values,
        scale=attention_module.scale,
        mask=None,
        sinks=attention_module.softmax_offset.astype(query.dtype),
    ).transpose(0, 2, 1, 3).reshape(batch, length, -1)
    attention = attention_module.o_proj(attention_core)
    attention_residual = hidden + attention
    mlp_input = layer.post_attention_layernorm(attention_residual)
    gate = layer.mlp.gate_proj(mlp_input)
    up = layer.mlp.up_proj(mlp_input)
    activation = nn.silu(gate) * up
    down = layer.mlp.down_proj(activation)
    output = attention_residual + down
    independent = projection_kernel(
        inputs=[engine.banks[0], native_session._state["metadata"], input_norm.reshape(-1)],
        grid=(40 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(40, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]
    native_qkv = mx.concatenate(
        (
            query.reshape(32, 128),
            new_keys.reshape(4, 128),
            new_values.reshape(4, 128),
        ),
        axis=0,
    )
    mx.eval(independent, native_qkv)
    independent_unequal = mx.sum(independent != native_qkv, axis=1).tolist()
    print(
        json.dumps(
            dict(
                stage="independent_qkv",
                exact=not any(independent_unequal),
                unequal_by_head=independent_unequal,
            )
        ),
        flush=True,
    )
    references = [("input_norm", 0, input_norm)]
    references.extend(
        (
            ("query", 10, query),
            ("key", 11, new_keys),
            ("value", 12, new_values),
            ("attention_core", 9, attention_core),
            ("attention_output", 1, attention),
            ("attention_residual", 2, attention_residual),
            ("post_attention_norm", 3, mlp_input),
            ("gate", 4, gate),
            ("up", 5, up),
            ("activation", 6, activation),
            ("down", 7, down),
            ("layer_output", 8, output),
        )
    )
    results = []
    for name, slot, reference in references:
        expected = reference.reshape(-1)
        actual = stages[slot, : expected.size].reshape(expected.shape)
        unequal = int(mx.sum(actual != expected).item())
        max_abs = float(
            mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
        )
        row = dict(stage=name, exact=unequal == 0, unequal=unequal, max_abs=max_abs)
        if name in ("query", "key", "value"):
            by_head = mx.sum((actual != expected).reshape(-1, 128), axis=1).tolist()
            row["unequal_by_head"] = by_head
        results.append(row)
        print(json.dumps(row), flush=True)
    first = next((row["stage"] for row in results if not row["exact"]), None)
    print(json.dumps(dict(first_divergent_stage=first)), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument("--output", type=Path, help="Optional JSON file for the results")
    parser.add_argument(
        "--workers",
        type=int,
        help="Experimental resident-worker override; leaves MLP/head workers unchanged",
    )
    parser.add_argument(
        "--sustained-steps",
        type=int,
        default=0,
        help="Run only repeated-token native parity for this many decode steps",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--stage-diagnostics",
        action="store_true",
        help="Compare layer-0 intermediate stages with native MLX and stop",
    )
    parser.add_argument(
        "--qkv-single-owner",
        action="store_true",
        help="In stage diagnostics, assign complete Q/K/V heads using q4_local",
    )
    parser.add_argument(
        "--qkv-single-owner-block",
        action="store_true",
        help="In stage diagnostics, assign complete Q/K/V heads using q4_local_block",
    )
    args = parser.parse_args()
    if args.sustained_steps < 0 or args.repeats < 1:
        parser.error("--sustained-steps must be nonnegative and --repeats must be positive")
    if args.stage_diagnostics and (args.workers is None or args.sustained_steps):
        parser.error("--stage-diagnostics requires --workers and cannot be combined with --sustained-steps")
    if (args.qkv_single_owner or args.qkv_single_owner_block) and not args.stage_diagnostics:
        parser.error("single-owner QKV modes require --stage-diagnostics")
    if args.qkv_single_owner and args.qkv_single_owner_block:
        parser.error("choose only one single-owner QKV mode")
    if args.workers is not None:
        from dataclasses import replace

        from rec_mlx.kernels import device

        base_profile = device.profile_for_device(mx.device_info())
        if base_profile.chip != "M1 Max":
            parser.error("--workers override is reserved for the M1 Max experiment")
        experimental_profile = replace(base_profile, workers=args.workers)
        device.profile_for_device = lambda _info: experimental_profile
    mx.set_cache_limit(256 * 1024**2)
    model = QATModel(args.model)
    specification = {}
    metal = megakernel._metal

    def capture_specification(name, **options):
        if name == "flint_decode_fast_activation":
            specification.update(options)
        return metal(name=name, **options)

    megakernel._metal = capture_specification
    engine, resident = build_decoder(model, optimized=True)
    print("profile:", engine.device_profile, flush=True)
    if args.stage_diagnostics:
        stage_kernel = instrument_stage_kernel(
            specification,
            metal,
            workers=args.workers,
            logbase=math.log2(model.config["rope_theta"]),
            eps=model.config["rms_norm_eps"],
            single_owner_qkv=args.qkv_single_owner,
            single_owner_block=args.qkv_single_owner_block,
        )
        projection_kernel = make_single_head_projection_kernel(specification, metal)
        captured_stages = {}

        def capture_stages(*call_args, **call_kwargs):
            call_kwargs = dict(call_kwargs)
            call_kwargs["output_shapes"] = [*call_kwargs["output_shapes"], (13, 6144)]
            call_kwargs["output_dtypes"] = [*call_kwargs["output_dtypes"], mx.bfloat16]
            outputs = stage_kernel(*call_args, **call_kwargs)
            captured_stages["values"] = outputs[-1]
            return outputs[:-1]

        engine.kernel = capture_stages
        validate_layer0_stages(engine, model, captured_stages, projection_kernel)
        return
    if args.sustained_steps:
        validate_sustained_native(
            engine, model, args.sustained_steps, args.repeats
        )
        return
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=False
    )
    text = "Write a complete Python program using openpyxl to create an Excel sales report, with sample data, formulas, formatting and a bar chart. Explain and test the code."
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=text)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    captured = {}
    persistent_kernel, greedy_kernel, filtered_run = (
        resident.kernel,
        resident.greedy_kernel,
        resident.filtered.run,
    )

    def persistent(**kwargs):
        out = persistent_kernel(**kwargs)
        captured["logits"], captured["trace"] = out[:2]
        return out

    def filtered(*a):
        out = filtered_run(*a)
        captured["logits"], captured["trace"] = out[3:5]
        return out

    def greedy(**kwargs):
        out = greedy_kernel(**kwargs)
        captured["logits"], captured["trace"] = out[:2]
        return out

    resident.kernel, resident.greedy_kernel, resident.filtered.run = persistent, greedy, filtered
    settings = [
        (0, 0.95, 20, 0),
        (0.05, 1, 0, 0),
        (0.6, 1, 0, 0),
        (1, 1, 0, 0),
        (1.7, 1, 0, 0),
        (2, 1, 0, 0),
        (1, 0.95, 0, 0),
        (0.6, 0.95, 20, 0),
        (1, 1, 1, 0),
        (2, 0.5, 149999, 0),
        (1, 0.01, 0, 0),
        (0, 1, 0, 1.5),
        (1, 1, 0, 1.5),
        (1, 0.95, 20, 1.5),
        (1, 0.95, 20, -1.5),
    ]
    results = []
    for context in (len(prompt), 513, 8192):
        ids = (prompt * ((context + len(prompt) - 1) // len(prompt)))[:context]
        for temperature, top_p, top_k, penalty in settings if context != 8192 else settings[:1]:
            for seed in (42, 73) if context != 8192 else (42,):
                steps = 520 if temperature == 0 and context == len(prompt) and seed == 42 else 16
                reference = engine.make_session(max_new_tokens=steps + 1)
                candidate = engine.make_session(max_new_tokens=steps + 1)
                a, b = reference.prefill(ids), candidate.prefill(ids)
                assert mx.array_equal(a, b).item(), "Prefill logits differ"
                assert same_kv(reference._state["kv"], candidate._state["kv"])
                sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
                unfiltered_t1 = (temperature, top_p, top_k) == (1, 1, 0)

                seen = set()

                def choose(logits, index):
                    flat = logits.reshape(-1).astype(mx.float32)
                    if penalty and seen:
                        flat[mx.array(list(seen), mx.int32)] -= penalty
                    if temperature == 0:
                        return int(mx.argmax(flat).item())
                    if unfiltered_t1:
                        noise = mx.random.gumbel(flat.shape, key=mx.random.key(seed + index))
                        return int(mx.argmax(flat + noise).item())
                    return int(sample(flat - mx.logsumexp(flat)).item())

                mx.random.seed(seed)
                pending = choose(a, 0)
                seen.add(pending)
                if penalty:
                    candidate._state["presence_mask"] = mx.zeros(a.size, mx.float32)
                    candidate._state["presence_mask"][pending] = 1.0
                scalar_key, packet_key = mx.random.state[0], mx.random.state[0]
                # Shared global KV is mutated by decode; use a separate native reference.
                native_session = engine.make_session(max_new_tokens=steps + 1)
                native_session.prefill(ids)
                native_cache = native_session.prompt_cache.layers
                for row, cache in zip(engine.metadata[:49], native_cache, strict=True):
                    if not row[1]:
                        cache.grow(context + steps + 1)
                native_checks = 0
                position, packets = 0, 0
                while position < steps and pending not in model.eos_tokens:
                    count = min((1, 3, 8)[packets % 3], steps - position)
                    packet_input = pending
                    expected, logits, traces = [], [], []
                    mx.random.state[0] = scalar_key
                    for offset in range(count):
                        value, trace = engine.forward(reference._state, pending)
                        if temperature == 0 and position + offset < 8:
                            h = model.native.model.embed_tokens(mx.array([[pending]], mx.int32))
                            native_trace = [h.reshape(-1)]
                            for layer, cache in zip(model.native.layers, native_cache, strict=True):
                                h = layer(h, None, cache)
                                native_trace.append(h.reshape(-1))
                            native_logits = model.native.lm_head(model.native.model.norm(h))
                            assert mx.array_equal(mx.stack(native_trace), trace).item(), (
                                "Native layer mismatch"
                            )
                            assert mx.array_equal(native_logits.reshape(-1), value).item(), (
                                "Native logits mismatch"
                            )
                            native_checks += 1
                        pending = choose(value, position + offset + 1)
                        seen.add(pending)
                        expected.append(pending)
                        logits.append(value)
                        traces.append(trace)
                        if pending in model.eos_tokens:
                            break
                    scalar_key = mx.random.state[0]
                    mx.random.state[0] = packet_key
                    actual = resident.block(
                        candidate._state,
                        packet_input,
                        count,
                        seed + position + 1,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        presence_penalty=penalty,
                    )
                    packet_key = mx.random.state[0]
                    assert actual == expected, (
                        context,
                        temperature,
                        top_p,
                        top_k,
                        position,
                        actual,
                        expected,
                    )
                    assert mx.array_equal(packet_key, scalar_key).item(), "Sampler key mismatch"
                    for offset in range(len(actual)):
                        assert mx.array_equal(captured["logits"][offset], logits[offset]).item(), (
                            "Packet logits mismatch"
                        )
                        assert mx.array_equal(captured["trace"][offset], traces[offset]).item(), (
                            "Packet layer mismatch"
                        )
                    if penalty:
                        mask = mx.zeros(a.size, mx.float32)
                        mask[mx.array(list(seen), mx.int32)] = 1
                        assert mx.array_equal(candidate._state["presence_mask"], mask).item()
                    assert reference._state["position"] == candidate._state["position"]
                    assert same_kv(reference._state["kv"], candidate._state["kv"]), "KV mismatch"
                    position += len(actual)
                    packets += 1
                row = dict(
                    context=context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=penalty,
                    seed=seed,
                    positions=position,
                    packets=packets,
                    native_checks=native_checks,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                del reference, candidate, native_session, native_cache, a, b
                mx.clear_cache()

    stopping = validate_stopping(engine, resident, specification, tokenizer, prompt)
    print(json.dumps(dict(stopping=stopping)), flush=True)
    performance = []
    ids = prompt
    for temperature, top_p, top_k, penalty in (
        settings[0],
        settings[2],
        settings[3],
        settings[7],
        (1, 0.95, 20, 0),
        (1, 0.95, 20, 1.5),
    ):
        sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        timings = {"scalar": [], "packet": []}
        for repeat in range(4):
            for variant in ("scalar", "packet") if repeat % 2 == 0 else ("packet", "scalar"):
                session = engine.make_session(max_new_tokens=257)
                logits = session.prefill(ids)
                mx.random.seed(42)
                if temperature == 0:
                    token = int(mx.argmax(logits).item())
                elif (temperature, top_p, top_k) == (1, 1, 0):
                    token = int(
                        mx.random.categorical(
                            logits.astype(mx.float32), key=mx.random.key(42)
                        ).item()
                    )
                else:
                    flat = logits.reshape(-1).astype(mx.float32)
                    token = int(sample(flat - mx.logsumexp(flat)).item())
                seen = {token}
                if penalty:
                    session._state["presence_mask"] = mx.zeros(logits.size, mx.float32)
                    session._state["presence_mask"][token] = 1.0
                count = 0
                mx.synchronize()
                started = time.perf_counter()
                while count < 256 and token not in model.eos_tokens:
                    if variant == "packet":
                        tokens = resident.block(
                            session._state,
                            token,
                            min(8, 256 - count),
                            43 + count,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            presence_penalty=penalty,
                        )
                        count += len(tokens)
                        token = tokens[-1]
                    else:
                        value = session.forward_token(token).reshape(-1).astype(mx.float32)
                        if penalty:
                            value[mx.array(list(seen), mx.int32)] -= penalty
                        if temperature == 0:
                            token = int(mx.argmax(value).item())
                        elif (temperature, top_p, top_k) == (1, 1, 0):
                            token = int(
                                mx.random.categorical(value, key=mx.random.key(43 + count)).item()
                            )
                        else:
                            token = int(sample(value - mx.logsumexp(value)).item())
                        seen.add(token)
                        count += 1
                mx.synchronize()
                speed = count / (time.perf_counter() - started)
                if repeat:
                    timings[variant].append(speed)
                del session
        row = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=penalty,
            **{name: statistics.median(values) for name, values in timings.items()},
        )
        performance.append(row)
        print(json.dumps(dict(performance=row)), flush=True)
    result = dict(
        device=mx.device_info(),
        config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        cases=results,
        stopping=stopping,
        performance=performance,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
