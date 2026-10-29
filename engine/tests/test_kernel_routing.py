import unittest
from types import SimpleNamespace

import mlx.core as mx

from rec_mlx.kernels import expert_teams, megakernel, mixed_bits, rank_routing
from rec_mlx.kernels.device import profile_for_device
from rec_mlx.kernels.kv_teams import function
from rec_mlx.kernels.kv_teams import transform as kv_teams_transform
from rec_mlx.kernels.resident_block import transform as resident_block_transform
from rec_mlx.kernels.resident_optimizations import transform as resident_transform


def m4_m5_resident_device():
    try:
        return profile_for_device(mx.device_info()).chip in {"M4 Pro", "M5 Pro", "M5 Max"}
    except ValueError:
        return False


def selected_weights(probabilities, bias):
    selected = mx.argpartition(-(probabilities + bias), kth=5)[:6]
    chosen = probabilities[selected]
    total = mx.array(0.0, mx.float32)
    for i in range(6):
        total = chosen[i] + total
    return selected, chosen / (total + 1e-20)


def m1_max_device():
    try:
        return profile_for_device(mx.device_info()).chip == "M1 Max"
    except ValueError:
        return False


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class RoutingTests(unittest.TestCase):
    @unittest.skipUnless(
        mx.metal.is_available() and m4_m5_resident_device(), "requires M4 Pro, M5 Pro or M5 Max"
    )
    def test_scores_survive_attention_scratch_reuse(self):
        probabilities = (mx.arange(256).astype(mx.float32) + 1) / 1024
        partial = mx.arange(256).astype(mx.float32) % 9 / 16
        partial[mx.array([7, 31, 64])] = 4
        partial[mx.array([2, 17, 93, 201])] = 3
        cases = [
            ("unbiased", mx.zeros((256,))),
            ("all_ties_with_bias", 1 - probabilities),
            ("partial_ties_with_bias", partial - probabilities),
        ]
        for expert in range(256):
            bias = (mx.arange(256).astype(mx.float32) % 13) / 128
            bias[expert] = 2
            cases.append((f"boost_{expert}", bias))

        for workers in (32, 64):
            source = """
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x,phase=0;
    device uint* ids=routing+group*16;
    device float* scores=statistics+256;
    device atomic_uint* control=(device atomic_uint*)private_control;
    threadgroup uint ready[1];
    #define SYNC() if(!sync_all(control,tid,phase,ready,profile,group)){if(tid==0)diagnostic[group]=phase;return;}
    threadgroup uint d[10];
    const device uchar* bank=(const device uchar*)bias;
    if(tid==0){d[9]=0;diagnostic[group]=0;}
    if(tid<16)ids[tid]=0xffffffffu;
    if(group==0)statistics[tid]=probabilities[tid];
    SYNC();
    ROUTE
    // The production down-projection join publishes every leader's chosen score.
    SYNC();
    if(group==0 && tid<6)statistics[256+tid]=NAN;
    if(tid==0)routing[group*16+14]=123u;
    SYNC();
            float sums[8];
    uint r=lane;
    if(simdgroup_index_in_threadgroup==0 && lane<6)
        actual[group*6+lane]=1.0f*scores[r]+0.0f*scores[0];
""".replace("ROUTE", rank_routing.ROUTE_CALL)
            sync = function(megakernel._HEADER, "sync_all")
            sync = sync.replace("__WORKERS__", str(workers)).replace("__PROFILE__", "0")
            header, source = rank_routing.transform(sync, source, workers)
            kernel = megakernel._metal(
                name="routing_scores_survive_attention_reuse",
                input_names=["bias", "probabilities", "private_control"],
                output_names=["routing", "statistics", "actual", "diagnostic", "profile"],
                header=header,
                source=source,
                compile_options={"math_mode": "safe"},
            )
            for case, bias in cases:
                with self.subTest(mlp_workers=workers, case=case):
                    selected, expected = selected_weights(probabilities, bias)
                    routing, _, actual, diagnostic, _ = kernel(
                        inputs=[bias, probabilities, mx.zeros((1024,), mx.uint32)],
                        output_shapes=[(workers, 16), (262,), (workers, 6), (workers,), (1,)],
                        output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.uint32, mx.uint32],
                        grid=(workers * 256, 1, 1),
                        threadgroup=(256, 1, 1),
                    )
                    self.assertEqual(diagnostic.tolist(), [0] * workers)
                    self.assertTrue(mx.all(mx.isfinite(actual)).item())
                    self.assertLess(float(mx.max(mx.abs(actual - expected)).item()), 1e-7)
                    ids = routing[:, :6].tolist()
                    selected_ids = selected.tolist()
                    for group in range(workers):
                        expected_ids = [0xFFFFFFFF] * 6
                        if group < workers * 3 // 4:
                            role = group // (workers // 8)
                            expected_ids[role] = selected_ids[role]
                        self.assertEqual(ids[group], expected_ids, group)
                    self.assertEqual(routing[:, 14].tolist(), [123] * workers)
                    self.assertEqual(routing[:, 15].tolist(), [0xFFFFFFFF] * workers)

    def test_m1_profile_uses_32_resident_mlp_and_head_workers(self):
        profile = profile_for_device(
            {"architecture": "applegpu_g13s", "device_name": "Apple M1 Max"}
        )
        self.assertEqual((profile.workers, profile.mlp_workers, profile.head_workers), (32, 32, 32))
        header = expert_teams.transform(megakernel._HEADER, profile.mlp_workers)
        gateup = function(header, "expert_team_gateup")
        handoff = function(header, "expert_team_handoff")
        self.assertIn("bool shared=group>=24;", gateup)
        self.assertIn("uint teams=shared?8:4", gateup)
        self.assertIn("uint first=group>=24?24", handoff)
        self.assertIn("count=group>=24?8:4", handoff)
        legacy_sync = function(megakernel._HEADER, "sync_all")
        self.assertIn(
            "atomic_fetch_add_explicit(control+1,0u,memory_order_relaxed)==phase",
            legacy_sync,
        )
        self.assertIn(
            "atomic_fetch_add_explicit((device atomic_uint*)(routing+peer*16+14),0u,memory_order_relaxed)<layer+1",
            handoff,
        )

        optimized_header = resident_transform(header)
        optimized_header = optimized_header.replace("__WORKERS__", str(profile.workers))
        optimized_header = optimized_header.replace("__PROFILE__", "0")
        optimized_header = optimized_header.replace(
            "__ATTENTION_GROUPS__", str(profile.workers * 8)
        )
        optimized_header = optimized_header.replace("__MERGE_GROUPS__", str(profile.workers * 8))
        sync = function(optimized_header, "sync_all")
        self.assertEqual(
            sync.count("atomic_fetch_add_explicit(control,0u,memory_order_relaxed)"), 2
        )
        rendered_body = megakernel._BODY.replace("__WORKERS__", str(profile.workers))
        rendered_body = rendered_body.replace("__MERGE_WORKERS__", str(profile.workers))
        team_header, _ = kv_teams_transform(
            optimized_header,
            rendered_body,
            profile.workers,
        )
        team_sync = function(team_header, "kv_team_sync")
        self.assertEqual(
            team_sync.count("atomic_fetch_add_explicit(team_control,0u,memory_order_relaxed)"),
            2,
        )

    @unittest.skipUnless(mx.metal.is_available() and m1_max_device(), "requires M1 Max Metal")
    def test_m1_global_and_kv_team_barrier_payload_stress(self):
        for workers in (8, 12):
            for team_size in (None, workers // 4):
                modes = (
                    (("optimized", False), ("generation", True))
                    if team_size is None
                    else (("kv_team", False),)
                )
                for mode, original_barrier in modes:
                    for coherent in (False, True):
                        for repeat in range(3):
                            with self.subTest(
                                workers=workers,
                                mode=mode,
                                coherent=coherent,
                                repeat=repeat + 1,
                            ):
                                payload, diagnostics, bad_peer_mask, _profile = (
                                    self.run_barrier_payload_probe(
                                        workers,
                                        team_size,
                                        original_barrier=original_barrier,
                                        writer_coherent=coherent,
                                        reader_coherent=coherent,
                                    )
                                )
                                observed = diagnostics.tolist()
                                details = (
                                    f"diagnostics={observed} bad_peer_mask={bad_peer_mask.tolist()} "
                                    f"payload={payload.tolist()}"
                                )
                                if workers == 12 and not coherent:
                                    self.assertTrue(any(observed), msg=details)
                                else:
                                    self.assertEqual(observed, [0] * workers, msg=details)
                                self.assertEqual(len(payload.tolist()), workers)

    @unittest.skipUnless(mx.metal.is_available() and m1_max_device(), "requires M1 Max Metal")
    def test_m1_32_worker_coherent_global_scratch_visibility(self):
        for repeat in range(3):
            payload, diagnostics, bad_peer_mask, _profile = self.run_barrier_payload_probe(
                32,
                None,
                writer_coherent=True,
                reader_coherent=True,
            )
            self.assertEqual(
                diagnostics.tolist(),
                [0] * 32,
                msg=f"repeat={repeat} mask={bad_peer_mask.tolist()} payload={payload.tolist()}",
            )

    @unittest.skipUnless(mx.metal.is_available() and m1_max_device(), "requires M1 Max Metal")
    def test_m1_resident_sampling_argmax_qualifies_cross_group_statistics(self):
        source = megakernel._BODY.replace("__PROFILE__", "0")
        ordinary = resident_block_transform(source, 32)
        coherent = resident_block_transform(source, 32, coherent_statistics=True)
        captured = resident_block_transform(
            source, 32, coherent_statistics=True, capture_group_tokens=True
        )
        self.assertIn("statistics[group*2]=maximum;", ordinary)
        self.assertIn("float other=statistics[peer*2];", ordinary)
        self.assertIn(
            "((device coherent(device) float*)statistics)[group*2]=maximum;",
            coherent,
        )
        self.assertIn(
            "float other=((device coherent(device) float*)statistics)[peer*2];",
            coherent,
        )
        self.assertIn(
            "((device coherent(device) uint*)statistics)[group*2+1]=choice;",
            coherent,
        )
        self.assertIn(
            "uint other_word=((device coherent(device) uint*)statistics)[peer*2+1];",
            coherent,
        )
        self.assertIn(
            "atomic_store_explicit(control+3,choice,memory_order_relaxed);",
            coherent,
        )
        self.assertIn(
            "atomic_fetch_add_explicit(control+3,0u,memory_order_relaxed)",
            coherent,
        )
        self.assertIn("profile[17u+block_step*32u+group]=block_input_token;", captured)

    def run_barrier_payload_probe(
        self,
        workers,
        team_size,
        iterations=256,
        original_barrier=False,
        writer_coherent=False,
        reader_coherent=False,
    ):
        if original_barrier:
            header = megakernel._HEADER
            header = header.replace("__WORKERS__", str(workers))
            header = header.replace("__PROFILE__", "0")
        else:
            header = resident_transform(megakernel._HEADER)
            header = header.replace("__WORKERS__", str(workers))
            header = header.replace("__PROFILE__", "0")
            header = header.replace("__ATTENTION_GROUPS__", str(workers * 8))
            header = header.replace("__MERGE_GROUPS__", str(workers * 8))
        if team_size is None:
            sync_name = "sync_all"
            sync_header = function(header, sync_name)
            team_start, team_count = 0, workers
        else:
            body = megakernel._BODY.replace("__WORKERS__", str(workers))
            body = body.replace("__MERGE_WORKERS__", str(workers))
            team_header, _ = kv_teams_transform(header, body, workers)
            sync_name = "kv_team_sync"
            sync_header = function(team_header, sync_name)
            team_start = f"(group/{team_size})*{team_size}"
            team_count = team_size
        coherent_pointer = (
            "device coherent(device) uint* coherent_payload=(device coherent(device) uint*)payload;"
            if writer_coherent or reader_coherent
            else ""
        )
        write_pointer = "coherent_payload" if writer_coherent else "payload"
        read_pointer = "coherent_payload" if reader_coherent else "payload"
        source = f"""
            uint tid=thread_index_in_threadgroup;
            uint group=threadgroup_position_in_grid.x;
            device atomic_uint* control=(device atomic_uint*)private_control;
            {coherent_pointer}
            threadgroup uint ready[1];
            uint phase=0,mismatches=0;
            for(uint iteration=0;iteration<{iterations}u;++iteration) {{
                if(tid==0)
                    {write_pointer}[group]=((iteration+1u)*97u)^((group+1u)*2654435761u);
                if(!{sync_name}(control,tid,phase,ready,profile,group)) {{
                    if(tid==0)diagnostic[group]=0x80000000u|phase;
                    return;
                }}
                if(tid==0) {{
                    uint first={team_start},bad_mask=0;
                    for(uint peer=first;peer<first+{team_count}u;++peer) {{
                        uint expected=((iteration+1u)*97u)^((peer+1u)*2654435761u);
                        if({read_pointer}[peer]!=expected) {{
                            ++mismatches;
                            if(iteration==0)bad_mask|=1u<<peer;
                        }}
                    }}
                    if(iteration==0)bad_peer_mask[group]=bad_mask;
                }}
                if(!{sync_name}(control,tid,phase,ready,profile,group)) {{
                    if(tid==0)diagnostic[group]=0x80000000u|phase;
                    return;
                }}
            }}
            if(tid==0)diagnostic[group]=mismatches;
        """
        kernel = megakernel._metal(
            name=f"m1_{sync_name}_payload_stress_{workers}_{'generation' if original_barrier else 'optimized'}_w{int(writer_coherent)}r{int(reader_coherent)}",
            input_names=["private_control"],
            output_names=["payload", "diagnostic", "bad_peer_mask", "profile"],
            header=sync_header,
            source=source,
            compile_options={"math_mode": "safe"},
        )
        outputs = kernel(
            inputs=[mx.zeros((1024,), mx.uint32)],
            output_shapes=[(workers,), (workers,), (workers,), (workers * iterations * 2,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
            grid=(workers * 256, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(*outputs)
        return outputs

    def test_native_bf16_expert_reduction(self):
        for workers in (32, 64):
            for bits, specialized in ((4, False), (3, False), (3, True)):
                for ranked in (False, True):
                    with self.subTest(
                        workers=workers, bits=bits, specialized=specialized, ranked=ranked
                    ):
                        self.check_reduction(workers, bits, specialized, ranked)

    def check_reduction(self, workers, bits, specialized, ranked):
        model = SimpleNamespace(
            embedding=SimpleNamespace(bits=8),
            head=SimpleNamespace(bits=4),
            config={"mlx_export": {"routed_expert_bits": bits}},
        )
        header = expert_teams.transform(megakernel._HEADER, workers)
        body = megakernel._BODY
        if ranked:
            header, body = rank_routing.transform(header, body, workers)
        _, body = mixed_bits.transform(header, body, model, workers, specialized)
        start = body.index(
            "            float w3_chosen=" if ranked else "            float sums[8];"
        )
        end = body.index("        } else {", start)
        source = """
            uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
            uint group=0, layer=0;
            device uint* routing=(device uint*)scores;
            bfloat16_t h[8];
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=residual[i];}
        """ + body[start:end]
        kernel = megakernel._metal(
            name="qat_native_bf16_expert_mix",
            input_names=["down", "scores", "residual"],
            output_names=["trace"],
            source=source,
            compile_options={"math_mode": "safe"},
        )
        for seed in (19, 31, 47):
            mx.random.seed(seed)
            down = mx.random.normal((7, 2048)).astype(mx.bfloat16)
            residual = mx.random.normal((2048,)).astype(mx.bfloat16)
            probabilities = mx.random.uniform(shape=(256,))
            bias = mx.random.uniform(low=-0.5, high=0.5, shape=(256,))
            selected, weights = selected_weights(probabilities, bias)
            products = down[:6] * weights.astype(mx.bfloat16)[:, None]
            expected = (products.sum(axis=0) + down[6]) + residual
            fp32_mix = (
                products.astype(mx.float32).sum(axis=0).astype(mx.bfloat16) + down[6]
            ) + residual
            self.assertFalse(mx.array_equal(expected, fp32_mix).item())
            scores = (
                mx.concatenate([mx.zeros((8,)), probabilities[selected]])
                if ranked
                else mx.concatenate([weights, mx.zeros((8,))])
            )
            (trace,) = kernel(
                inputs=[down, scores, residual],
                output_shapes=[(4096,)],
                output_dtypes=[mx.bfloat16],
                grid=(256, 1, 1),
                threadgroup=(256, 1, 1),
            )
            self.assertTrue(mx.array_equal(trace[2048:], expected).item(), seed)


if __name__ == "__main__":
    unittest.main()
