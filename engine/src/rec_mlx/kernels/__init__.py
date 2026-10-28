def build_decoder(model, *, optimized):
    import mlx.core as mx

    # One engine per process: the transforms specialize the module's Metal factory.
    from rec_mlx.kernels import megakernel
    from rec_mlx.kernels.device import profile_for_device

    profile = profile_for_device(mx.device_info())
    from rec_mlx.kernels.expert_teams import transform as expert_teams

    megakernel._HEADER = expert_teams(megakernel._HEADER, profile.mlp_workers)

    if optimized:
        from rec_mlx.kernels.resident_optimizations import transform
        from rec_mlx.kernels.routing_attention import install

        megakernel._HEADER = transform(megakernel._HEADER)
        from rec_mlx.kernels.mixed_bits import transform as qat_transform

        def post_transform(header, body):
            header, body = qat_transform(
                header, body, model, profile.mlp_workers, profile.specialize_mixed_experts
            )
            # The shared-projection KV-team schedule covers M1 counts 12–36.
            if profile.chip == "M1 Max" and profile.workers > 8:
                if profile.workers not in range(12, 37, 4):
                    raise ValueError(
                        "Coherent resident scratch is only implemented for M1 worker counts 12–36 in steps of 4"
                    )
                from rec_mlx.kernels.coherent_scratch import transform as coherent_scratch

                header, body = coherent_scratch(header, body)
            return header, body

        captures = install(megakernel, profile, post_transform=post_transform)
    if not optimized:
        from rec_mlx.kernels.mixed_bits import install as install_qat

        install_qat(megakernel, model, profile.mlp_workers, profile.specialize_mixed_experts)
    engine = megakernel.FullModel(
        model,
        workers=profile.workers,
        mlp_workers=profile.mlp_workers,
        head_workers=profile.head_workers,
    )
    engine.device_profile = profile
    engine.prefill_chunk_size = profile.prefill_chunk_size
    if not optimized:
        return engine, None

    from rec_mlx.kernels.activation_optimization import guard
    from rec_mlx.kernels.activation_optimization import transform as fast_activation
    from rec_mlx.kernels.kv_teams import guard as guard_schedule
    from rec_mlx.kernels.sampling import ResidentSampler

    guard_schedule(engine)
    spec = captures["decode"]
    guard(megakernel, engine, spec)
    spec = dict(spec, header=fast_activation(spec["header"]))
    engine.kernel = megakernel._metal(name="flint_decode_fast_activation", **spec)
    return engine, ResidentSampler(megakernel, engine, spec)
