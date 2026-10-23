def replace_body(source, name, body):
    start = source.index("{", source.index(name + "("))
    end, depth = start + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[: start + 1] + "\n" + body + "\n" + source[end - 1 :]


def transform(header, *, rms=True, monotonic=True):
    if rms:
        header = replace_body(
            header,
            "rms_256_w4",
            r"""
    float sums[2];
    for(uint leaf_set=0;leaf_set<2;++leaf_set) {
        float squared=0.0f;
        for(uint j=0;j<4;++j){float v=float(x[leaf_set*4+j]);squared+=v*v;}
        sums[leaf_set]=simd_sum(squared);
    }
    if(lane==0){temp[sg]=sums[0];temp[sg+8]=sums[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float leaf=lane<16?temp[lane]:0.0f;
    float total=simd_sum(leaf);
    float inverse=metal::precise::rsqrt(total/2048.0f+eps);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[i]=w[i]*bfloat16_t(float(x[j])*inverse);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
""",
        )
    if monotonic:
        header = replace_body(
            header,
            "sync_all",
            r"""
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*__WORKERS__;
        uint arrived=atomic_fetch_add_explicit(control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_fetch_add_explicit(control,0u,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        if(__PROFILE__) {uint at=(phase*__WORKERS__+group)*2;profile[at]=spins;profile[at+1]=arrived-phase*__WORKERS__;}
        ready[0]=uint(atomic_fetch_add_explicit(control,0u,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;
""",
        )
    return header
