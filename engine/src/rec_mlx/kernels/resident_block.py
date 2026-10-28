import re

WIDTH = 8
VOCAB = 150144


def once(source, old, new):
    assert source.count(old) == 1, (source.count(old), old[:100])
    return source.replace(old, new)


ARGMAX = r"""
        // All original vocabulary-head writes precede the complete argmax.
        SYNC();
        float best=-as_type<float>(0x7f800000u);
        uint selected=0xffffffffu;
        for(uint word=index;word<150144u;word+=10240u) {
            float value=float(logits[word]);
            if(value>best || (value==best && word<selected)) {
                best=value;selected=word;
            }
        }
        float peak=simd_max(best);
        uint first=simd_min(best==peak?selected:0xffffffffu);
        if(lane==0) {
            reduction[simdgroup_index_in_threadgroup]=peak;
            layer_descriptor[simdgroup_index_in_threadgroup]=first;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(simdgroup_index_in_threadgroup==0) {
            float value=lane<8?reduction[lane]:-as_type<float>(0x7f800000u);
            uint word=lane<8?layer_descriptor[lane]:0xffffffffu;
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0) {
                statistics[group*2]=maximum;
                ((device uint*)statistics)[group*2+1]=choice;
            }
        }
        // Publish every group's maximum before any group chooses the next input.
        SYNC();
        if(simdgroup_index_in_threadgroup==0) {
            float value=-as_type<float>(0x7f800000u);
            uint word=0xffffffffu;
            for(uint peer=lane;peer<40;peer+=32) {
                float other=statistics[peer*2];
                uint other_word=((device uint*)statistics)[peer*2+1];
                if(other>value || (other==value && other_word<word)) {
                    value=other;word=other_word;
                }
            }
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0)layer_descriptor[0]=choice;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        block_input_token=layer_descriptor[0];
        if(index==0)profile[block_step*2+1]=block_input_token;
"""


def transform(
    source, workers=40, *, coherent_statistics=False, capture_group_tokens=False
):
    begin = source.index("    for(uint j=0;j<8;++j) {", source.index("Matrix embedding="))
    finish = source.index("    if(tid==0)diagnostic[group]=0;", begin)
    prefix, core, suffix = source[:begin], source[begin:finish], source[finish:]
    assert suffix.count("diagnostic[group]=0;") == 1
    assert suffix.count("profile[512*") == 1
    assert "0 && tid==0" in suffix, "Profiling must be disabled."
    assert "16384.0f" not in core
    core = once(core, "size_t row=uint(token);", "size_t row=block_input_token;")
    core = re.sub(r"\bposition\b", "block_position", core)
    core = once(
        core,
        "expert_team_handoff(routing,control,tid,group,layer,phase,ready)",
        "expert_team_handoff(routing,control,tid,group,block_step*49u+layer,phase,ready)",
    )
    assert core.count("uint(block_position)") >= 3
    loop = """
    uint block_input_token=uint(token);
    #pragma clang loop unroll(disable)
    for(uint block_step=0;block_step<8u;++block_step) {
        uint block_position=uint(position)+block_step;
        device bfloat16_t* logits=block_logits+block_step*150144u;
        device bfloat16_t* trace=block_trace+block_step*(50u*2048u);
        if(index==0)profile[block_step*2]=block_input_token;
"""
    argmax = ARGMAX
    if coherent_statistics:
        for old, new in (
            (
                "statistics[group*2]=maximum;",
                "((device coherent(device) float*)statistics)[group*2]=maximum;",
            ),
            (
                "((device uint*)statistics)[group*2+1]=choice;",
                "((device coherent(device) uint*)statistics)[group*2+1]=choice;",
            ),
            (
                "float other=statistics[peer*2];",
                "float other=((device coherent(device) float*)statistics)[peer*2];",
            ),
            (
                "uint other_word=((device uint*)statistics)[peer*2+1];",
                "uint other_word=((device coherent(device) uint*)statistics)[peer*2+1];",
            ),
        ):
            argmax = once(argmax, old, new)
        argmax = once(
            argmax,
            """        if(simdgroup_index_in_threadgroup==0) {
            float value=-as_type<float>(0x7f800000u);
            uint word=0xffffffffu;
            for(uint peer=lane;peer<40;peer+=32) {
                float other=((device coherent(device) float*)statistics)[peer*2];
                uint other_word=((device coherent(device) uint*)statistics)[peer*2+1];
                if(other>value || (other==value && other_word<word)) {
                    value=other;word=other_word;
                }
            }
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0)layer_descriptor[0]=choice;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        block_input_token=layer_descriptor[0];""",
            """        // sync_all uses control[0:3]; control[3] broadcasts the selected token.
        if(group==0 && simdgroup_index_in_threadgroup==0) {
            float value=-as_type<float>(0x7f800000u);
            uint word=0xffffffffu;
            for(uint peer=lane;peer<40;peer+=32) {
                float other=((device coherent(device) float*)statistics)[peer*2];
                uint other_word=((device coherent(device) uint*)statistics)[peer*2+1];
                if(other>value || (other==value && other_word<word)) {
                    value=other;word=other_word;
                }
            }
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0)atomic_store_explicit(control+3,choice,memory_order_relaxed);
        }
        SYNC();
        // Use the same atomic RMW polling pattern as the global barrier.
        if(tid==0)layer_descriptor[0]=atomic_fetch_add_explicit(control+3,0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        block_input_token=layer_descriptor[0];""",
        )
    if capture_group_tokens:
        argmax = once(
            argmax,
            "        if(index==0)profile[block_step*2+1]=block_input_token;",
            f"        if(tid==0)profile[17u+block_step*{workers}u+group]=block_input_token;\n"
            "        if(index==0)profile[block_step*2+1]=block_input_token;",
        )
    argmax = argmax.replace("10240u", f"{workers * 256}u").replace(
        "peer<40", f"peer<{workers}"
    )
    return prefix + loop + core + argmax + "\n    }\n    if(tid==0)diagnostic[group]=0;\n"
