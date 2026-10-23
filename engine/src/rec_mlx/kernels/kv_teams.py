from .shader_source import replace_body


def once(text, old, new, count=1):
    actual = text.count(old)
    if actual != count:
        raise ValueError(f"Expected {count} anchors, found {actual}: {old}")
    return text.replace(old, new)


def function(text, name):
    at = text.index(name + "(")
    begin = text.rfind("inline ", 0, at)
    start = text.index("{", at)
    end, depth = start + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


PROJECTION_BLOCK = r"""
template<uint K>
inline void q4_local_block(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint first) {
    float a[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint block=0;block<K;block+=512) {
        uint scalar=block+lane*16;
        float v[16];float sum=load16_local(x+scalar,v);
        for(uint r=0;r<4;++r) {
            uint row=first+r;uint gi=row*(K/64)+scalar/64;
            a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
        }
    }
    for(uint r=0;r<4;++r) {
        float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
    }
}
"""


def shared_projection(body, workers, groups):
    body = once(body, "uint kv_phase=0;", "uint kv_phase=0;uint kv_qkv_base=0;")
    loop_begin = body.index("        for(uint work=group;work<40;")
    begin = body.rindex("        // Each group owns a full Q, K or V head", 0, loop_begin)
    end = body.index("        SYNC();\n        bool rounded=")
    loop = body[loop_begin:end]
    norm = loop.index("            if(simdgroup_index_in_threadgroup==0 && head<36)")
    norm = loop[norm : loop.index(";", norm) + 1]
    heads = (
        "uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;\n"
        "            uint local=head<32?head:head<36?head-32:head-36;\n"
    )
    shared = f"""        uint kv_team=group/{groups},kv_member=group%{groups};
        for(;;) {{
            uint ticket=0;
            if(lane==0)ticket=atomic_fetch_add_explicit(control+40+kv_team,1u,memory_order_relaxed);
            ticket=simd_broadcast(ticket,0);
            if(ticket>=kv_qkv_base+320u)break;
            uint unit=ticket-kv_qkv_base,member=unit/32u;
            {heads}            Matrix m=weight_matrix(bank,d+(head<32?10:head<36?15:20));
            m.w+=size_t(local)*128*1024;m.s+=local*128*32;m.b+=local*128*32;m.n=128;
            q4_local_block<2048>(m,n,(head<32?q:head<36?k:v)+local*128,lane,(unit%32u)*4u);
        }}
        kv_qkv_base+={320 + groups * 8}u;
        KV_TEAM_SYNC();
        for(uint member=kv_member;member<10;member+={groups}) {{
            {heads}            device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
{norm}
            threadgroup_barrier(mem_flags::mem_device);
            if(head>=32) {{
                KVView target=head<36?keys:vals;
                for(uint dim=tid;dim<128;dim+=256)target.store(local,insert,dim,out[dim]);
            }}
        }}
        KV_TEAM_SYNC();
        bool rounded="""
    return body[:begin] + shared + body[end + len("        SYNC();\n        bool rounded=") :]


def transform(header, body, workers=40):
    groups = workers // 4
    simds = groups * 8
    sync = function(header, "sync_all")
    assert "if(0)" in sync, "Global profiling must be disabled"
    sync = once(sync, "sync_all(", "kv_team_sync(")
    sync = once(
        sync,
        "uint group) {\n",
        f"uint group) {{\n    device atomic_uint* team_control=control+4+(group/{groups})*2;\n",
    )
    sync = once(sync, f"uint goal=(phase+1u)*{workers};", f"uint goal=(phase+1u)*{groups};")
    sync = once(
        sync,
        "atomic_fetch_add_explicit(control,1u,memory_order_relaxed)",
        "atomic_fetch_add_explicit(team_control,1u,memory_order_relaxed)",
    )
    sync = once(
        sync,
        "atomic_fetch_add_explicit(control,0u,memory_order_relaxed)",
        "atomic_fetch_add_explicit(team_control,0u,memory_order_relaxed)",
        count=2,
    )
    header += "\n" + sync + "\n"

    def partial(code):
        return once(
            code,
            f"for(uint task=sg;task<(32/HEADS)*parts;task+={workers * 8})",
            f"for(uint task=(sg/{simds})*((8/HEADS)*parts)+sg%{simds};"
            f"task<(sg/{simds}+1)*((8/HEADS)*parts);task+={simds})",
        )

    def merge(code):
        return once(
            code,
            f"for(uint task=sg;task<32*8;task+={workers * 8})",
            f"for(uint task=(sg/{simds})*64+sg%{simds};task<(sg/{simds}+1)*64;task+={simds})",
        )

    header = replace_body(header, "attention_partial_path_w53", partial)
    header = replace_body(header, "attention_merge_vec4_w2", merge)
    body = once(body, "uint phase=0;", "uint phase=0;uint kv_phase=0;")
    macro = "#define KV_TEAM_SYNC() if(!kv_team_sync(control,tid,kv_phase,ready,profile,group)){if(tid==0)diagnostic[group]=atomic_fetch_add_explicit(control+2,0u,memory_order_relaxed);return;}"
    body = macro + "\n" + body
    if groups >= 10:
        body = once(
            body,
            f"for(uint work=group;work<40;work+={workers})",
            f"for(uint work=group;work<{workers};work+={workers})",
        )
        body = once(
            body,
            "uint head=0?(work<8?work+32:work-8):work;",
            f"uint kv_team=work/{groups},kv_member=work%{groups};\n"
            "            if(kv_member>=10)continue;\n"
            "            uint head=kv_member<8?kv_team*8+kv_member:"
            "kv_member==8?32+kv_team:36+kv_team;",
        )
        body = once(body, "SYNC();\n        bool rounded=", "KV_TEAM_SYNC();\n        bool rounded=")
    else:
        header += "\n" + PROJECTION_BLOCK + "\n"
        body = shared_projection(body, workers, groups)
    body = once(
        body,
        f"SYNC();\n        if(group<{workers}) {{if(rounded) attention_merge_vec4_w2",
        f"KV_TEAM_SYNC();\n        if(group<{workers}) {{if(rounded) attention_merge_vec4_w2",
    )
    assert "SYNC();\n        q4<4096>" in body
    return header, body


def guard(engine):
    profile = engine.device_profile
    expected = {
        "workers": profile.workers,
        "attention": profile.workers,
        "merge": profile.workers,
        "router": profile.workers,
        "mlp": profile.mlp_workers,
        "head": profile.head_workers,
        "profile": False,
        "kv_first": False,
    }
    for key, value in expected.items():
        if engine.schedule[key] != value:
            raise ValueError(f"Unsupported KV-team schedule: {key}")
