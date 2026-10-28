"""Qualify resident scratch buffers for cross-threadgroup M1 memory visibility."""


def _replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one coherent-scratch anchor, found {count}: {old}")
    return text.replace(old, new, 1)


def _replace_count(text, old, new, count):
    actual = text.count(old)
    if actual != count:
        raise ValueError(f"Expected {count} coherent-scratch anchors, found {actual}: {old}")
    return text.replace(old, new)


def _replace_function(text, name, replacements):
    at = text.index(name + "(")
    start = text.rfind("inline ", 0, at)
    brace = text.index("{", start)
    depth = 0
    for end in range(brace, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                function = text[start : end + 1]
                for old, new in replacements:
                    function = _replace_once(function, old, new)
                return text[:start] + function + text[end + 1 :]
    raise ValueError(f"Unclosed Metal function: {name}")


def _extract_function(text, anchor):
    start = text.index(anchor)
    brace = text.index("{", start)
    depth = 0
    for end in range(brace, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return start, end + 1, text[start : end + 1]
    raise ValueError(f"Unclosed Metal function: {anchor}")


def _append_coherent_q4_helpers(header):
    if "q4_coherent(" in header:
        raise ValueError("Coherent Q4 helpers are already present")

    _, _, load16 = _extract_function(header, "inline float load16(")
    load16 = load16.replace("inline float load16(", "inline float load16_coherent(", 1)
    load16 = _replace_once(
        load16,
        "const device bfloat16_t* x",
        "const device coherent(device) bfloat16_t* x",
    )
    _, _, q4 = _extract_function(header, "template<uint K>\ninline void q4(")
    q4 = q4.replace("inline void q4(", "inline void q4_coherent(", 1)
    q4 = _replace_once(
        q4,
        "const device bfloat16_t* x,device bfloat16_t* out",
        "const device coherent(device) bfloat16_t* x,device coherent(device) bfloat16_t* out",
    )
    q4 = _replace_once(q4, "load16(x+scalar,v)", "load16_coherent(x+scalar,v)")

    load_start, load_end, _ = _extract_function(header, "inline float load16(")
    del load_start
    header = header[:load_end] + "\n" + load16 + header[load_end:]
    return header + "\n" + q4 + "\n"


def transform(header, body):
    """Make shared decode intermediates coherent across resident threadgroups.

    M1 Max barrier stress tests observed stale peer payloads through ordinary
    device pointers despite device fences. The coherent(device) qualifier on both
    producer and consumer pointers passed those same publication checks.
    """
    header_replacements = (
        (
            "const device bfloat16_t *p0, *p1, *p2, *p3;",
            "const device coherent(device) bfloat16_t *p0, *p1, *p2, *p3;",
        ),
        (
            "const device bfloat16_t* address(uint i) const",
            "const device coherent(device) bfloat16_t* address(uint i) const",
        ),
        (
            "const device bfloat16_t* p=block==0?p0:block==1?p1:block==2?p2:p3;",
            "const device coherent(device) bfloat16_t* p=block==0?p0:block==1?p1:block==2?p2:p3;",
        ),
        (
            "*((device bfloat16_t*)this->head(head,dim).address(position*128))=value;",
            "*((device coherent(device) bfloat16_t*)this->head(head,dim).address(position*128))=value;",
        ),
        (
            "inline void q4_local_block(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,",
            "inline void q4_local_block(Matrix m,const threadgroup bfloat16_t* x,device coherent(device) bfloat16_t* out,",
        ),
        (
            "inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,",
            "inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device coherent(device) bfloat16_t* out,",
        ),
        (
            "inline void head_norm(device bfloat16_t* x,const device bfloat16_t* w,uint heads,",
            "inline void head_norm(device coherent(device) bfloat16_t* x,const device bfloat16_t* w,uint heads,",
        ),
        (
            "device bfloat16_t* p=x+h*128;",
            "device coherent(device) bfloat16_t* p=x+h*128;",
        ),
        (
            "inline void attention_merge_vec4_w2(device bfloat16_t* out,\n                      device float* partial,device float* stats,",
            "inline void attention_merge_vec4_w2(device coherent(device) bfloat16_t* out,\n                      device coherent(device) float* partial,device coherent(device) float* stats,",
        ),
        (
            "inline void attention_partial_path_w53(const device bfloat16_t* q,KVView k,",
            "inline void attention_partial_path_w53(const device coherent(device) bfloat16_t* q,KVView k,",
        ),
        (
            "                      device bfloat16_t* out,device float* partial,device float* stats,\n                      uint count,uint cap,uint parts,bool runtime_rounded,uint lane,uint sg) {",
            "                      device coherent(device) bfloat16_t* out,device coherent(device) float* partial,device coherent(device) float* stats,\n                      uint count,uint cap,uint parts,bool runtime_rounded,uint lane,uint sg) {",
        ),
        (
            "const device vec<bfloat16_t,4>*)(((const device bfloat16_t*)partial)+at)",
            "const device coherent(device) vec<bfloat16_t,4>*)(((const device coherent(device) bfloat16_t*)partial)+at)",
        ),
        (
            "const device float4*)(partial+at)",
            "const device coherent(device) float4*)(partial+at)",
        ),
    )
    for old, new in header_replacements:
        header = _replace_once(header, old, new)

    header = _replace_function(
        header,
        "gateup_all",
        [
            ("const device uint* ids,", "const device coherent(device) uint* ids,"),
            ("device bfloat16_t* out,uint lane,uint sg)", "device coherent(device) bfloat16_t* out,uint lane,uint sg)"),
        ],
    )
    header = _replace_function(
        header,
        "down_all_w197_bias23",
        [
            (
                "const device bfloat16_t* x,const device uint* ids,\n                     device bfloat16_t* out,",
                "const device coherent(device) bfloat16_t* x,const device coherent(device) uint* ids,\n                     device coherent(device) bfloat16_t* out,",
            ),
            (
                "const device bfloat16_t* input=x+role*512;",
                "const device coherent(device) bfloat16_t* input=x+role*512;",
            ),
            ("load16(input+scalar,v)", "load16_coherent(input+scalar,v)"),
        ],
    )
    header = _replace_function(
        header,
        "expert_team_gateup",
        [
            ("const device uint* ids,", "const device coherent(device) uint* ids,"),
            (
                "device bfloat16_t* out,\n                      uint lane,",
                "device coherent(device) bfloat16_t* out,\n                      uint lane,",
            ),
        ],
    )
    header = _replace_function(
        header,
        "expert_team_down",
        [
            (
                "const device bfloat16_t* x,const device uint* ids,\n                     device bfloat16_t* out,",
                "const device coherent(device) bfloat16_t* x,const device coherent(device) uint* ids,\n                     device coherent(device) bfloat16_t* out,",
            ),
            (
                "const device bfloat16_t* input=x+role*512;",
                "const device coherent(device) bfloat16_t* input=x+role*512;",
            ),
            ("load16(input+scalar,v)", "load16_coherent(input+scalar,v)"),
        ],
    )
    header = _replace_function(
        header,
        "route_dot_bf16",
        [("device float* probs,uint lane,uint sg)", "device coherent(device) float* probs,uint lane,uint sg)")],
    )
    header = _replace_function(
        header,
        "route_top_lane_scores",
        [
            ("device float* probs,", "device coherent(device) float* probs,"),
            (
                "device uint* ids,device float* scores,",
                "device coherent(device) uint* ids,device coherent(device) float* scores,",
            ),
        ],
    )
    if "inline void w3_route_own_rank(" in header:
        header = _replace_function(
            header,
            "w3_route_own_rank",
            [
                ("device float* probs,", "device coherent(device) float* probs,"),
                (
                    "device float* chosen,device uint* ids,",
                    "device coherent(device) float* chosen,device coherent(device) uint* ids,",
                ),
            ],
        )

    query_base = (
        "        KVView value_base=v.head(kv,lane*4);\n"
        "        const device bfloat16_t* query_base=q+first*128+lane*4;"
    )
    header = _replace_once(
        header,
        query_base,
        query_base.replace("const device bfloat16_t*", "const device coherent(device) bfloat16_t*"),
    )
    partial_start = header.index("inline void attention_partial_path_w53(")
    partial_path = header[partial_start:]
    for old, new in (
        ("device float* stat_out=stats+", "device coherent(device) float* stat_out=stats+"),
        (
            "device bfloat16_t* rounded_out=rounded?((device bfloat16_t*)partial)+",
            "device coherent(device) bfloat16_t* rounded_out=rounded?((device coherent(device) bfloat16_t*)partial)+",
        ),
        (":(device bfloat16_t*)partial;", ":(device coherent(device) bfloat16_t*)partial;"),
        ("device float* unrounded_out=rounded?partial:", "device coherent(device) float* unrounded_out=rounded?partial:"),
        (
            "*((device vec<bfloat16_t,4>*)(rounded_out+j))=",
            "*((device coherent(device) vec<bfloat16_t,4>*)(rounded_out+j))=",
        ),
        (
            "*((device float4*)(unrounded_out+j))=",
            "*((device coherent(device) float4*)(unrounded_out+j))=",
        ),
    ):
        partial_path = _replace_once(partial_path, old, new)
    header = header[:partial_start] + partial_path
    header = _append_coherent_q4_helpers(header)

    source_replacements = (
        ("device bfloat16_t* q=workspace+4096;", "device coherent(device) bfloat16_t* q=(device coherent(device) bfloat16_t*)workspace+4096;"),
        ("device bfloat16_t* k=workspace+8192;", "device coherent(device) bfloat16_t* k=(device coherent(device) bfloat16_t*)workspace+8192;"),
        ("device bfloat16_t* v=workspace+8704;", "device coherent(device) bfloat16_t* v=(device coherent(device) bfloat16_t*)workspace+8704;"),
        ("device bfloat16_t* a=workspace+9216;", "device coherent(device) bfloat16_t* a=(device coherent(device) bfloat16_t*)workspace+9216;"),
        ("device bfloat16_t* o=workspace+13312;", "device coherent(device) bfloat16_t* o=(device coherent(device) bfloat16_t*)workspace+13312;"),
        ("device bfloat16_t* gate=workspace+15360;", "device coherent(device) bfloat16_t* gate=(device coherent(device) bfloat16_t*)workspace+15360;"),
        ("device bfloat16_t* up=workspace+31744;", "device coherent(device) bfloat16_t* up=(device coherent(device) bfloat16_t*)workspace+31744;"),
        ("device bfloat16_t* act=workspace+48128;", "device coherent(device) bfloat16_t* act=(device coherent(device) bfloat16_t*)workspace+48128;"),
        ("device bfloat16_t* down=workspace+64512;", "device coherent(device) bfloat16_t* down=(device coherent(device) bfloat16_t*)workspace+64512;"),
        ("device uint* ids=routing+group*16;", "device coherent(device) uint* ids=(device coherent(device) uint*)(routing+group*16);"),
        ("device float* scores=(device float*)(routing+group*16+8);", "device coherent(device) float* scores=(device coherent(device) float*)(routing+group*16+8);"),
        (
            "KVView keys={kv0,kv1,kv2,kv3,d[61],cap,global};",
            "KVView keys={(const device coherent(device) bfloat16_t*)kv0,(const device coherent(device) bfloat16_t*)kv1,(const device coherent(device) bfloat16_t*)kv2,(const device coherent(device) bfloat16_t*)kv3,d[61],cap,global};",
        ),
    )
    for old, new in source_replacements:
        body = _replace_once(body, old, new)
    body = _replace_once(
        body,
        "device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;",
        "device coherent(device) bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;",
    )
    body = _replace_count(
        body,
        ",a,partials,statistics,count",
        ",a,(device coherent(device) float*)partials,(device coherent(device) float*)statistics,count",
        4,
    )
    body = _replace_count(
        body,
        "(a,partials,statistics,parts,lane,sg)",
        "((device coherent(device) bfloat16_t*)a,(device coherent(device) float*)partials,(device coherent(device) float*)statistics,parts,lane,sg)",
        2,
    )
    body = _replace_once(
        body,
        "q4<4096>(weight_matrix(bank,d+25),a,o,lane,sg,",
        "q4_coherent<4096>(weight_matrix(bank,d+25),a,o,lane,sg,",
    )
    body = _replace_once(
        body,
        "route_dot_bf16(n,(const device bfloat16_t*)(bank+d[60]),statistics,lane,sg);",
        "route_dot_bf16(n,(const device bfloat16_t*)(bank+d[60]),(device coherent(device) float*)statistics,lane,sg);",
    )
    route_top = "route_top_lane_scores<true>((const device float*)(bank+d[9]),statistics,ids,scores,lane,simdgroup_index_in_threadgroup);"
    if route_top in body:
        body = _replace_once(
            body,
            route_top,
            "route_top_lane_scores<true>((const device float*)(bank+d[9]),(device coherent(device) float*)statistics,ids,scores,lane,simdgroup_index_in_threadgroup);",
        )
    else:
        body = _replace_once(
            body,
            "w3_route_own_rank((const device float*)(bank+d[9]),statistics,(device float*)(routing+8),ids,lane,simdgroup_index_in_threadgroup,group);",
            "w3_route_own_rank((const device float*)(bank+d[9]),(device coherent(device) float*)statistics,(device coherent(device) float*)(routing+8),ids,lane,simdgroup_index_in_threadgroup,group);",
        )
        body = _replace_once(
            body,
            "((device float*)(routing+8))[lane]",
            "((device coherent(device) float*)(routing+8))[lane]",
        )
    body = _replace_once(
        body,
        "q4<6144>(dm,act,down,lane,sg,",
        "q4_coherent<6144>(dm,act,down,lane,sg,",
    )
    return header, body
