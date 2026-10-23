from .kv_teams import function


def transform(header, mlp_workers):
    assert mlp_workers in (8, 32, 64)
    team = mlp_workers // 8
    for name in ("expert_team_gateup", "expert_team_down", "expert_team_handoff"):
        old = function(header, name)
        new = old.replace("group>=24", f"group>={team * 6}")
        new = new.replace("group-24", f"group-{team * 6}")
        new = new.replace("group/4", f"group/{team}").replace("group%4", f"group%{team}")
        new = new.replace("shared?8:4", f"shared?{team * 2}:{team}")
        new = new.replace("?24:", f"?{team * 6}:")
        new = new.replace(")*4, count=", f")*{team}, count=")
        new = new.replace("?8:4;", f"?{team * 2}:{team};")
        header = header.replace(old, new)
    return header
