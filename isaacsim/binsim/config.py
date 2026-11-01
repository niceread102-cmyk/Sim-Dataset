"""Experiment config: YAML with `defaults:` includes, per-part files and dotted command line overrides (key=value), merged in that order."""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml

PATH_KEYS = {"mesh", "usd", "collision_mesh", "output_dir", "cache_dir", "experience", "grasp_file"}


class Cfg(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}' (have: {list(self.keys())})") from e

    def __setattr__(self, key, value):
        self[key] = value

    def get_path(self, dotted: str, default=None):
        node: Any = self
        for k in dotted.split("."):
            if isinstance(node, dict) and k in node:
                node = node[k]
            elif isinstance(node, list) and k.isdigit() and int(k) < len(node):
                node = node[int(k)]
            else:
                return default
        return node


def _wrap(obj):
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_paths(node, base_dir: str):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in PATH_KEYS and isinstance(v, str) and v:
                node[k] = resolve_path(v, base_dir)
            else:
                _resolve_paths(v, base_dir)
    elif isinstance(node, list):
        for v in node:
            _resolve_paths(v, base_dir)
    return node


def resolve_path(p: str, base_dir: str) -> str:
    if p.startswith("{") or "://" in p:
        return p
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _load_yaml_tree(path: str, _stack=()) -> dict:
    path = os.path.abspath(path)
    if path in _stack:
        raise ValueError(f"circular `defaults:` include: {' -> '.join(_stack + (path,))}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base_dir = os.path.dirname(path)
    defaults = data.pop("defaults", []) or []
    merged: dict = {}
    for inc in defaults:
        inc_path = resolve_path(inc if inc.endswith((".yaml", ".yml")) else inc + ".yaml", base_dir)
        merged = deep_merge(merged, _load_yaml_tree(inc_path, _stack + (path,)))
    _resolve_paths(data, base_dir)
    return deep_merge(merged, data)


def _expand_parts(parts: list, config_dir: str) -> list:
    out = []
    for i, entry in enumerate(parts or []):
        entry = dict(entry)
        part_file = entry.pop("config", None)
        part = {}
        if part_file:
            part_file = resolve_path(part_file, config_dir)
            part = _load_yaml_tree(part_file)
        part = deep_merge(part, entry)
        part.setdefault("name", f"part{i}")
        out.append(part)
    return out


def apply_override(tree: dict, dotted: str, value: Any):
    keys = dotted.split(".")
    node: Any = tree
    for k in keys[:-1]:
        if isinstance(node, list):
            node = node[int(k)]
        else:
            node = node.setdefault(k, {})
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def parse_overrides(items: Iterable[str]) -> list[tuple[str, Any]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"override '{it}' must look like key.sub=value")
        k, v = it.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _find_part_file(obj: str, config_dir: str) -> str:
    path = os.path.expanduser(obj)
    if path.endswith((".yaml", ".yml")) and os.path.isfile(path):
        return os.path.abspath(path)
    parts_dir = os.path.normpath(os.path.join(config_dir, "..", "object"))
    names = sorted(f for f in os.listdir(parts_dir) if f.endswith((".yaml", ".yml"))) if os.path.isdir(parts_dir) else []
    want = _slug(os.path.splitext(os.path.basename(obj))[0] if obj.endswith((".yaml", ".yml")) else obj)
    for f in names:
        if _slug(os.path.splitext(f)[0]) == want:
            return os.path.join(parts_dir, f)
    raise ValueError(f"obj:={obj}: no part yaml in {parts_dir} (have: {', '.join(os.path.splitext(f)[0] for f in names if f != 'bin.yaml')}); "
                     f"make one with make_object_config.py obj:=<name>")


def expand_shorthand(items: Iterable[str], config_path: str) -> list[str]:
    obj = num = out = None
    extra = []
    rest = []
    for it in items or []:
        if it.startswith("scenes:="):
            extra.append(f"run.num_scenes={it[len('scenes:='):]}")
        elif it.startswith("views:="):
            extra.append(f"run.views_per_scene={it[len('views:='):]}")
        elif it.startswith("obj:="):
            obj = it[len("obj:="):]
        elif it.startswith("obj_num:="):
            num = it[len("obj_num:="):]
        elif it.startswith("out:="):
            out = it[len("out:="):]
        elif it.split("=", 1)[0].endswith(":") and it.split(":=", 1)[0].isidentifier():
            raise ValueError(f"unknown short word '{it}' (known: obj:=, obj_num:=, out:=, scenes:=, views:=)")
        else:
            rest.append(it)
    if num is not None and obj is None:
        raise ValueError("obj_num:= needs obj:=<part> (to change the count of map.yaml's part use parts.0.count=[min,max])")
    new = list(extra)
    if obj is not None:
        part_file = _find_part_file(obj, os.path.dirname(os.path.abspath(config_path)))
        new.append(f"parts.0.config={part_file}")
        count = yaml.safe_load(num) if num is not None else (yaml.safe_load(open(part_file)) or {}).get("count")
        if count is not None:
            count = [count, count] if isinstance(count, int) else list(count)
            new.append(f"parts.0.count={count}")
    if out is not None:
        new.append(f"run.output_dir={out}")
    if new:
        print("[INFO] short words -> " + " ".join(new), flush=True)
    return new + rest


def load_config(path: str, overrides: Iterable[str] = ()) -> Cfg:
    tree = _load_yaml_tree(path)
    for k, v in parse_overrides(overrides):
        apply_override(tree, k, v)
    config_dir = os.path.dirname(os.path.abspath(path))
    tree["parts"] = _expand_parts(tree.get("parts", []), config_dir)
    for k, v in parse_overrides(overrides):
        if k.startswith("parts."):
            apply_override(tree, k, v)
    for key in ("output_dir", "cache_dir"):
        for section in ("run", "assets"):
            v = tree.get(section, {}).get(key)
            if isinstance(v, str):
                tree[section][key] = resolve_path(v, config_dir)
    cfg = _wrap(tree)
    validate(cfg)
    return cfg


def _range(v, name, lo=None):
    if isinstance(v, (int, float)):
        v = [v, v]
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1]):
        raise ValueError(f"{name} must be a number or [min, max] with min <= max, got {v!r}")
    if lo is not None and v[0] < lo:
        raise ValueError(f"{name} must be >= {lo}, got {v!r}")
    return tuple(v)


def validate(cfg: Cfg):
    errs = []

    def need(dotted):
        if cfg.get_path(dotted) is None:
            errs.append(f"missing required key: {dotted}")

    for k in ("run.num_envs", "run.num_scenes", "run.views_per_scene", "bin", "camera.resolution",
              "camera.intrinsics", "loading.mode"):
        need(k)
    if not cfg.get("parts"):
        errs.append("`parts` must list at least one part")
    for i, p in enumerate(cfg.get("parts", [])):
        if not p.get("mesh"):
            errs.append(f"parts[{i}] ({p.get('name')}) needs `mesh` (STEP/STL/OBJ/PLY path)")
        try:
            _range(p.get("count", 1), f"parts[{i}].count", lo=0)
        except ValueError as e:
            errs.append(str(e))
    mode = cfg.get_path("loading.mode")
    if mode not in ("random", "layers", "mixed", "stack"):
        errs.append(f"loading.mode must be random|layers|mixed|stack, got {mode!r}")
    b = cfg.get("bin", {})
    if not b.get("mesh") and not b.get("size_m"):
        errs.append("bin needs either `mesh` or `size_m` ([x, y, z] outer size in meters)")
    if b.get("mesh") and not b.get("inner_size_m"):
        errs.append("bin.mesh given: also set bin.inner_size_m ([x, y, z] usable inner volume)")
    intr = cfg.get_path("camera.intrinsics") or {}
    if intr and not ({"fx", "fy"} <= set(intr) or "hfov_deg" in intr):
        errs.append("camera.intrinsics needs fx/fy(/cx/cy) or hfov_deg")
    if errs:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(errs))


def count_range(part) -> tuple[int, int]:
    lo, hi = _range(part.get("count", 1), "count", lo=0)
    return int(lo), int(hi)


def as_range(v, name="value") -> tuple[float, float]:
    return _range(v, name)


def dump_config(cfg: Cfg, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False)
