"""Bind completed trials to an independently sealed CPU-input replay."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MICROBATCHES = 40000
BUDGET = dict(updates=10000, attempts=10000, skipped_updates=0,
              microbatches=MICROBATCHES, finite_losses=MICROBATCHES)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs(path, count):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != count or [row["micro"] for row in rows] != list(range(1, count + 1)):
        raise ValueError("input audit has missing, duplicate or reordered microbatches")
    stream = hashlib.sha256()
    values = []
    for row in rows:
        value = row["sha256"]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("invalid batch SHA256")
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("invalid batch SHA256") from exc
        if len(raw) != 32:
            raise ValueError("invalid batch SHA256")
        stream.update(raw)
        values.append(value)
    return tuple(values), stream.hexdigest()


def load_certificate(run, path, expected_sha256):
    run, path = Path(run).resolve(), Path(path).resolve()
    if len(path.parents) < 3 or path.parents[2] != run or sha256(path) != expected_sha256:
        raise ValueError("input certificate is outside this run or differs from its seal")
    proof = path.parent
    certificate = json.loads(path.read_text())
    protocol_path = proof / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (certificate.get("complete") is not True
            or certificate.get("microbatches") != MICROBATCHES
            or certificate.get("all_non_coordinates_exact") is not True
            or certificate.get("retained_actual_all40000_verified") is not True
            or certificate.get("protocol_sha256") != sha256(protocol_path)
            or protocol.get("microbatches") != MICROBATCHES):
        raise ValueError("incomplete input replay or changed replay protocol")
    bound = protocol.get("atol")
    if (not isinstance(bound, (int, float)) or not math.isfinite(bound)
            or not 0 < bound <= 1e-4
            or protocol.get("rtol") != 0 or certificate.get("rtol") != 0
            or certificate.get("atol") != bound
            or protocol.get("coord_scale") != 16 or certificate.get("coord_scale") != 16
            or not math.isclose(certificate.get("absolute_bound_A", math.nan), bound * 16)):
        raise ValueError("coordinate equivalence bound differs")
    sources = {str(proof / name) for name in ("replay.py", "run.sh", "test_replay.py")}
    if set(protocol.get("source_sha256", {})) != sources or any(
        sha256(name) != value for name, value in protocol["source_sha256"].items()
    ):
        raise ValueError("replay source differs from its admitted protocol")
    statistics = certificate.get("coordinate_statistics", {})
    if set(statistics) != {"x_clean", "x_t"}:
        raise ValueError("missing coordinate replay statistics")
    for value in statistics.values():
        scalars = [value.get(name, math.nan) for name in ("max_abs", "rms", "sum_squares")]
        if (type(value.get("elements")) is not int or value["elements"] < MICROBATCHES
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                           for v in scalars)
                or value["max_abs"] > bound or value["rms"] > bound
                or not math.isclose(value["rms"],
                                    math.sqrt(value["sum_squares"] / value["elements"]),
                                    rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError("invalid or excessive coordinate replay error")
    streams = {}
    raw_streams = certificate.get("raw_streams", {})
    if set(raw_streams) != {"retained", "new"}:
        raise ValueError("missing certified raw input streams")
    for role in raw_streams:
        receipt = json.loads((proof / f"{role}-complete.json").read_text())
        inputs = proof / f"{role}-inputs.jsonl"
        values, digest = read_inputs(inputs, MICROBATCHES)
        if (receipt.get("microbatches") != MICROBATCHES
                or receipt.get("inputs_sha256") != sha256(inputs)
                or receipt.get("input_stream_sha256") != digest or raw_streams[role] != digest):
            raise ValueError("certified raw stream identity differs")
        if digest in streams and streams[digest] != values:
            raise ValueError("ambiguous raw input stream")
        streams[digest] = values
    return certificate, streams


def verify_training_inputs(run, results, certificate_path, certificate_sha256):
    certificate, streams = load_certificate(run, certificate_path, certificate_sha256)
    protocol = json.loads((Path(run) / "protocol.json").read_text())
    if set(results) != set(protocol["arms"]):
        raise ValueError("incomplete arm coverage for input equivalence")
    bindings = {}
    for arm, result in results.items():
        entry = protocol["arms"][arm]
        training = Path(entry["training"])
        complete = json.loads((training / "audit/complete.json").read_text())
        recorded = result["training"]
        if any(complete.get(key) != value or recorded.get(key) != value
               for key, value in BUDGET.items()):
            raise ValueError(f"incomplete training budget: {arm}")
        values, digest = read_inputs(training / "audit/inputs.jsonl", MICROBATCHES)
        if (digest != complete.get("input_stream_sha256")
                or digest != recorded.get("input_stream_sha256")
                or digest not in streams or values != streams[digest]):
            raise ValueError(f"uncertified delivered input stream: {arm}")
        smoke, _ = read_inputs(Path(entry["preflight"]) / "audit/inputs.jsonl", 32)
        if (tuple(recorded.get("probe_input_sha256", ())) != values[:16]
                or smoke[:16] != values[:16]):
            raise ValueError(f"probe inputs do not bind to training: {arm}")
        bindings[arm] = digest
    return dict(
        method="certified_coordinate_equivalence", certificate_sha256=certificate_sha256,
        certificate=str(Path(certificate_path).resolve()), microbatches_per_arm=MICROBATCHES,
        atol=certificate["atol"], rtol=0, absolute_bound_A=certificate["absolute_bound_A"],
        coordinate_statistics=certificate["coordinate_statistics"], arm_streams=bindings,
        limits="Only CPU coordinate roundoff is bounded; noise/time/ESM/masks are exact. "
        "This does not establish identical gradients, CUDA arithmetic or optimizer trajectories.",
    )
