#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${SIMPLEFOLD_RAW_DIR:-${repo_root}/data/simplefold_official/raw}"
out_root="${SIMPLEFOLD_OUT_ROOT:-${repo_root}/data/simplefold_official/testsets}"
eigenfold_dir="${EIGENFOLD_DIR:-${repo_root}/data/simplefold_official/upstream/EigenFold}"
eigenfold_commit="87911791549ae46c7d48b24108a5e0e9e758aa94"

mkdir -p "${raw_dir}/cameo22" "$(dirname "${eigenfold_dir}")"

download_checked() {
    local url="$1"
    local output="$2"
    local expected_sha256="$3"
    if [[ ! -f "${output}" ]]; then
        wget -O "${output}.partial" "${url}"
        mv "${output}.partial" "${output}"
    fi
    printf '%s  %s\n' "${expected_sha256}" "${output}" | sha256sum --check --status
}

download_checked \
    "https://ml-site.cdn-apple.com/models/simplefold/cameo22_predictions.zip" \
    "${raw_dir}/cameo22_predictions.zip" \
    "c3964d55cd5caa00a4fc7a4575d9f84a6df13fafab65d1193d9a4fedd18fcd72"
download_checked \
    "https://ml-site.cdn-apple.com/models/simplefold/apo_predictions.zip" \
    "${raw_dir}/apo_predictions.zip" \
    "dc2c8121d089d1d064138a537fa6640e56d394377eadbeeb79420f51b09fa4dc"
download_checked \
    "https://ml-site.cdn-apple.com/models/simplefold/codnas_predictions.zip" \
    "${raw_dir}/codnas_predictions.zip" \
    "f686d883df7df6622330df671ca645af4567d73fb5aa12345300c366091b2326"
download_checked \
    "https://files.rcsb.org/download/8QCW.cif" \
    "${raw_dir}/cameo22/8qcw.cif" \
    "a118995c6ee5e685d83be8f03e1f19cbe6145e185f32553289635baf75ed424e"

if [[ ! -d "${eigenfold_dir}/.git" ]]; then
    git clone https://github.com/bjing2016/EigenFold.git "${eigenfold_dir}"
fi
if [[ "$(git -C "${eigenfold_dir}" rev-parse HEAD)" != "${eigenfold_commit}" ]]; then
    git -C "${eigenfold_dir}" fetch origin "${eigenfold_commit}"
    git -C "${eigenfold_dir}" checkout --detach "${eigenfold_commit}"
fi

"${repo_root}/.venv/bin/python" \
    "${repo_root}/scripts/prepare_simplefold_testsets.py" \
    --raw-dir "${raw_dir}" \
    --eigenfold-dir "${eigenfold_dir}" \
    --out-root "${out_root}"
