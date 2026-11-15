#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Scientific integrity audit, Step 1B

Scope
-----
1. Cross-split near-duplicate screening on ORIGINAL images.
2. Verification that tiled patches remain attached to their original split.
3. Exact patch-duplicate screening across tiled splits.
4. Search for evidence of the exact YOLO pretrained-weight loading command.
5. Search for tiling/preparation scripts that document the data preparation order.

READ-ONLY:
- No dataset/model/result file is modified.
- Writes only audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--dataset-root", type=Path, default=Path("dataset_avions"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/scientific_audit/step1b_near_duplicates_tiling"),
    )
    return p.parse_args()


def sha256_file(path: Path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def list_images(d: Path):
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_gray(path: Path, size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def phash64(gray):
    x = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(x)
    block = dct[:8, :8].copy()
    vals = block.flatten()
    med = np.median(vals[1:])
    bits = vals > med
    return bits.astype(np.uint8)


def dhash64(gray):
    x = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (x[:, 1:] > x[:, :-1]).flatten()
    return bits.astype(np.uint8)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def corrcoef_resized(a, b):
    x = a.astype(np.float32).reshape(-1)
    y = b.astype(np.float32).reshape(-1)
    x -= x.mean()
    y -= y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / den)


def orb_similarity(path_a: Path, path_b: Path):
    a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return {"kp_a": 0, "kp_b": 0, "good_matches": 0, "inlier_ratio": 0.0}

    # Limit large aerial images while preserving local structure.
    max_side = 1600
    for name, img in (("a", a), ("b", b)):
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            if name == "a":
                a = resized
            else:
                b = resized

    orb = cv2.ORB_create(nfeatures=2000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)

    if da is None or db is None or not ka or not kb:
        return {
            "kp_a": len(ka) if ka else 0,
            "kp_b": len(kb) if kb else 0,
            "good_matches": 0,
            "inlier_ratio": 0.0,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(da, db, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    inlier_ratio = 0.0
    if len(good) >= 8:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_ratio = float(mask.ravel().mean())

    return {
        "kp_a": len(ka),
        "kp_b": len(kb),
        "good_matches": len(good),
        "inlier_ratio": inlier_ratio,
    }


def find_tiled_roots(project: Path):
    roots = []
    for current, dirs, files in os.walk(project):
        cur = Path(current)
        low = str(cur).lower()

        # prune irrelevant/huge runtime folders
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__", "node_modules", ".cache"}
        ]

        if cur.name in SPLITS:
            parent = cur.parent
            if "avions_1024" in str(parent).lower():
                # parent should contain split dirs
                if all((parent / s / "images").is_dir() for s in SPLITS):
                    roots.append(parent.resolve())

    # unique
    out = []
    seen = set()
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return sorted(out)


def infer_parent_id(stem: str, all_ids):
    # Longest IDs first prevents prefix collisions.
    candidates = []
    for image_id in all_ids:
        if (
            stem == image_id
            or stem.startswith(image_id + "__")
            or stem.startswith(image_id + "_")
            or stem.startswith(image_id + "-")
            or re.match(rf"^{re.escape(image_id)}(?:\D|$)", stem)
        ):
            candidates.append(image_id)
    if not candidates:
        return None
    return max(candidates, key=len)


def search_text_evidence(project: Path):
    yolo_hits = []
    tiling_hits = []

    include_ext = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json"}
    skip_dirs = {
        ".git", "__pycache__", "node_modules", ".cache",
        "runs", "weights"
    }

    yolo_patterns = [
        re.compile(r'load\s*\(\s*["\']yolo26l\.pt["\']\s*\)', re.I),
        re.compile(r'YOLO\s*\(\s*["\']yolo26l-obb\.yaml["\']\s*\)', re.I),
        re.compile(r'yolo26l\.pt', re.I),
    ]
    tiling_patterns = [
        re.compile(r'overlap.{0,20}200|200.{0,20}overlap', re.I),
        re.compile(r'1024.{0,30}(tile|crop|split)|(tile|crop|split).{0,30}1024', re.I),
        re.compile(r'avions_1024', re.I),
    ]

    for current, dirs, files in os.walk(project):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            p = Path(current) / fn
            if p.suffix.lower() not in include_ext:
                continue
            # avoid huge generated files
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for i, line in enumerate(text.splitlines(), 1):
                if any(rx.search(line) for rx in yolo_patterns):
                    yolo_hits.append({"path": str(p), "line": i, "text": line.strip()[:500]})
                if any(rx.search(line) for rx in tiling_patterns):
                    tiling_hits.append({"path": str(p), "line": i, "text": line.strip()[:500]})

    # compact unique
    def uniq(rows, maxn=100):
        out = []
        seen = set()
        for r in rows:
            key = (r["path"], r["line"], r["text"])
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out[:maxn]

    return uniq(yolo_hits), uniq(tiling_hits)


def main():
    args = parse_args()
    project = args.project_root.resolve()
    dataset = (project / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root.resolve()
    out = (project / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("INFO6544 - STEP 1B: NEAR-DUPLICATES + SPLIT/TILING AUDIT")
    print("=" * 100)
    print("READ-ONLY: yes")

    # ------------------------------------------------------------
    # Original-image perceptual fingerprints.
    # ------------------------------------------------------------
    print("\n[1/4] Computing perceptual fingerprints for original images...")

    records = []
    split_ids = {}
    all_ids = set()

    for split in SPLITS:
        imgs = list_images(dataset / split / "images")
        split_ids[split] = {p.stem for p in imgs}
        all_ids |= split_ids[split]

        for p in imgs:
            g = load_gray(p)
            records.append({
                "split": split,
                "image_id": p.stem,
                "path": p,
                "gray": g,
                "phash": phash64(g),
                "dhash": dhash64(g),
            })

    cross_pairs = []
    for i in range(len(records)):
        a = records[i]
        for j in range(i+1, len(records)):
            b = records[j]
            if a["split"] == b["split"]:
                continue
            pd = hamming(a["phash"], b["phash"])
            dd = hamming(a["dhash"], b["dhash"])
            corr = corrcoef_resized(a["gray"], b["gray"])
            cross_pairs.append({
                "split_a": a["split"], "image_a": a["image_id"], "path_a": str(a["path"]),
                "split_b": b["split"], "image_b": b["image_id"], "path_b": str(b["path"]),
                "phash_hamming": pd,
                "dhash_hamming": dd,
                "resized_gray_corr": corr,
            })

    # Objective candidate rule for further local-feature verification.
    candidates = [
        r for r in cross_pairs
        if (
            r["phash_hamming"] <= 10
            or r["dhash_hamming"] <= 10
            or r["resized_gray_corr"] >= 0.90
        )
    ]

    # ORB only on candidate pairs, avoiding expensive all-pairs local matching.
    path_lookup = {str(r["path"]): r["path"] for r in records}
    for r in candidates:
        orb = orb_similarity(path_lookup[r["path_a"]], path_lookup[r["path_b"]])
        r.update(orb)

    # Flag only strong suspicious candidates, but retain all candidate data.
    for r in candidates:
        r["strong_suspicion"] = bool(
            (r["phash_hamming"] <= 6 and r["dhash_hamming"] <= 8)
            or (r["resized_gray_corr"] >= 0.96)
            or (r.get("good_matches", 0) >= 40 and r.get("inlier_ratio", 0.0) >= 0.50)
        )

    candidates.sort(
        key=lambda r: (
            not r["strong_suspicion"],
            r["phash_hamming"],
            r["dhash_hamming"],
            -r["resized_gray_corr"],
        )
    )

    cand_csv = out / "near_duplicate_candidates.csv"
    fields = [
        "split_a","image_a","path_a","split_b","image_b","path_b",
        "phash_hamming","dhash_hamming","resized_gray_corr",
        "kp_a","kp_b","good_matches","inlier_ratio","strong_suspicion"
    ]
    with cand_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in candidates:
            w.writerow({k: r.get(k) for k in fields})

    # Keep top 30 nearest pHash pairs, even if no threshold triggered.
    top_pairs = sorted(
        cross_pairs,
        key=lambda r: (r["phash_hamming"], r["dhash_hamming"], -r["resized_gray_corr"])
    )[:30]
    with (out / "top30_cross_split_nearest_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        fields2 = [
            "split_a","image_a","path_a","split_b","image_b","path_b",
            "phash_hamming","dhash_hamming","resized_gray_corr"
        ]
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows([{k:r[k] for k in fields2} for r in top_pairs])

    # ------------------------------------------------------------
    # Tiling audit.
    # ------------------------------------------------------------
    print("\n[2/4] Auditing tiled datasets and parent split mapping...")

    tiled_roots = find_tiled_roots(project)
    tiling_reports = []

    for root in tiled_roots:
        report = {
            "root": str(root),
            "splits": {},
            "cross_split_parent_ids": {},
            "cross_split_exact_patch_hash_groups": 0,
        }

        parent_to_patch_splits = defaultdict(set)
        hash_to_members = defaultdict(list)

        for split in SPLITS:
            imgs = list_images(root / split / "images")
            mapped = 0
            unmapped = []
            mismatched = []
            parent_counts = defaultdict(int)

            for idx, p in enumerate(imgs, 1):
                parent = infer_parent_id(p.stem, all_ids)
                if parent is None:
                    unmapped.append(p.name)
                else:
                    mapped += 1
                    parent_counts[parent] += 1
                    parent_to_patch_splits[parent].add(split)
                    if parent not in split_ids[split]:
                        mismatched.append({
                            "patch": p.name,
                            "inferred_parent": parent,
                            "patch_split": split,
                            "parent_expected_split": next(
                                (s for s in SPLITS if parent in split_ids[s]),
                                None
                            ),
                        })

                # exact raw patch duplicate across tiled splits
                h = sha256_file(p)
                hash_to_members[h].append((split, p.name, str(p)))

                if idx % 2000 == 0:
                    print(f"  {root.name}/{split}: {idx}/{len(imgs)} patches checked")

            report["splits"][split] = {
                "num_patches": len(imgs),
                "mapped_to_original_parent": mapped,
                "unmapped_count": len(unmapped),
                "unmapped_examples": unmapped[:20],
                "split_mismatch_count": len(mismatched),
                "split_mismatch_examples": mismatched[:20],
                "num_unique_parent_ids": len(parent_counts),
                "parent_ids": sorted(parent_counts.keys()),
            }

        cross_parent = {
            parent: sorted(splits)
            for parent, splits in parent_to_patch_splits.items()
            if len(splits) > 1
        }
        report["cross_split_parent_ids"] = cross_parent

        cross_patch_dups = []
        for h, members in hash_to_members.items():
            splits = sorted(set(x[0] for x in members))
            if len(splits) > 1:
                cross_patch_dups.append({
                    "sha256": h,
                    "splits": splits,
                    "members": members,
                })
        report["cross_split_exact_patch_hash_groups"] = len(cross_patch_dups)
        report["cross_split_exact_patch_hash_examples"] = cross_patch_dups[:20]

        tiling_reports.append(report)

    # ------------------------------------------------------------
    # Search source evidence.
    # ------------------------------------------------------------
    print("\n[3/4] Searching source files for YOLO-loading and tiling evidence...")
    yolo_hits, tiling_hits = search_text_evidence(project)

    # ------------------------------------------------------------
    # Compact summary.
    # ------------------------------------------------------------
    strong = [r for r in candidates if r["strong_suspicion"]]

    summary = {
        "audit_step": "1B_near_duplicates_tiling",
        "read_only": True,
        "original_images_total": len(records),
        "cross_split_pairs_screened": len(cross_pairs),
        "near_duplicate_candidate_count": len(candidates),
        "strong_suspicion_count": len(strong),
        "strong_suspicion_examples": strong[:20],
        "tiled_dataset_roots_found": len(tiled_roots),
        "tiling_reports": tiling_reports,
        "yolo_pretrained_loading_evidence_hits": yolo_hits,
        "tiling_source_evidence_hits": tiling_hits,
        "method_notes": {
            "near_duplicate_screening": (
                "All cross-split original-image pairs were screened using 64-bit pHash, "
                "64-bit dHash and resized grayscale correlation. Candidate pairs then "
                "received ORB+RANSAC local-feature verification."
            ),
            "strong_suspicion_rule": (
                "(pHash<=6 and dHash<=8) OR correlation>=0.96 OR "
                "(ORB good matches>=40 and homography inlier ratio>=0.50)."
            ),
            "limitation": (
                "Perceptual duplicate screening cannot mathematically prove absence of all "
                "semantic/scene overlap; flagged candidates require manual visual review."
            ),
            "tiling_interpretation": (
                "If every patch maps to an original parent in the same split and no parent "
                "appears across tiled splits, the observed dataset is consistent with "
                "split-before-tiling and contains no parent-image leakage. Chronology is "
                "proved only if preparation source/log evidence also documents that order."
            ),
        },
    }

    (out / "audit_step1b_full.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    # Human compact file for upload.
    lines = []
    lines += [
        "="*88,
        "STEP 1B COMPACT SUMMARY",
        "="*88,
        f"Original images total: {len(records)}",
        f"Cross-split pairs screened: {len(cross_pairs)}",
        f"Near-duplicate candidates: {len(candidates)}",
        f"Strong suspicion candidates: {len(strong)}",
        "",
        "TOP SUSPICIOUS / NEAREST PAIRS:",
    ]

    show = candidates[:10] if candidates else top_pairs[:10]
    for r in show:
        lines.append(
            f"{r['split_a']}/{r['image_a']} <-> {r['split_b']}/{r['image_b']} | "
            f"pHash={r['phash_hamming']} dHash={r['dhash_hamming']} "
            f"corr={r['resized_gray_corr']:.4f} "
            f"ORBgood={r.get('good_matches','NA')} "
            f"inlier={r.get('inlier_ratio','NA')} "
            f"strong={r.get('strong_suspicion','NA')}"
        )

    lines += ["", "TILING AUDIT:"]
    if not tiling_reports:
        lines.append("No tiled avions_1024 root automatically found.")
    for rep in tiling_reports:
        lines.append(f"ROOT: {rep['root']}")
        for split in SPLITS:
            s = rep["splits"][split]
            lines.append(
                f"  {split}: patches={s['num_patches']} mapped={s['mapped_to_original_parent']} "
                f"unmapped={s['unmapped_count']} mismatched={s['split_mismatch_count']} "
                f"unique_parents={s['num_unique_parent_ids']}"
            )
        lines.append(f"  cross_split_parent_ids={len(rep['cross_split_parent_ids'])}")
        lines.append(f"  cross_split_exact_patch_hash_groups={rep['cross_split_exact_patch_hash_groups']}")

    lines += ["", "YOLO PRETRAINED LOAD EVIDENCE:"]
    if yolo_hits:
        for h in yolo_hits[:20]:
            lines.append(f"{h['path']}:{h['line']}: {h['text']}")
    else:
        lines.append("No explicit yolo26l.pt loading line found by source scan.")

    lines += ["", "TILING SOURCE EVIDENCE:"]
    if tiling_hits:
        for h in tiling_hits[:20]:
            lines.append(f"{h['path']}:{h['line']}: {h['text']}")
    else:
        lines.append("No explicit tiling source evidence found by source scan.")

    lines += [
        "",
        "FILES:",
        str(cand_csv),
        str(out / "top30_cross_split_nearest_pairs.csv"),
        str(out / "audit_step1b_full.json"),
    ]

    compact = out / "audit_step1b_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")

    print("\n[4/4] Done.")
    print(f"Compact summary: {compact}")
    print(f"Full JSON      : {out / 'audit_step1b_full.json'}")
    print(f"Candidates CSV : {cand_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
