"""Offline application of held-out map bias and conservative 2D CI proposal.

This is NOT ESKF fusion: covariance intersection yields a separate 2D
candidate, never changes any component of the 15-state navigation filter.
Independently validated flags are user declarations requiring evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .map_covariance_intersection import propose_map_ci_correction
from .orthophoto_localization import MapLocalizationFix
from .run_map_reference_audit import _rows, _indexed

_PRIOR = {"timestamp_us","prior_n_m","prior_e_m",
          "prior_cov_nn_m2","prior_cov_ne_m2","prior_cov_ee_m2"}
_MAP = {"timestamp_us","map_status","candidate_n_m","candidate_e_m",
        "temporal_integrity_status"}


def run_map_ci_offline(
    prior_csv: str | Path, map_candidate_csv: str | Path,
    calibration_json: str | Path, verification_yaml: str | Path,
    output_dir: str | Path,
) -> dict:
    with Path(calibration_json).open(encoding="utf-8") as f:
        calibrated=json.load(f)
    with Path(verification_yaml).open(encoding="utf-8") as f:
        verify=yaml.safe_load(f)
    if not isinstance(verify,dict) or not isinstance(calibrated,dict):
        raise ValueError("Некорректный файл калибровки/подтверждений")
    # Merely reading an internally produced calibration output must never
    # upgrade its source "EMPIRICAL_HOLDOUT_UNCERTIFIED" to validated status.
    if calibrated.get("covariance_model_status") != "EMPIRICAL_HOLDOUT_UNCERTIFIED":
        raise ValueError("Неизвестное происхождение модели ошибки карты")
    if calibrated.get("flightworthiness_approved") is not False:
        raise ValueError("Модель картографической погрешности не должна разрешать полет")
    bias=np.asarray(calibrated.get("bias_ne_m"),dtype=float)
    covariance=np.asarray(calibrated.get("covariance_ne_m2"),dtype=float)
    if bias.shape != (2,) or not np.isfinite(bias).all():
        raise ValueError("Недопустимое смещение карты")
    p_rows=_indexed(_rows(Path(prior_csv),_PRIOR))
    m_rows=_indexed(_rows(Path(map_candidate_csv),_MAP))
    if set(p_rows)!=set(m_rows):
        raise ValueError("Несовпадающие временные метки: нельзя скрыто интерполировать карту")
    statuses=Counter()
    results=[]
    for t,map_row in m_rows.items():
        prior=p_rows[t]
        map_valid=map_row["map_status"]=="GEOMETRICALLY_ACCEPTED_UNVALIDATED"
        adjusted=(
            np.asarray((float(map_row["candidate_n_m"]),
                        float(map_row["candidate_e_m"]))) - bias
            if map_valid else np.array([float("nan"),float("nan")])
        )
        temporal_ok=(
            map_row["temporal_integrity_status"]==
                "MAP_TEMPORALLY_CONSISTENT_UNVALIDATED"
        )
        fix=MapLocalizationFix(
            map_valid, map_row["map_status"], "map",
            float(adjusted[0]),float(adjusted[1])
        )
        result=propose_map_ci_correction(
            fix,
            prior_ne_m=(float(prior["prior_n_m"]),float(prior["prior_e_m"])),
            prior_covariance_ne_m2=(
                (float(prior["prior_cov_nn_m2"]),float(prior["prior_cov_ne_m2"])),
                (float(prior["prior_cov_ne_m2"]),float(prior["prior_cov_ee_m2"]))
            ),
            map_covariance_ne_m2=tuple(map(tuple,covariance)),
            prior_covariance_independently_validated=(
                verify.get("prior_covariance_independently_validated") is True
            ),
            map_covariance_independently_validated=(
                verify.get("map_covariance_independently_validated") is True
            ),
            reference_frames_and_timestamps_aligned=(
                verify.get("reference_frames_and_timestamps_aligned") is True
            ),
            map_integrity_checked=(
                temporal_ok and verify.get("independent_map_integrity_review") is True
            ),
            max_position_disagreement_m=float(
                verify.get("max_position_disagreement_m",20.)
            )
        )
        statuses[result.status]+=1
        results.append({
            "timestamp_us":t,
            "status":result.status,
            "candidate_n_m":(
                result.position_ne_m[0] if result.position_ne_m else float("nan")
            ),
            "candidate_e_m":(
                result.position_ne_m[1] if result.position_ne_m else float("nan")
            ),
            "candidate_cov_nn_m2":(
                result.covariance_ne_m2[0][0] if result.covariance_ne_m2 else float("nan")
            ),
            "candidate_cov_ne_m2":(
                result.covariance_ne_m2[0][1] if result.covariance_ne_m2 else float("nan")
            ),
            "candidate_cov_ee_m2":(
                result.covariance_ne_m2[1][1] if result.covariance_ne_m2 else float("nan")
            ),
            "prior_weight":result.weight_prior,
            "applied_to_eskf":0,
        })
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"map_ci_candidates.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary={
        "frames":len(results),
        "statuses":dict(sorted(statuses.items())),
        "applied_to_eskf":0,
        "calibration_source_status":calibrated["covariance_model_status"],
        "validation_flags_are_unverified_operator_declarations":True,
        "warnings":[
            "Ковариационное пересечение дает отдельную двумерную гипотезу, а не обновление полного 15D ESKF",
            "Маркировка validated в YAML сама по себе не доказывает реальную метрологическую проверку",
            "Временная устойчивость карты не защищает от систематически ложной геопривязки",
            "Никакие изменения не выдаются в PX4 или управление БВС",
        ]
    }
    with (output/"map_ci_summary.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    return summary


def main():
    p=argparse.ArgumentParser(description="Офлайн-гипотеза CI без корректировки ESKF")
    p.add_argument("--prior-csv",required=True)
    p.add_argument("--map-candidates-csv",required=True)
    p.add_argument("--calibration-json",required=True)
    p.add_argument("--verification-yaml",required=True)
    p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    print(json.dumps(run_map_ci_offline(
        a.prior_csv,a.map_candidates_csv,a.calibration_json,
        a.verification_yaml,a.output_dir
    ),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
