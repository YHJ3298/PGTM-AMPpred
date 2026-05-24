# -*- coding: utf-8 -*-
import time
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch

from config import (
    CHECKPOINT_PATH,
    PROTREK_MODEL_ROOT,
    PROTREK_CODE_ROOT,
    OUTPUT_DIR,
    FIXED_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    get_default_device,
)

from protrek_feature_extractor import (
    load_protrek_650m_local,
    extract_features_from_fasta,
)

from gated_tabm_model import (
    build_model_from_checkpoint,
    embeddings_to_feature_df,
    standardize_features,
    predict_proba,
    predict_label_from_prob,
    label_to_text,
)


class AMPPredictorEngine:
    def __init__(
        self,
        checkpoint_path: str = str(CHECKPOINT_PATH),
        protrek_model_root: str = str(PROTREK_MODEL_ROOT),
        protrek_code_root: str = str(PROTREK_CODE_ROOT),
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.protrek_model_root = str(protrek_model_root)
        self.protrek_code_root = str(protrek_code_root)

        self.device = device if device is not None else get_default_device()

        self.protrek_model = None
        self.gated_tabm_bundle = None

    def _log(
        self,
        msg: str,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def check_paths(self):
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Classification model checkpoint not found: {self.checkpoint_path}"
            )

        if not Path(self.protrek_model_root).exists():
            raise FileNotFoundError(
                f"ProTrek model directory not found: {self.protrek_model_root}"
            )

        if not Path(self.protrek_code_root).exists():
            raise FileNotFoundError(
                f"ProTrek-main source code directory not found: {self.protrek_code_root}"
            )

    def load_models(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.check_paths()

        if self.protrek_model is None:
            self._log(
                f"[{time.ctime()}] Loading ProTrek model...",
                log_callback,
            )

            self.protrek_model = load_protrek_650m_local(
                device=self.device,
                model_root=self.protrek_model_root,
                protrek_code_root=self.protrek_code_root,
                log_callback=log_callback,
            )

        if self.gated_tabm_bundle is None:
            self._log(
                f"[{time.ctime()}] Loading Gated-TabM checkpoint...",
                log_callback,
            )

            self.gated_tabm_bundle = build_model_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                seed=42,
            )

            self._log(
                f"[{time.ctime()}] Gated-TabM loaded successfully.",
                log_callback,
            )

        return self

    def predict_fasta(
        self,
        fasta_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        threshold: float = FIXED_THRESHOLD,
        save_csv: bool = True,
        output_csv: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        fasta_path = str(fasta_path)

        if not Path(fasta_path).exists():
            raise FileNotFoundError(f"Input FASTA file does not exist: {fasta_path}")

        self.load_models(log_callback=log_callback)

        self._log(f"[{time.ctime()}] Start prediction.", log_callback)
        self._log(f"Input FASTA: {fasta_path}", log_callback)
        self._log(f"Device     : {self.device}", log_callback)
        self._log(f"Threshold  : {threshold}", log_callback)

        names, sequences, seq_lengths, embeddings = extract_features_from_fasta(
            fasta_file=fasta_path,
            client=self.protrek_model,
            batch_size=batch_size,
            max_length=max_length,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        bundle = self.gated_tabm_bundle

        feature_cols = bundle["feature_cols"]
        scaler_mean = bundle["scaler_mean"]
        scaler_scale = bundle["scaler_scale"]
        model = bundle["model"]

        feature_df = embeddings_to_feature_df(
            embeddings=embeddings,
            feature_cols=feature_cols,
        )

        X = standardize_features(
            feature_df=feature_df,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

        prob = predict_proba(
            model=model,
            X=X,
            device=self.device,
            batch_size=2048,
        )

        pred = predict_label_from_prob(
            prob=prob,
            threshold=threshold,
        )

        result_df = pd.DataFrame({
            "protein_name": names,
            "sequence": sequences,
            "sequence_length": seq_lengths,
            "prob": prob,
            "pred": pred,
            "prediction_label": [
                label_to_text(x)
                for x in pred
            ],
        })

        if save_csv:
            if output_csv is None:
                output_csv = OUTPUT_DIR / "prediction_result.csv"
            else:
                output_csv = Path(output_csv)

            output_csv.parent.mkdir(parents=True, exist_ok=True)

            result_df.to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
            )

            self._log(
                f"[{time.ctime()}] Result saved to: {output_csv}",
                log_callback,
            )

        self._log(f"[{time.ctime()}] Prediction finished.", log_callback)

        return result_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PGTM-AMPpred command-line predictor"
    )

    parser.add_argument(
        "--fasta",
        type=str,
        default="examples/example.fasta",
        help="Input FASTA file",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="outputs/prediction_result.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device: auto, cpu, cuda",
    )

    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not detected in the current environment, cannot use cuda.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = get_default_device()

    engine = AMPPredictorEngine(device=device)

    df = engine.predict_fasta(
        fasta_path=args.fasta,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_csv=args.out,
        save_csv=True,
    )

    print(df)


if __name__ == "__main__":
    main()