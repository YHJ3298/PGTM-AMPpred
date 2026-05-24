# -*- coding: utf-8 -*-
import argparse
from types import SimpleNamespace

from explainability_stage1_umap import run_stage1
from explainability_stage2_fragments import run_stage2


def main():
    parser = argparse.ArgumentParser(description="Run cleaned explainability pipeline")
    parser.add_argument("--train_merged", type=str, default="train_protrek650m_gated_tabm.csv")
    parser.add_argument("--pos_test", type=str, default="positive_protrek650m_test.csv")
    parser.add_argument("--neg_test", type=str, default="negative_protrek650m_test.csv")
    parser.add_argument("--pos_test_fasta", type=str, required=True)
    parser.add_argument("--neg_test_fasta", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="explainability_outputs/gated_tabm_model.pt")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--out_root", type=str, default="explainability_outputs")
    parser.add_argument("--protrek_model_root", type=str, required=True)
    parser.add_argument("--protrek_code_root", type=str, required=True)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--title_size", type=int, default=16)
    parser.add_argument("--label_size", type=int, default=14)
    parser.add_argument("--tick_size", type=int, default=12)
    parser.add_argument("--mask_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--focus_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--require_correct", action="store_true")
    parser.add_argument("--top_n_per_seq", type=int, default=3)
    args = parser.parse_args()

    stage1_args = SimpleNamespace(
        train_merged=args.train_merged,
        pos_test=args.pos_test,
        neg_test=args.neg_test,
        label_col=args.label_col,
        seed=args.seed,
        checkpoint=args.checkpoint,
        force_retrain=args.force_retrain,
        out_dir=f"{args.out_root}/stage1_umap",
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        title_size=args.title_size,
        label_size=args.label_size,
        tick_size=args.tick_size,
    )
    run_stage1(stage1_args)

    stage2_args = SimpleNamespace(
        train_merged=args.train_merged,
        pos_test=args.pos_test,
        neg_test=args.neg_test,
        pos_test_fasta=args.pos_test_fasta,
        neg_test_fasta=args.neg_test_fasta,
        label_col=args.label_col,
        seed=args.seed,
        checkpoint=args.checkpoint,
        force_retrain=False,
        out_dir=f"{args.out_root}/stage2_fragments",
        protrek_model_root=args.protrek_model_root,
        protrek_code_root=args.protrek_code_root,
        mask_batch_size=args.mask_batch_size,
        max_length=args.max_length,
        window_size=args.window_size,
        stride=args.stride,
        focus_label=args.focus_label,
        n_samples=args.n_samples,
        require_correct=args.require_correct,
        top_n_per_seq=args.top_n_per_seq,
    )
    run_stage2(stage2_args)


if __name__ == "__main__":
    main()
