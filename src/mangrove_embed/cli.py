import argparse
import logging

from . import config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mangrove", description="Mangrove embedding diagnostics")
    ap.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth", help="one-time Earth Engine login (opens browser)")

    e = sub.add_parser("extract", help="pull embeddings + labels to data/processed")
    e.add_argument("--project", help="Earth Engine Cloud project id")
    e.add_argument("--regions", nargs="*", help="subset of regions from config")
    e.add_argument("--n-per-class", type=int)
    e.add_argument("--workers", type=int)

    d = sub.add_parser("diagnose", help="run Q1-Q4 diagnostics on extracted data")
    d.add_argument("--data", help="dir with *_labeled.parquet (default: data/processed)")
    d.add_argument("--out", help="results dir (default: <results_dir>/point_diagnostics)")
    d.add_argument("--models", nargs="*", default=["knn", "logreg", "mlp"])
    d.add_argument("--features", nargs="*")
    d.add_argument("--anchor", help="single-region training probe (default: config anchor)")
    d.add_argument("--skip-within", action="store_true")
    d.add_argument("--no-centering", action="store_true")

    c = sub.add_parser("chips", help="pull dense embedding + label chips for segmentation")
    c.add_argument("--project", help="Earth Engine Cloud project id")
    c.add_argument("--regions", nargs="*")
    c.add_argument("--workers", type=int)

    t = sub.add_parser("train-seg", help="train SegFormer on chips (needs the [seg] extra)")
    t.add_argument("--protocol", choices=["loro", "pooled"], default="loro")
    b = sub.add_parser("bench", help="multi-seed LORO comparison of models on the same splits")
    b.add_argument("--models", nargs="*", default=["mlp", "fcn", "segformer"])
    b.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    b.add_argument("--metric", default="mangrove_iou")
    for t in (t, b):
        _train_args(t)

    cs = sub.add_parser("compare-seg", help="tabulate all train-seg runs side by side")
    cs.add_argument("--out", help="results dir (default: <results_dir>/model_comparison)")
    cs.add_argument("--metric", default="mangrove_iou")
    cs.add_argument("--region", help="only this held-out region (e.g. mekong)")

    ev = sub.add_parser("eval-seg", help="score saved train-seg checkpoints on unseen chips")
    ev.add_argument("--chips", required=True, help="chips dir to test on (e.g. data/test/chips)")
    ev.add_argument("--runs", nargs="*",
                    help="run folders (default: every <results_dir>/model_comparison run with checkpoints)")
    ev.add_argument("--regions", nargs="*", help="only these test regions")
    ev.add_argument("--out", help="CSV (default: <results_dir>/generalization/<regions>.csv)")
    ev.add_argument("--wandb", action="store_true",
                    help="add the scores to each split's W&B run (or log an eval run)")

    px = sub.add_parser("compare-pixel",
                        help="per-pixel knn/logreg/mlp on chip pixels, scored like train-seg")
    px.add_argument("--chips", help="chips dir (default: <data_dir>/chips)")
    px.add_argument("--out", help="results dir (default: <results_dir>/model_comparison)")
    px.add_argument("--models", nargs="*", default=["knn", "logreg", "mlp"])
    px.add_argument("--holdout", default="all", help="region to hold out, or 'all'")
    px.add_argument("--regions", nargs="*", help="restrict to these regions")
    px.add_argument("--protocol", choices=["loro", "pooled"], default="loro",
                    help="pooled: train-seg's pooled chip splits (same seed = same test blocks)")
    px.add_argument("--tag", help="pooled: folder suffix, e.g. no_mexpac")
    px.add_argument("--test-chips", help="train on all regions, test on these chips instead of "
                    "LORO (writes <results_dir>/generalization/<regions>_pixel.csv)")
    px.add_argument("--n-train", type=int, default=30_000, help="training pixels per region")
    px.add_argument("--n-test", type=int, default=100_000, help="test pixels (held-out region)")
    px.add_argument("--seed", type=int, default=0)
    px.add_argument("--wandb", action="store_true", help="log each fit to Weights & Biases")
    px.add_argument("--wandb-project", default="mangrove-embed")
    px.add_argument("--wandb-entity", help="W&B team/user (default: your default entity)")

    sub.add_parser("plots", help="summary figures for model comparison and receptive field")

    sub.add_parser("demo", help="run diagnostics on synthetic data (no Earth Engine needed)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd in ("extract", "chips", "train-seg", "bench", "compare-pixel") else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if a.cmd == "auth":
        import ee
        ee.Authenticate()
        print("Authenticated. Project is read from configs/default.yaml (override: EE_PROJECT).")
        return

    cfg = config.load(a.config,
                      project=getattr(a, "project", None),
                      n_per_class=getattr(a, "n_per_class", None),
                      workers=getattr(a, "workers", None))

    if a.cmd == "extract":
        from .extract import extract
        extract(cfg, a.regions)
    elif a.cmd == "diagnose":
        from .diagnostics import run
        run(a.data or cfg.processed_dir, a.out or f"{cfg.results_dir}/point_diagnostics",
            a.models, a.features, a.anchor or cfg.anchor, a.skip_within, not a.no_centering,
            cfg.block_deg)
    elif a.cmd == "chips":
        from .chips import extract_chips
        extract_chips(cfg, a.regions)
    elif a.cmd == "train-seg":
        import os
        # Apple Metal: run any op MPS lacks on CPU instead of erroring (must precede torch import)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from .seg.train import run
        tc = _train_config(a, cfg)
        run(tc, a.protocol, a.holdout, a.regions, a.tag)
    elif a.cmd == "bench":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from .seg.train import bench
        tc = _train_config(a, cfg)
        bench(tc, a.models, a.seeds, a.holdout, a.regions, a.metric)
    elif a.cmd == "compare-seg":
        from .seg.train import compare
        compare(a.out or f"{cfg.results_dir}/model_comparison", a.metric, a.region)
    elif a.cmd == "eval-seg":
        from pathlib import Path
        from .seg.train import evaluate_runs, load_index
        runs = a.runs or sorted(str(d) for d in Path(f"{cfg.results_dir}/model_comparison").iterdir()
                                if any(d.glob("*_best.pt")))
        regs = a.regions or sorted(load_index(a.chips).region.unique())
        evaluate_runs(a.chips, runs, a.out or f"{cfg.results_dir}/generalization/{'_'.join(regs)}.csv",
                      a.regions, wandb=a.wandb)
    elif a.cmd == "compare-pixel" and a.protocol == "pooled":
        from .pixel import run_pooled
        run_pooled(a.chips or f"{cfg.data_dir}/chips", a.out or f"{cfg.results_dir}/model_comparison",
                   a.models, a.n_train, n_test=a.n_test, seed=a.seed, regions=a.regions,
                   tag=a.tag, wandb=_wb(a))
    elif a.cmd == "compare-pixel" and a.test_chips:
        from .pixel import run_external
        run_external(a.chips or f"{cfg.data_dir}/chips", a.test_chips,
                     a.out or f"{cfg.results_dir}/generalization", a.models, a.n_train,
                     n_test=a.n_test, seed=a.seed, regions=a.regions, wandb=_wb(a))
    elif a.cmd == "compare-pixel":
        from .pixel import run as run_pixel
        run_pixel(a.chips or f"{cfg.data_dir}/chips", a.out or f"{cfg.results_dir}/model_comparison",
                  a.models, a.holdout, a.n_train, n_test=a.n_test, seed=a.seed,
                  regions=a.regions, wandb=_wb(a))
    elif a.cmd == "plots":
        from .plots import run as run_plots
        run_plots(cfg.results_dir)
    elif a.cmd == "demo":
        from . import synthetic
        from .diagnostics import run
        data = synthetic.make(f"{cfg.data_dir}/synthetic")
        run(data, f"{cfg.results_dir}/synthetic", ("knn", "logreg"), anchor="jamaica")


def _wb(a):
    return dict(project=a.wandb_project, entity=a.wandb_entity) if a.wandb else None


def _train_args(t):
    t.add_argument("--holdout", default="all", help="region to hold out, or 'all' (loro)")
    t.add_argument("--regions", nargs="*", help="restrict to these regions")
    t.add_argument("--chips", help="chips dir (default: <data_dir>/chips)")
    t.add_argument("--out", help="output dir (default: <results_dir>/model_comparison)")
    t.add_argument("--weights", help="HF SegFormer weights (default: b0 ADE)")
    t.add_argument("--stem-stride", type=int, help="stage-1 stride (default 2; pretrained 4)")
    t.add_argument("--epochs", type=int)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--lr-backbone", type=float)
    t.add_argument("--lr-new", type=float)
    t.add_argument("--max-chips-per-region", type=int)
    t.add_argument("--loader-workers", type=int)
    t.add_argument("--device", help="cuda | mps | cpu (default: best available)")
    t.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="fp16 mixed precision (default: on for CUDA, off for MPS)")
    t.add_argument("--model", choices=["segformer", "fcn", "mlp"], help="default segformer")
    t.add_argument("--mlp-hidden", help="mlp hidden widths, comma-separated (default 64)")
    t.add_argument("--mlp-norm", action="store_true", help="BatchNorm after each mlp hidden layer")
    t.add_argument("--mlp-dropout", type=float, help="dropout after each mlp hidden layer")
    t.add_argument("--mlp-context", help="radii (px) of mean-pooled neighbourhoods added to the "
                   "mlp input, comma-separated, e.g. 1,3,7 (3x3, 7x7, 15x15); default none")
    t.add_argument("--pca", type=int, help="fixed PCA 64 -> k (fit on training regions only)")
    t.add_argument("--input-proj", type=int, help="learned 1x1 projection 64 -> k")
    t.add_argument("--fcn-dilations", help="fcn dilations, comma- or dash-separated, one block "
                   "each; 0 = 1x1 (no context). Default 1,2,4,8,16 (630 m receptive field)")
    t.add_argument("--fcn-width", type=int, help="fcn channels (default 64)")
    t.add_argument("--seed", type=int, help="random seed (adds _s<seed> to the results folder)")
    t.add_argument("--from-scratch", action="store_true", help="segformer with random init")
    t.add_argument("--wandb", action="store_true", help="log runs to Weights & Biases")
    t.add_argument("--wandb-project", help="default mangrove-embed")
    t.add_argument("--wandb-entity", help="W&B team/user (default: your default entity)")
    t.add_argument("--wandb-name", help="W&B run name (split is appended, e.g. rf11/holdout_mekong)")
    t.add_argument("--wandb-group", help="W&B group (default: results folder name)")
    t.add_argument("--wandb-tags", nargs="*", help="extra W&B tags")
    t.add_argument("--wandb-notes", help="W&B run notes")
    t.add_argument("--tag", help="subfolder for this run's results (default: protocol name)")


def _train_config(a, cfg):
    from .seg.train import TrainConfig
    tc = TrainConfig(chips_dir=a.chips or f"{cfg.data_dir}/chips",
                     out_dir=a.out or f"{cfg.results_dir}/model_comparison", seed=cfg.seed)
    mlp = (tuple(int(x) for x in a.mlp_hidden.replace("-", ",").split(",") if x != "")
           if a.mlp_hidden else None)
    if a.seed is not None:
        tc.seed, tc.name_seed = a.seed, True
    dil = (tuple(int(x) for x in a.fcn_dilations.replace("-", ",").split(",") if x != "")
           if a.fcn_dilations else None)
    ctx = (tuple(int(x) for x in a.mlp_context.replace("-", ",").split(",") if x != "")
           if a.mlp_context else None)
    for k, v in dict(fcn_dilations=dil, mlp_hidden=mlp, mlp_context=ctx, mlp_norm=a.mlp_norm or None,
                     mlp_dropout=a.mlp_dropout, fcn_width=a.fcn_width, weights=a.weights, stem_stride=a.stem_stride, epochs=a.epochs,
                     batch_size=a.batch_size, lr_backbone=a.lr_backbone, lr_new=a.lr_new,
                     max_chips_per_region=a.max_chips_per_region, workers=a.loader_workers,
                     device=a.device, model=a.model, pca=a.pca,
                     input_proj=a.input_proj, from_scratch=a.from_scratch or None,
                     wandb=a.wandb or None, wandb_project=a.wandb_project,
                     wandb_entity=a.wandb_entity, amp=a.amp, wandb_name=a.wandb_name,
                     wandb_group=a.wandb_group,
                     wandb_tags=tuple(a.wandb_tags) if a.wandb_tags else None,
                     wandb_notes=a.wandb_notes).items():
        if v is not None:
            setattr(tc, k, v)
    return tc


if __name__ == "__main__":
    main()
