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
    d.add_argument("--out", help="results dir (default: results)")
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
    t.add_argument("--holdout", default="all", help="region to hold out, or 'all' (loro)")
    t.add_argument("--regions", nargs="*", help="restrict to these regions")
    t.add_argument("--chips", help="chips dir (default: <data_dir>/chips)")
    t.add_argument("--out", help="output dir (default: <results_dir>/seg)")
    t.add_argument("--weights", help="HF SegFormer weights (default: b0 ADE)")
    t.add_argument("--stem-stride", type=int, help="stage-1 stride (default 2; pretrained 4)")
    t.add_argument("--epochs", type=int)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--lr-backbone", type=float)
    t.add_argument("--lr-new", type=float)
    t.add_argument("--max-chips-per-region", type=int)
    t.add_argument("--loader-workers", type=int)
    t.add_argument("--device")
    t.add_argument("--model", choices=["segformer", "fcn"], help="default segformer")
    t.add_argument("--pca", type=int, help="fixed PCA 64 -> k (fit on training regions only)")
    t.add_argument("--input-proj", type=int, help="learned 1x1 projection 64 -> k")
    t.add_argument("--from-scratch", action="store_true", help="segformer with random init")
    t.add_argument("--tag", help="subfolder for this run's results (default: protocol name)")

    cs = sub.add_parser("compare-seg", help="tabulate all train-seg runs side by side")
    cs.add_argument("--out", help="results dir (default: <results_dir>/seg)")
    cs.add_argument("--metric", default="mangrove_iou")

    sub.add_parser("demo", help="run diagnostics on synthetic data (no Earth Engine needed)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd in ("extract", "chips", "train-seg") else logging.WARNING,
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
        run(a.data or cfg.processed_dir, a.out or cfg.results_dir, a.models, a.features,
            a.anchor or cfg.anchor, a.skip_within, not a.no_centering, cfg.block_deg)
    elif a.cmd == "chips":
        from .chips import extract_chips
        extract_chips(cfg, a.regions)
    elif a.cmd == "train-seg":
        from .seg.train import TrainConfig, run
        tc = TrainConfig(chips_dir=a.chips or f"{cfg.data_dir}/chips",
                         out_dir=a.out or f"{cfg.results_dir}/seg", seed=cfg.seed)
        for k, v in dict(weights=a.weights, stem_stride=a.stem_stride, epochs=a.epochs,
                         batch_size=a.batch_size, lr_backbone=a.lr_backbone, lr_new=a.lr_new,
                         max_chips_per_region=a.max_chips_per_region, workers=a.loader_workers,
                         device=a.device, model=a.model, pca=a.pca,
                         input_proj=a.input_proj, from_scratch=a.from_scratch or None).items():
            if v is not None:
                setattr(tc, k, v)
        run(tc, a.protocol, a.holdout, a.regions, a.tag)
    elif a.cmd == "compare-seg":
        from .seg.train import compare
        compare(a.out or f"{cfg.results_dir}/seg", a.metric)
    elif a.cmd == "demo":
        from . import synthetic
        from .diagnostics import run
        data = synthetic.make(f"{cfg.data_dir}/synthetic")
        run(data, f"{cfg.results_dir}/synthetic", ("knn", "logreg"), anchor="jamaica")


if __name__ == "__main__":
    main()
