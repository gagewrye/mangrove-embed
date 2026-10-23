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

    sub.add_parser("demo", help="run diagnostics on synthetic data (no Earth Engine needed)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd == "extract" else logging.WARNING,
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
    elif a.cmd == "demo":
        from . import synthetic
        from .diagnostics import run
        data = synthetic.make(f"{cfg.data_dir}/synthetic")
        run(data, f"{cfg.results_dir}/synthetic", ("knn", "logreg"), anchor="jamaica")


if __name__ == "__main__":
    main()
