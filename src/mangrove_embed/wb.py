"""Weights & Biases helpers for runs outside the train-seg loop (eval-seg, compare-pixel)."""


def _wandb():
    try:
        import wandb
    except ImportError as e:
        raise SystemExit("--wandb needs `pip install -e .[track]`") from e
    return wandb


def metric_summary(test):
    """{region: metrics} -> flat test/<region>/<metric> keys (dict-valued metrics skipped)."""
    return {f"test/{r}/{k}": v for r, m in test.items() for k, v in m.items()
            if not isinstance(v, dict)}


def log_run(project, entity, group, name, job_type, config, summary, tags=()):
    """One finished W&B run holding only config + summary."""
    run = _wandb().init(project=project, entity=entity, group=group, name=name,
                        job_type=job_type, config=config, tags=list(tags),
                        reinit="finish_previous")
    run.summary.update(summary)
    run.finish()


def update_summary(project, entity, group, name, summary, match=None):
    """Add keys to an existing finished run's summary. `match` filters on config values (e.g.
    train_chips) when a name was reused; the newest match wins. Returns the run path or None."""
    api = _wandb().Api(timeout=60)
    path = f"{entity or api.default_entity}/{project}"
    runs = [r for r in api.runs(path, filters={"group": group, "display_name": name})
            if r.state == "finished"
            and all(r.config.get(k) == v for k, v in (match or {}).items())]
    if not runs:
        return None
    run = max(runs, key=lambda r: r.created_at)
    run.summary.update(summary)
    run.update()
    return "/".join(run.path)
