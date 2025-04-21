import click
from .pipeline    import run
from .calibration import calibrate_shr_tau, calibrate_te_limit, calibrate_gamma
from .config      import load

@click.group()
def cli(): pass

@cli.command()
@click.option("-c","--config", default="config.yaml", type=click.Path())
def calibrate(config):
    cfg = load(config)
    # grids you want to test
    taus  = [0.005, 0.01, 0.025, 0.05]
    tes   = [0.02, 0.04, 0.06]
    gammas= [2.0, 5.0, 7.5, 10.0]

    cfg["shr_tau"]        = calibrate_shr_tau(cfg, taus, lookback_months=60)
    cfg["te_limit"]       = calibrate_te_limit(cfg, tes,  lookback_months=60)
    cfg["risk_aversion"]  = calibrate_gamma(cfg, gammas, lookback_months=60)

    print("\nRecommended parameters:")
    print(f"  shr_tau       = {cfg['shr_tau']}")
    print(f"  te_limit      = {cfg['te_limit']}")
    print(f"  risk_aversion = {cfg['risk_aversion']}")

@cli.command()
@click.option("-c","--config", default="config.yaml", type=click.Path())
@click.option("-o","--output", default="weights.csv", type=click.Path())
@click.option("--refresh-data", is_flag=True)
def optimize(config, output, refresh_data):
    run(config, output, refresh_data)

if __name__=="__main__":
    cli()
