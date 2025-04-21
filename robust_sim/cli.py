import click
from .pipeline import run

@click.command()
@click.option("-c","--config", default="config.yaml", type=click.Path())
@click.option("-o","--output", default="weights.csv", type=click.Path())
@click.option("--refresh-data", is_flag=True, help="Rebuild raw price/excess CSVs.")
def main(config, output, refresh_data):
    run(config, output, refresh_data)

if __name__=="__main__":
    main()
