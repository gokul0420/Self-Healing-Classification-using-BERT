"""
Entry-point CLI.
Run:  python -m src.cli
"""
import click, sys
from src.dag import ClassificationDAG

@click.command()
@click.option("--model_dir", default="model/checkpoint", help="Path to fine-tuned model.")
@click.option("--threshold", default=0.75, help="Confidence threshold.")
@click.option("--fallback", type=click.Choice(["ask","backup"]), default="ask")
def main(model_dir, threshold, fallback):
    dag = ClassificationDAG(model_dir, threshold, fallback)

    click.echo("🔮 Self-Healing Classifier CLI.  Enter text (blank to quit).")
    while True:
        try:
            txt = click.prompt("\nUser input", type=str, default="", show_default=False)
            if not txt:
                sys.exit(0)
            dag.run(txt)
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

if __name__ == "__main__":
    main()
