import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def status() -> None:
    typer.echo("mode=research live_ready=false")


if __name__ == "__main__":
    app()
