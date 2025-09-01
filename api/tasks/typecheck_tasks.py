from invoke import task, Collection, Context
import subprocess


@task()
def run_task(ctx: Context) -> None:
    """Typecheck the project"""
    cmd = ["uv", "run", "mypy", ".", "--strict"]
    subprocess.run(cmd)


tasks = Collection()
tasks.add_task(run_task, "run", default=True)
