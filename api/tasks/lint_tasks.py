from invoke import task, Collection, Context
import subprocess


@task()
def run_task(ctx: Context) -> None:
    """Lint the project"""
    cmd = ["uv", "run", "ruff", "check"]
    subprocess.run(cmd)
    cmd = ["uv", "run", "ruff", "format"]
    subprocess.run(cmd)


tasks = Collection()
tasks.add_task(run_task, "run", default=True)
