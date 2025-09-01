from invoke import task, Collection, Context
import subprocess


@task(help={"port": "Port to run the server on"})
def run_task(ctx: Context, port: int = 8000, reload: bool = True) -> None:
    """Run the FastAPI server on a specific port"""
    cmd = ["uv", "run", "uvicorn", "http_api:app"]
    cmd.extend(["--port", str(port)])
    cmd.extend(["--host", "0.0.0.0"])
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd)


tasks = Collection()
tasks.add_task(run_task, "run", default=True)
