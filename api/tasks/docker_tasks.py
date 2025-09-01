from invoke import task, Collection, Context
import subprocess


@task(help={"watch": "run in watch mode", "build": "rebuild docker image"})
def run_task(ctx: Context, watch: bool = True, build: bool = False) -> None:
    """Run the server using the docker compose stack"""
    cmd = ["docker", "compose"]
    cmd.extend(["-p", "commando-boto-api"])
    cmd.extend(["up"])
    if watch:
        cmd.extend(["--watch"])
    if build:
        cmd.extend(["--build"])
    subprocess.run(cmd)


tasks = Collection()
tasks.add_task(run_task, "run", default=True)
