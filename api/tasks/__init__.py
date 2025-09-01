from .serve_tasks import tasks as serve_tasks
from .docker_tasks import tasks as docker_tasks
from invoke import Collection
from .lint_tasks import tasks as lint_tasks
from .typecheck_tasks import tasks as typecheck_tasks

ns = Collection()
ns.add_collection(docker_tasks, "docker", default=True)
ns.add_collection(serve_tasks, "serve")
ns.add_collection(lint_tasks, "lint")
ns.add_collection(typecheck_tasks, "typecheck")
