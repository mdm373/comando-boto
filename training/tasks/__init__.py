from invoke import task, Context

@task
def hello(_: Context):
    print('hello')