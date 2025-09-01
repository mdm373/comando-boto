from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "Worldzz"}


@app.get("/items/{item_id}")
def read_item(item_id: str, q: str = "") -> dict[str, str]:
    return {"item_id": item_id, "q": q}
