import uvicorn


def run_api() -> None:
    """Runs the FastAPI server via Uvicorn."""
    uvicorn.run("nuclear_mass_predictor.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run_api()
