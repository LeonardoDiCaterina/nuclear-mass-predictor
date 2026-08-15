from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware

from nuclear_mass_predictor.api.schemas import (
    HealthResponse,
    NuclearInput,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from nuclear_mass_predictor.api.service import ModelInferenceService

# Global service instance
inference_service = ModelInferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler to load trained model artifacts on API startup."""
    try:
        inference_service.load_model()
    except Exception as e:
        print(f"Warning: Could not pre-load model on startup: {e}")
    yield


app = FastAPI(
    title="Nuclear Mass Predictor API",
    description="REST API for predicting nuclear mass excess and total binding energy using Deep Neural Networks (JAX ANN7).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root() -> dict[str, str]:
    """Welcome endpoint with API details and documentation link."""
    return {
        "message": "Welcome to the Nuclear Mass Predictor API",
        "docs_url": "/docs",
        "health_check": "/health",
        "winning_model": "ANN7 (JAX)",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Checks the health of the inference engine and model availability."""
    if not inference_service.is_loaded:
        try:
            inference_service.load_model()
        except Exception:
            return HealthResponse(
                status="unhealthy",
                model_name="ANN7",
                framework="jax",
                version="1.0.0",
            )

    return HealthResponse(
        status="healthy",
        model_name="ANN7",
        framework="jax",
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_nuclei(request: PredictionRequest) -> PredictionResponse:
    """
    Predicts total binding energy (MeV) and binding energy per nucleon (MeV/A)
    for a list of nuclear input specifications (Z, N).
    """
    results: list[PredictionResult] = []

    for item in request.nuclei:
        try:
            be_total, be_per_a = inference_service.predict_single(item.z, item.n)
            results.append(
                PredictionResult(
                    z=item.z,
                    n=item.n,
                    a=item.z + item.n,
                    predicted_binding_energy_total_mev=round(be_total, 4),
                    predicted_binding_energy_per_nucleon_mev=round(be_per_a, 4),
                    model_name="ANN7",
                    framework="jax",
                )
            )
        except FileNotFoundError as fnf_err:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(fnf_err),
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error for nucleus Z={item.z}, N={item.n}: {err}",
            )

    return PredictionResponse(predictions=results, count=len(results))


@app.get("/predict/{z}/{n}", response_model=PredictionResult, tags=["Inference"])
async def predict_single_path(
    z: int = Path(..., description="Proton number Z", ge=1, le=120),
    n: int = Path(..., description="Neutron number N", ge=1, le=200),
) -> PredictionResult:
    """
    Convenient GET endpoint to query prediction for a single nucleus by Z and N path parameters.
    Example: `/predict/20/28` (Calcium-48).
    """
    try:
        be_total, be_per_a = inference_service.predict_single(z, n)
        return PredictionResult(
            z=z,
            n=n,
            a=z + n,
            predicted_binding_energy_total_mev=round(be_total, 4),
            predicted_binding_energy_per_nucleon_mev=round(be_per_a, 4),
            model_name="ANN7",
            framework="jax",
        )
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error for nucleus Z={z}, N={n}: {err}",
        )
