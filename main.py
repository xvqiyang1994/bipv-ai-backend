"""
BIPV.ai Backend API
====================
FastAPI server that receives 3D building models and returns
per-vertex solar radiation analysis.

Endpoints:
  POST /api/analyze    - Upload OBJ/GLB, get radiation data
  GET  /api/cities     - List available city presets
  GET  /api/health     - Health check

Usage:
  uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import time

from .solar_engine import compute_solar_radiation, CITY_PRESETS, SolarResult
from .mesh_parser import parse_obj, parse_glb, center_and_normalize


app = FastAPI(
    title="BIPV.ai Solar Analysis API",
    description="Professional-grade solar radiation analysis for building facades",
    version="0.1.0",
)

# CORS — allow frontend to call from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response Models ────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Solar analysis result returned to the frontend."""
    # Per-vertex data for Three.js rendering
    vertex_radiation: List[float]   # radiation value per vertex (kWh/m²/yr)
    vertices: List[List[float]]     # [[x,y,z], ...] — analysis mesh vertices
    faces: List[List[int]]          # [[v0,v1,v2], ...] — face indices

    # Summary metrics
    min_radiation: float
    max_radiation: float
    mean_radiation: float

    # Financial
    total_facade_area_m2: float
    total_facade_area_sqft: float
    annual_generation_kwh: float
    annual_savings_usd: float
    payback_years: float
    install_cost_usd: float

    # Metadata
    num_faces: int
    num_vertices: int
    computation_time_sec: float
    city: str
    mode: str


class CityInfo(BaseModel):
    code: str
    name: str
    latitude: float
    ghi_annual: float


# ── City names ─────────────────────────────────────────────────────────

CITY_NAMES = {
    "sf": "San Francisco, CA",
    "la": "Los Angeles, CA",
    "nyc": "New York, NY",
    "chi": "Chicago, IL",
    "sea": "Seattle, WA",
    "mia": "Miami, FL",
    "phx": "Phoenix, AZ",
    "den": "Denver, CO",
    "bos": "Boston, MA",
    "aus": "Austin, TX",
}


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "bipv-ai-solar-engine"}


@app.get("/api/cities", response_model=List[CityInfo])
async def list_cities():
    """List all available city presets with their solar parameters."""
    result = []
    for code, params in CITY_PRESETS.items():
        result.append(CityInfo(
            code=code,
            name=CITY_NAMES.get(code, code),
            latitude=params.latitude,
            ghi_annual=params.ghi_annual,
        ))
    return result


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    city: str = Query("sf", description="City code (sf, la, nyc, etc.)"),
    mode: str = Query("annual", description="Analysis mode: annual, summer, winter"),
    panel_efficiency: float = Query(0.18, ge=0.05, le=0.35),
    electricity_rate: float = Query(0.28, ge=0.05, le=1.0),
    subdivide: bool = Query(True, description="Subdivide mesh for higher resolution"),
    max_edge_length: float = Query(3.0, ge=0.5, le=20.0),
):
    """
    Upload a 3D building model (OBJ or GLB) and receive
    per-vertex solar radiation analysis.

    The response includes:
    - Per-vertex radiation values for heatmap rendering
    - The analysis mesh (possibly subdivided for smoothness)
    - Financial metrics (generation, savings, payback)
    """
    t0 = time.time()

    # Validate city
    if city not in CITY_PRESETS:
        raise HTTPException(400, f"Unknown city: {city}. Available: {list(CITY_PRESETS.keys())}")

    # Validate mode
    if mode not in ("annual", "summer", "winter"):
        raise HTTPException(400, f"Unknown mode: {mode}. Use: annual, summer, winter")

    # Read file
    content = await file.read()
    filename = file.filename or "model.obj"
    ext = filename.rsplit('.', 1)[-1].lower()

    # Parse mesh
    try:
        if ext == "obj":
            vertices, faces = parse_obj(content.decode('utf-8', errors='replace'))
        elif ext in ("glb", "gltf"):
            vertices, faces = parse_glb(content)
        else:
            raise HTTPException(400, f"Unsupported file format: .{ext}. Use OBJ or GLB.")
    except ValueError as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    # Center mesh
    vertices = center_and_normalize(vertices)

    # Safety check — limit mesh size
    if len(faces) > 200000:
        raise HTTPException(400, f"Mesh too large ({len(faces)} faces). Maximum: 200,000 faces.")

    # Run solar analysis
    try:
        result: SolarResult = compute_solar_radiation(
            vertices=vertices,
            faces=faces,
            city=city,
            mode=mode,
            panel_efficiency=panel_efficiency,
            electricity_rate=electricity_rate,
            subdivide=subdivide,
            max_edge_length=max_edge_length,
        )
    except Exception as e:
        raise HTTPException(500, f"Solar analysis failed: {str(e)}")

    computation_time = time.time() - t0

    return AnalysisResponse(
        vertex_radiation=result.vertex_radiation.tolist(),
        vertices=result.vertices.tolist(),
        faces=result.faces.tolist(),
        min_radiation=round(result.min_radiation, 1),
        max_radiation=round(result.max_radiation, 1),
        mean_radiation=round(result.mean_radiation, 1),
        total_facade_area_m2=round(result.total_facade_area_m2, 1),
        total_facade_area_sqft=round(result.total_facade_area_m2 * 10.764, 1),
        annual_generation_kwh=round(result.annual_generation_kwh, 0),
        annual_savings_usd=round(result.annual_savings_usd, 0),
        payback_years=round(result.payback_years, 1),
        install_cost_usd=round(result.install_cost_usd, 0),
        num_faces=len(result.faces),
        num_vertices=len(result.vertices),
        computation_time_sec=round(computation_time, 2),
        city=city,
        mode=mode,
    )


@app.post("/api/analyze-quick")
async def analyze_quick(
    file: UploadFile = File(...),
    city: str = Query("sf"),
    mode: str = Query("annual"),
):
    """
    Quick analysis without mesh subdivision — faster but lower resolution.
    Useful for preview / loading state.
    """
    return await analyze(
        file=file,
        city=city,
        mode=mode,
        subdivide=False,
        max_edge_length=20.0,
    )
