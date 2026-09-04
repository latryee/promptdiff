"""FastAPI Live Interactive Web Server & API Playground for promptdiff (promptdiff serve / web)."""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from pydantic import BaseModel

from promptdiff.sdk import compare, fuzz, shrink

logger = logging.getLogger("promptdiff.cli.server")


class CompareRequest(BaseModel):
    v1_prompt: str
    v2_prompt: str
    dataset: list[dict[str, Any]]
    model: str = "gpt-4o"
    mock: bool = True


class FuzzRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    mock: bool = True


class ShrinkRequest(BaseModel):
    prompt: str
    target_reduction: float = 0.30
    mock: bool = True


def create_app() -> Any:
    """Create and configure FastAPI application."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, status
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.warning("FastAPI not installed. Run `pip install fastapi uvicorn` to enable promptdiff serve.")
        return None

    api = FastAPI(
        title="⚡ PromptDiff Enterprise API & Playground",
        version="3.0.0",
        description="RESTful API for live LLM prompt regression testing, red-teaming, and token compression.",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> None:
        expected_key = os.getenv("PROMPTDIFF_API_KEY")
        if not expected_key:
            return
        if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing X-API-Key header",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    @api.get("/")
    def root() -> dict[str, str]:
        return {"status": "ok", "service": "PromptDiff Live Server", "version": "3.0.0"}

    @api.post("/api/v1/compare", dependencies=[Depends(verify_api_key)])
    def api_compare(req: CompareRequest) -> dict[str, Any]:
        report = compare(
            v1=req.v1_prompt,
            v2=req.v2_prompt,
            dataset=req.dataset,
            model=req.model,
            mock=req.mock,
        )
        return {
            "passed": report.verdict.passed,
            "cost_delta_pct": report.verdict.cost_delta_pct,
            "latency_delta_pct": report.verdict.latency_delta_pct,
            "total_cases": report.total_cases,
        }

    @api.post("/api/v1/fuzz", dependencies=[Depends(verify_api_key)])
    def api_fuzz(req: FuzzRequest) -> dict[str, Any]:
        rep = fuzz(prompt=req.prompt, model=req.model, mock=req.mock)
        return {
            "resilience_score_pct": rep.resilience_score_pct,
            "total_attacks": rep.total_attacks,
            "bypasses_found": rep.bypasses_found,
            "recommendations": rep.recommendations,
        }

    @api.post("/api/v1/shrink", dependencies=[Depends(verify_api_key)])
    def api_shrink(req: ShrinkRequest) -> dict[str, Any]:
        res = shrink(prompt=req.prompt, target_reduction=req.target_reduction, mock=req.mock)
        return {
            "compressed_prompt": res.compressed_prompt,
            "tokens_saved": res.tokens_saved,
            "token_reduction_pct": res.token_reduction_pct,
            "quality_retained_pct": res.quality_retained_pct,
        }

    return api


def launch_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch Uvicorn HTTP server."""
    api_key = os.getenv("PROMPTDIFF_API_KEY")
    if not api_key and host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Security Warning: PROMPTDIFF_API_KEY is not set. Rebinding server to '127.0.0.1' for safety.",
        )
        host = "127.0.0.1"

    try:
        import uvicorn

        app_instance = create_app()
        if app_instance:
            uvicorn.run(app_instance, host=host, port=port)
    except Exception as e:
        logger.error(f"Could not start promptdiff server: {e}")
