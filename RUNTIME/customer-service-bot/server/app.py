from __future__ import annotations

import sys
import os
import time
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import CustomerServiceEnv
from models import Action, ActionType, Observation, Reward, State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("customer-service-bot")

env: Optional[CustomerServiceEnv] = None


class ResetRequest(BaseModel):
    task_type: Optional[str] = None
    scenario_index: int = 0
    seed: Optional[int] = None


class StepRequest(BaseModel):
    message: str
    action_type: ActionType
    confidence: float = 0.5


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = CustomerServiceEnv()
    logger.info("Environment initialized")
    yield
    logger.info("Environment shut down")


app = FastAPI(title="Customer Service Bot Environment", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start = time.time()
    body = None

    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            body = body_bytes.decode("utf-8", errors="replace")
            request._body = body_bytes
        except Exception:
            pass

    response = await call_next(request)
    elapsed = time.time() - start

    logger.info(
        f"{request.method} {request.url.path} | status={response.status_code} | {elapsed*1000:.0f}ms"
    )

    if body and request.url.path in ("/reset", "/step"):
        try:
            data = json.loads(body)
            if request.url.path == "/reset":
                logger.info(
                    f"  -> reset: task={data.get('task_type','auto')}, scenario={data.get('scenario_index',0)}, seed={data.get('seed','random')}"
                )
            elif request.url.path == "/step":
                msg = data.get("message", "")
                preview = msg[:80] + ("..." if len(msg) > 80 else "")
                logger.info(
                    f"  -> step: action={data.get('action_type','?')}, message=\"{preview}\""
                )
        except Exception:
            pass

    return response


@app.post("/reset")
async def reset(request: ResetRequest) -> Observation:
    global env
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    if request.seed is not None:
        env = CustomerServiceEnv(seed=request.seed)

    observation = env.reset(
        task_type=request.task_type,
        scenario_index=request.scenario_index,
    )
    logger.info(
        f"  <- reset: task={observation.task_type}, customer=\"{observation.customer_message[:80]}...\""
    )
    return observation


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    global env
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    action = Action(
        message=request.message,
        action_type=request.action_type,
        confidence=request.confidence,
    )

    observation, reward, done, info = env.step(action)

    tr = info.get("task_result", {})
    logger.info(
        f"  <- step: reward={reward.value:.3f}, task_score={tr.get('score', 0):.3f}, "
        f"done={done}, mood={info.get('customer_mood', 0):.1f}, "
        f"resolved={info.get('is_resolved', False)}, feedback=\"{reward.feedback[:60]}\""
    )

    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
async def get_state() -> State:
    global env
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    return env.state()


@app.get("/health")
async def health():
    return {"status": "ok"}
