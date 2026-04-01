"""FastAPI server exposing reset/step/state for the environment."""

from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from env.environment import CodeReviewEnv
from env.models import Action


app = FastAPI(title="AI Code Review OpenEnv")
ENV = CodeReviewEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "ai-code-review-env"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    obs = ENV.reset(task_id=req.task_id)
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    obs, reward, done, info = ENV.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    return ENV.state()


def main() -> None:
    """Entrypoint for `server` script in pyproject.toml."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
