# server/app.py – OpenEnv HTTP server
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from environment import CodeReviewEnv
from models import AnyAction, Observation, Reward, State, action_adapter

app = FastAPI(title="Code Review Environment", version="1.0.0")
env = CodeReviewEnv()

# ----------------------------------------------------------------------
# Health & metadata endpoints
# ----------------------------------------------------------------------
@app.get("/")
def root():
    print("[ROOT] Health check hit")
    return {"status": "crazy good"}
    
@app.get("/health")
def health():
    print("[HEALTH] Service is healthy")
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    print("[METADATA] Requested")
    return {
        "name": "Code Review Environment",
        "description": "Multi-turn code review with AST injection, DPO training, and author negotiation."
    }

@app.get("/schema")
def schema():
    print("[SCHEMA] Requested")
    return {
        "action": AnyAction.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema()
    }

@app.post("/mcp")
def mcp():
    print("[MCP] Ping received")
    return {"jsonrpc": "2.0", "result": None}

# ----------------------------------------------------------------------
# Environment endpoints
# ----------------------------------------------------------------------
@app.post("/reset")
def reset(task: str = "easy"):
    try:
        print(f"[RESET] Starting new episode | task={task}")

        env.set_task(task)
        obs = env.reset()

        print(f"[RESET DONE] step={env._step_count}")

        return obs.__dict__
    except Exception as e:
        print(f"[RESET ERROR] {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: dict):
    try:
        print(f"[STEP INPUT] {action}")

        parsed_action = action_adapter.validate_python(action)
        obs, reward, done, info = env.step(parsed_action)

        print(f"[STEP OUTPUT] reward={reward.value:.4f} | done={done}")

        return {
            "observation": obs.__dict__,
            "reward": reward.value,
            "done": done,
            "info": info
        }
    except Exception as e:
        print(f"[STEP ERROR] {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    print("[STATE] Requested")
    s = env.state()
    return s.__dict__

# ----------------------------------------------------------------------
# Main entry point (for local testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("[SERVER START] Running on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)