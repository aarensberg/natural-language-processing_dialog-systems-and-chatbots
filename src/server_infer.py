from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import torch

from src.feedback_ex6 import FeedbackStore, apply_feedback
from src.vocab import Vocabulary
from src.model import Seq2SeqChatbot
from src import config
from src.text import tokenize, detokenize


class InferRequest(BaseModel):
    query: str
    method: str = "greedy"


app = FastAPI()


def _find_checkpoint_path(requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            config.EX6_CHECKPOINT_DIR / "seq2seq_attention.pt",
            config.EX4_CHECKPOINT_DIR / "seq2seq_attention.pt",
            config.EX3_CHECKPOINT_DIR / "cornell_plus_persona" / "seq2seq_attention.pt",
            config.EX2_CHECKPOINT_DIR / "seq2seq_attention.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No chatbot checkpoint found for inference.")


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocabulary = Vocabulary(checkpoint["vocab"])
    model = Seq2SeqChatbot(
        vocab_size=len(vocabulary.token_to_id),
        embed_dim=checkpoint.get("embed_dim", config.EMBED_DIM),
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        num_layers=checkpoint.get("num_layers", config.MODEL_NUM_LAYERS),
        bidirectional_encoder=config.USE_BIDIRECTIONAL_ENCODER,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary


@app.on_event("startup")
def startup():
    global model, vocabulary, feedback_store
    device = torch.device("cpu")
    checkpoint = None
    checkpoint_path = _find_checkpoint_path(checkpoint)
    model, vocabulary = _load_model(checkpoint_path, device)
    feedback_store = FeedbackStore()


@app.post("/infer")
def infer(req: InferRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Empty query")
    # tensorize and decode using model methods
    source_tokens = vocabulary.encode(tokenize(req.query), add_eos=True)
    source_ids = torch.tensor([source_tokens], dtype=torch.long)
    source_length = torch.tensor([len(source_tokens)], dtype=torch.long)
    if req.method == "greedy":
        gen = model.greedy_decode(
            source_ids,
            source_length,
            max_length=config.MAX_TARGET_TOKENS,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
        )
    elif req.method == "beam":
        gen = model.beam_search(
            source_ids,
            source_length,
            max_length=config.MAX_TARGET_TOKENS,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
        )
    else:
        gen = model.sample_decode(
            source_ids,
            source_length,
            max_length=config.MAX_TARGET_TOKENS,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
            top_k=30,
            temperature=0.8,
        )
    text = detokenize(vocabulary.decode(gen.token_ids, stop_at_eos=True))
    text = apply_feedback(req.query, feedback_store, text)
    return {"response": text}


@app.post("/add_correction")
def add_correction(payload: dict):
    q = payload.get("query")
    corr = payload.get("correction")
    if not q or not corr:
        raise HTTPException(
            status_code=400, detail="Both 'query' and 'correction' required"
        )
    feedback_store.add_correction(q, corr)
    return {"status": "ok"}


@app.get("/feedback")
def list_feedback():
    return feedback_store.export()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
