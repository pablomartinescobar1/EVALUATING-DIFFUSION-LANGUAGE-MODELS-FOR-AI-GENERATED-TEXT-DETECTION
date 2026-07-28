from aitext.strategies import embedding, pawn, score, zero_shot

STRATEGY_MODULES = {
    "score": score,
    "pawn": pawn,
    "embedding": embedding,
    "zero_shot": zero_shot,
}

__all__ = ["score", "pawn", "embedding", "zero_shot", "STRATEGY_MODULES"]
