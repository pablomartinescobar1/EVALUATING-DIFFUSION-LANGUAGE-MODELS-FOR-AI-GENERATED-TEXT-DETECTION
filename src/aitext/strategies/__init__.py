from aitext.strategies import embedding, pawn, score

STRATEGY_MODULES = {
    "score": score,
    "pawn": pawn,
    "embedding": embedding,
}

__all__ = ["score", "pawn", "embedding", "STRATEGY_MODULES"]
