def parse_snapshot(snapshot: dict) -> list[dict]:
    """
    Extracts the top 20 largest live tensors from torch.cuda.memory._snapshot().
    Returns list of: { size_mb, file, line, function }
    """
    if not snapshot or "segments" not in snapshot:
        return []

    tensors = []
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            history = block.get("history", [])
            if not history:
                continue
            frame = history[-1]
            frames = frame.get("frames", [])
            if not frames:
                continue
            user_frame = next(
                (f for f in frames if "torch" not in f.get("filename", "").lower()),
                frames[0],
            )
            size_mb = block.get("size", 0) / 1024**2
            tensors.append({
                "size_mb": round(size_mb, 2),
                "file": user_frame.get("filename", "unknown"),
                "line": user_frame.get("line", 0),
                "function": user_frame.get("name", "unknown"),
            })

    tensors.sort(key=lambda t: t["size_mb"], reverse=True)
    return tensors[:20]
