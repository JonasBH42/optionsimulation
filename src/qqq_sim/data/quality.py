from .models import Diagnostic


def summarize_diagnostics(diagnostics):
    counts = {}
    for item in diagnostics:
        counts[item.code] = counts.get(item.code, 0) + 1
    return {"counts": counts, "errors": sum(item.severity == "ERROR" for item in diagnostics),
            "warnings": sum(item.severity == "WARNING" for item in diagnostics)}
