"""Dependency-free report and chart writers."""
from pathlib import Path
from xml.sax.saxutils import escape
import textwrap


def write_chart(path, values):
    bars = []
    for i, (name, value) in enumerate(values.items()):
        if name == "overall": continue
        y = 25 + i * 22
        bars.append(f'<text x="4" y="{y+12}" font-size="10">{escape(name[:20])}</text>'
                    f'<rect x="145" y="{y}" width="{max(1, value)*2}" height="14" fill="#35c98f"/>'
                    f'<text x="350" y="{y+12}" font-size="10">{value:.1f}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="{max(80, len(bars)*22+30)}">'
                    + ''.join(bars) + "</svg>")


def write_pdf(path, result):
    model = result.get("model", {})
    scores = result.get("scores", {})
    configuration = result.get("configuration", {})
    options = configuration.get("generation_options", {})
    performance = result.get("performance", {})
    context = result.get("scores", {}).get("context_behavior", {})
    lines = [
        "LOCAL AI BENCHMARK",
        f"Model: {model.get('name', 'Unknown')}",
        f"Date: {result.get('timestamp', 'Unknown')}",
        f"Profile: {configuration.get('profile', 'Unknown')}",
        f"Quality: {scores.get('quality', 'Unknown')}/100",
        f"Performance: {scores.get('performance', 'Unknown')}/100",
        f"Overall: {scores.get('overall', 'Unknown')}/100",
        f"Primary tests passed: {len(result.get('tests', [])) - len(result.get('failures', []))}/{len(result.get('tests', []))}",
        "",
        "CATEGORY SCORES",
    ]
    categories = result.get("quality", {}).get("categories", {})
    for name, value in categories.items():
        if name != "overall":
            lines.append(f"{name}: {'Not tested' if value is None else f'{value}/100'}")

    lines.extend([
        "",
        "PERFORMANCE MEASUREMENTS",
        f"Throughput: {performance.get('tokens_per_second', 'Unknown')} tokens/second",
        f"Performance score: {performance.get('score', scores.get('performance', 'Unknown'))}/100",
        f"Time to first token: {performance.get('ttft_s', 'Unknown')} seconds",
        f"Generation time: {performance.get('generation_s', 'Unknown')} seconds",
        f"Latency mean / p50 / p95 / standard deviation: "
        f"{performance.get('latency_mean_s', 'Unknown')} / "
        f"{performance.get('latency_p50_s', 'Unknown')} / "
        f"{performance.get('latency_p95_s', 'Unknown')} / "
        f"{performance.get('latency_stddev_s', 'Unknown')} seconds",
        f"Warm-up runs: {performance.get('warmup_runs', configuration.get('warmup_runs', 'Unknown'))}; "
        f"measured runs: {performance.get('measured_runs', configuration.get('measured_runs', 'Unknown'))}; "
        f"concurrency levels: {', '.join(map(str, performance.get('concurrency_levels', []))) or 'Unknown'}",
        "",
        "CONTEXT MEASUREMENTS",
        f"Fresh context score: {context.get('fresh', 'Unknown')}/100",
        f"Accumulated context score: {context.get('accumulated', 'Unknown')}/100",
        f"Context degradation: {context.get('degradation', 'Unknown')}",
        "Target | Fresh score | Accumulated score",
    ])
    context_scaling = result.get("context_scaling", {})
    fresh_by_size = {
        item.get("context_size"): item.get("score")
        for item in context_scaling.get("fresh", []) if item.get("context_size") is not None
    }
    accumulated_by_size = {
        item.get("context_size"): item.get("score")
        for item in context_scaling.get("accumulated", []) if item.get("context_size") is not None
    }
    for size in sorted(set(fresh_by_size) | set(accumulated_by_size)):
        fresh_score = fresh_by_size.get(size)
        accumulated_score = accumulated_by_size.get(size)
        lines.append(
            f"{size:,} tokens | {'Not tested' if fresh_score is None else f'{fresh_score * 10:g}/100'} | "
            f"{'Not tested' if accumulated_score is None else f'{accumulated_score * 10:g}/100'}"
        )
    if not fresh_by_size and not accumulated_by_size:
        lines.append("Per-size context measurements unavailable.")

    repeat_scores = result.get("uncertainty", {}).get("repeat_run_scores", [])
    lines.extend([
        "",
        "SETTINGS AND RUNTIME",
        f"Generation settings: {', '.join(f'{name}={value}' for name, value in options.items()) or 'Unknown'}",
        f"Primary test repeats: {configuration.get('repeat_runs', 1)}",
        f"Observed quality mean / standard deviation: "
        f"{result.get('uncertainty', {}).get('quality_mean', 'Unknown')} / "
        f"{result.get('uncertainty', {}).get('quality_stddev', 'Unknown')}",
        f"Repeat-run quality scores: {', '.join(map(str, repeat_scores)) or 'Not estimated'}",
        f"Backend: {model.get('backend', 'Unknown')}",
        f"Backend version: {model.get('backend_version', 'Unknown')}",
        f"Context length: {model.get('context_length', 'Unknown')}",
        f"GPU offload: {model.get('gpu_offload', 'Unknown')}",
        f"Server generation defaults: {model.get('generation_defaults', {})}",
        "",
        "Methodology: deterministic generated tasks, objective exact-match evaluation.",
        "Measurements are controlled benchmark results, not universal guarantees.",
    ])
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(str(line), width=88, break_long_words=True,
                                          break_on_hyphens=False) or [""])
    pages = [wrapped_lines[index:index + 47] for index in range(0, len(wrapped_lines), 47)]
    if not pages:
        pages = [[]]

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + i * 2} 0 R' for i in range(len(pages)))}] "
        f"/Count {len(pages)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for index, page_lines in enumerate(pages):
        page_number = 4 + index * 2
        content_number = page_number + 1
        content = ["BT /F1 10 Tf 48 748 Td"]
        content.extend(f"({_pdf_string(line)}) Tj 0 -14 Td" for line in page_lines)
        content.append("ET")
        stream = "\n".join(content).encode("cp1252", errors="replace")
        objects.extend([
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
        ])
    out = b"%PDF-1.4\n"; offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        obj_bytes = obj if isinstance(obj, bytes) else obj.encode("ascii")
        out += f"{i} 0 obj\n".encode() + obj_bytes + b"\nendobj\n"
    xref = len(out); out += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    out += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)


def _pdf_string(value):
    """Escape a text string for a PDF literal string."""
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
