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

    content_streams = [_visual_summary(result)]
    for page_lines in pages:
        content = ["BT /F1 10 Tf 48 748 Td"]
        content.extend(f"({_pdf_string(line)}) Tj 0 -14 Td" for line in page_lines)
        content.append("ET")
        content_streams.append("\n".join(content).encode("cp1252", errors="replace"))

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{5 + i * 2} 0 R' for i in range(len(content_streams)))}] "
        f"/Count {len(content_streams)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    for index, stream in enumerate(content_streams):
        page_number = 5 + index * 2
        content_number = page_number + 1
        objects.extend([
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {content_number} 0 R >>",
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


def _visual_summary(result):
    """Draw the PDF's chart-based summary page using dependency-free PDF vectors."""
    scores = result.get("scores", {})
    model = result.get("model", {})
    categories = [
        (name, value) for name, value in result.get("quality", {}).get("categories", {}).items()
        if name != "overall"
    ]
    context_scaling = result.get("context_scaling", {})
    fresh = {item.get("context_size"): item.get("score")
             for item in context_scaling.get("fresh", []) if item.get("context_size") is not None}
    accumulated = {item.get("context_size"): item.get("score")
                   for item in context_scaling.get("accumulated", [])
                   if item.get("context_size") is not None}
    colors = {
        "navy": (0.055, 0.11, 0.18), "ink": (0.12, 0.18, 0.25),
        "muted": (0.38, 0.45, 0.53), "line": (0.86, 0.89, 0.92),
        "track": (0.91, 0.93, 0.95), "green": (0.10, 0.66, 0.44),
        "amber": (0.91, 0.62, 0.12), "red": (0.84, 0.29, 0.29),
        "blue": (0.22, 0.49, 0.78), "white": (1, 1, 1),
        "card": (0.97, 0.98, 0.99),
    }
    commands = []

    def fill(color):
        commands.append(f"{' '.join(map(str, colors[color]))} rg")

    def rect(x, y, width, height, color):
        fill(color)
        commands.append(f"{x} {y} {width} {height} re f")

    def text(x, y, value, size=9, color="ink", bold=False):
        fill(color)
        font = "/F2" if bold else "/F1"
        commands.append(f"BT {font} {size} Tf {x} {y} Td ({_pdf_string(value)}) Tj ET")

    def score_color(value):
        if value is None:
            return "muted"
        if value >= 80:
            return "green"
        if value >= 50:
            return "amber"
        return "red"

    rect(0, 710, 612, 82, "navy")
    text(40, 759, "LOCAL AI BENCHMARK", 20, "white", True)
    text(40, 738, str(model.get("name", "Unknown"))[:82], 10, "white")
    text(40, 720, f"{result.get('timestamp', 'Unknown')}  |  "
                   f"{result.get('configuration', {}).get('profile', 'Unknown')} profile",
         8, "white")

    metric_cards = [
        ("QUALITY", scores.get("quality")),
        ("PERFORMANCE", scores.get("performance")),
        ("OVERALL", scores.get("overall")),
    ]
    for index, (label, value) in enumerate(metric_cards):
        x = 40 + index * 180
        rect(x, 632, 168, 64, "card")
        text(x + 12, 676, label, 8, "muted", True)
        text(x + 12, 648, f"{'Unknown' if value is None else value} / 100", 17,
             score_color(value), True)

    text(40, 610, "CATEGORY SCORES", 11, "ink", True)
    text(502, 610, "0", 7, "muted")
    text(558, 610, "100", 7, "muted")
    for index, (name, value) in enumerate(categories[:12]):
        y = 590 - index * 19
        text(42, y, str(name)[:25], 8)
        rect(214, y - 2, 276, 8, "track")
        if value is not None:
            rect(214, y - 2, max(1, min(100, value) * 2.76), 8, score_color(value))
        text(504, y, "N/T" if value is None else f"{value:g}", 8,
             score_color(value), True)
    if len(categories) > 12:
        text(42, 359, f"+ {len(categories) - 12} more categories in the detailed report",
             8, "muted")

    text(40, 346, "CONTEXT SCALING", 11, "ink", True)
    rect(405, 343, 8, 8, "green")
    text(417, 344, "Fresh", 7, "muted")
    rect(468, 343, 8, 8, "blue")
    text(480, 344, "Accumulated", 7, "muted")
    sizes = sorted(set(fresh) | set(accumulated))[:7]
    if not sizes:
        text(42, 321, "No per-size context results recorded", 8, "muted")
    for index, size in enumerate(sizes):
        y = 323 - index * 16
        text(42, y, f"{size:,}", 8)
        rect(132, y + 2, 250, 4, "track")
        rect(132, y - 4, 250, 4, "track")
        fresh_score = fresh.get(size)
        accumulated_score = accumulated.get(size)
        if fresh_score is not None:
            rect(132, y + 2, max(1, min(10, fresh_score) * 25), 4, "green")
        if accumulated_score is not None:
            rect(132, y - 4, max(1, min(10, accumulated_score) * 25), 4, "blue")
        text(390, y, f"{'N/T' if fresh_score is None else f'{fresh_score * 10:g}'} / "
                     f"{'N/T' if accumulated_score is None else f'{accumulated_score * 10:g}'}",
             7, "muted")

    performance = result.get("performance", {})
    rect(40, 113, 532, 62, "card")
    text(52, 158, "PERFORMANCE SNAPSHOT", 8, "muted", True)
    stats = [
        ("THROUGHPUT", performance.get("tokens_per_second"), "tokens/sec"),
        ("TIME TO FIRST TOKEN", performance.get("ttft_s"), "seconds"),
        ("P95 LATENCY", performance.get("latency_p95_s"), "seconds"),
    ]
    for index, (label, value, unit) in enumerate(stats):
        x = 52 + index * 172
        text(x, 140, label, 7, "muted")
        text(x, 123, f"{'Unknown' if value is None else value} {unit}", 10,
             "ink", True)

    text(40, 85, "Controlled benchmark measurements; not universal performance guarantees.",
         8, "muted")
    text(40, 69, "Detailed measurements, settings, and methodology follow on the next page.",
         8, "muted")
    return "\n".join(commands).encode("cp1252", errors="replace")
