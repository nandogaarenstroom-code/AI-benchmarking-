"""Dependency-free report and chart writers."""
from pathlib import Path
from xml.sax.saxutils import escape


def write_chart(path, values):
    bars = []
    for i, (name, value) in enumerate(values.items()):
        if name == "overall": continue
        y = 25 + i * 22
        not_tested = value is None
        score = 0 if not_tested else min(100, max(0, float(value)))
        bars.append(f'<text x="4" y="{y+12}" font-size="10">{escape(name[:20])}</text>'
                    f'<rect x="145" y="{y}" width="{max(1, score)*2}" height="14" '
                    f'fill="{"#64748b" if not_tested else "#35c98f"}"/>'
                    f'<text x="350" y="{y+12}" font-size="10">'
                    f'{"N/T" if not_tested else f"{score:.1f}"}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="{max(80, len(bars)*22+30)}">'
                    + ''.join(bars) + "</svg>")


def write_pdf(path, result):
    # Minimal standards-compliant PDF keeps offline installation dependency-free.
    tests = result.get("tests", [])
    passed = sum(test.get("status") == "PASS" for test in tests)
    failed = sum(test.get("status") == "FAILED" for test in tests)
    not_run = sum(test.get("status") == "NOT RUN" for test in tests)
    operational_errors = result.get("summary", {}).get("operational_errors", {})
    score = lambda key: (f"{result['scores'][key]}/100"
                         if result["scores"].get(key) is not None else "Not available")
    lines = ["LOCAL AI BENCHMARK", f"Model: {result['model']['name']}",
             f"Date: {result['timestamp']}", f"Quality: {score('quality')}",
             f"Performance: {score('performance')}",
             f"Overall: {score('overall')}",
             f"Task set / evaluator: {result.get('task_set_version', 'unknown')} / "
             f"{result.get('evaluator_version', 'unknown')}",
             f"Tests: {passed} passed, {failed} failed, {not_run} not run",
             "Operational errors: " +
             (", ".join(f"{code}={count}" for code, count in sorted(operational_errors.items()))
              if operational_errors else "none recorded"),
             "Difficulty tiers are not empirically calibrated.",
             "Pilot coverage: constraint reasoning, inventory code, incident handoff.",
             "", "Methodology: deterministic generated tasks with documented rule-based checks.",
             "Measurements are controlled benchmark results, not universal guarantees."]
    text = "BT /F1 12 Tf 48 760 Td " + " ".join(f"({line.replace('(', '[').replace(')', ']')}) Tj 0 -20 Td" for line in lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(text.encode())} >>\nstream\n{text}\nendstream"]
    out = b"%PDF-1.4\n"; offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out)); out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out); out += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    out += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)
