"""Dependency-free report and chart writers."""
import base64
import hashlib
import html
import json
from pathlib import Path
from xml.sax.saxutils import escape


def _pdf_literal(value):
    text = "".join(character if ord(character) >= 32 and ord(character) != 127 else " "
                   for character in str(value))
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


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
    not_run_by_code = result.get("summary", {}).get("not_run_by_code", {})
    performance = result.get("performance", {})
    score = lambda key: (f"{result['scores'][key]}/100"
                         if result["scores"].get(key) is not None else "Not available")
    lines = ["LOCAL AI BENCHMARK",
             f"Run type: {'DEMO DATA' if result.get('demo') else 'MODEL DATA'}",
             f"Model: {result['model']['name']}",
             f"Date: {result['timestamp']}", f"Quality: {score('quality')}",
             f"Performance: {score('performance')}",
             f"Overall: {score('overall')}",
             "Performance token counts: " + (
                 "estimated" if performance.get("token_count_is_estimated", True)
                 else "backend-reported"),
             "Performance timing source: " + str(performance.get("timing_source", "unavailable")),
             f"Task set / evaluator: {result.get('task_set_version', 'unknown')} / "
             f"{result.get('evaluator_version', 'unknown')}",
             f"Tests: {passed} passed, {failed} failed, {not_run} not run",
             "Not-run reasons: " +
             (", ".join(f"{code}={count}" for code, count in sorted(not_run_by_code.items()))
              if not_run_by_code else "none recorded"),
             "Difficulty tiers are not empirically calibrated.",
             "Pilot coverage: constraint reasoning, inventory code, incident handoff.",
             "", "Methodology: deterministic generated tasks with documented rule-based checks.",
             "Measurements are controlled benchmark results, not universal guarantees."]
    text = "BT /F1 12 Tf 48 760 Td " + " ".join(
        f"({_pdf_literal(line)}) Tj 0 -20 Td" for line in lines) + " ET"
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


# The isometric cube geometry mirrors the frontend voxel lab; all drawing code is static.
_HTML_VOXEL_SCRIPT = """
const towers=JSON.parse(document.getElementById('scores-data').textContent);
const canvas=document.getElementById('voxel-lab'),context=canvas.getContext('2d');
function shade(hex,amount){
 const number=parseInt(hex.slice(1),16);
 return '#'+[number>>16,(number>>8)&255,number&255].map(value=>Math.max(0,Math.min(255,value+amount)).toString(16).padStart(2,'0')).join('');
}
function polygon(points,fill){
 context.beginPath();points.forEach(([x,y],index)=>index?context.lineTo(x,y):context.moveTo(x,y));context.closePath();
 context.fillStyle=fill;context.fill();context.strokeStyle='#09111d';context.stroke();
}
function cube(x,y,size,height,color){
 const half=size/2,top=y-height;
 polygon([[x-half,top],[x,top+half/2],[x,y+half/2],[x-half,y]],shade(color,-28));
 polygon([[x+half,top],[x,top+half/2],[x,y+half/2],[x+half,y]],shade(color,-48));
 polygon([[x,top-half/2],[x+half,top],[x,top+half/2],[x-half,top]],shade(color,18));
}
context.strokeStyle='#1d3853';
for(let i=-10;i<=10;i++){
 context.beginPath();context.moveTo(500+i*55,80);context.lineTo(110+i*55,550);context.stroke();
 context.beginPath();context.moveTo(500+i*55,80);context.lineTo(890+i*55,550);context.stroke();
}
towers.forEach(([name,score],index)=>{
 const zone=index<3,row=zone?0:1+Math.floor((index-3)/5),column=zone?index*2+1:(index-3)%5;
 const x=zone?300+column*70:280+(column-row)*92,y=zone?170:270+(column+row)*38;
 const value=score===null?null:Math.max(0,Math.min(100,Number(score)));
 const blocks=value===null?1:Math.max(1,Math.ceil(value/10)),height=zone?15:12,size=zone?58:48;
 const color=value===null?'#64748b':value>=80?'#35c98f':value>=50?'#fbbf24':'#ef6a6a';
 for(let block=0;block<blocks;block++)cube(x,y-block*height,size,height,color);
 context.fillStyle='#dceafe';context.font=zone?'bold 12px system-ui':'11px system-ui';context.textAlign='center';
 context.fillText(name.length>18?name.slice(0,16)+'…':name,x,y+size/2+15);
 context.fillText(value===null?'N/T':value.toFixed(1),x,y+size/2+29);
});
"""


def write_html(path, result):
    """Write a self-contained offline report without executing model-generated content."""
    escaped = lambda value: html.escape(str(value), quote=True)
    scores = result.get("scores", {})
    categories = result.get("quality", {}).get("categories", {})
    towers = [
        ("QUALITY ZONE", scores.get("quality")),
        ("PERFORMANCE ZONE", scores.get("performance")),
        ("CONTEXT ZONE", (scores.get("context_behavior") or {}).get("fresh")),
        *((name, value) for name, value in categories.items() if name != "overall"),
    ]
    safe_json = json.dumps(towers, ensure_ascii=True, allow_nan=False).replace(
        "&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    script_hash = base64.b64encode(hashlib.sha256(_HTML_VOXEL_SCRIPT.encode()).digest()).decode()
    score_rows = "".join(
        f"<tr><th>{escaped(name)}</th><td>{'Not tested' if value is None else escaped(value)}</td></tr>"
        for name, value in [
            ("Overall", scores.get("overall")), ("Quality", scores.get("quality")),
            ("Performance", scores.get("performance")), *categories.items()])
    evidence = []
    sections = [("Primary tests", result.get("tests", []))]
    sections.extend((f"Context: {name}", rows)
                    for name, rows in result.get("context_scaling", {}).items())
    for label, tests in sections:
        rows = []
        for test in tests:
            evaluation = test.get("evaluation") or {}
            measurements = {key: test[key] for key in (
                "latency_s", "ttft_s", "generation_s", "prompt_tokens", "output_tokens",
                "prefill_tokens_per_second", "decode_tokens_per_second",
                "token_count_is_estimated", "timing_is_estimated", "timing_source",
                "backend_metrics", "finish_reason", "output_truncated",
                "estimated_total_context_tokens", "estimated_context_token_limit",
                "context_budget_is_estimated") if key in test}
            rows.append(
                f"<details><summary>{escaped(test.get('id', 'Test'))} · "
                f"{escaped(test.get('status', 'Unknown'))} · Score: "
                f"{escaped(test.get('score', 'N/T'))}</summary>"
                f"<h4>Prompt</h4><pre>{escaped(test.get('prompt', ''))}</pre>"
                f"<h4>Answer</h4><pre>{escaped(test.get('answer', ''))}</pre>"
                f"<h4>Expected answer</h4><pre>{escaped(test.get('expected_answer', ''))}</pre>"
                f"<h4>Evaluation evidence</h4><pre>{escaped(json.dumps(evaluation, ensure_ascii=False, indent=2))}</pre>"
                f"<h4>Measurement provenance</h4><pre>{escaped(json.dumps(measurements, ensure_ascii=False, indent=2))}</pre>"
                f"<h4>Attempts and diagnostics</h4><pre>{escaped(json.dumps(test.get('attempts', []), ensure_ascii=False, indent=2))}</pre>"
                f"</details>")
        evidence.append(f"<section><h2>{escaped(label)}</h2>{''.join(rows)}</section>")
    metadata = {key: result.get(key) for key in (
        "configuration", "hardware", "performance", "summary", "notes",
        "task_set_version", "evaluator_version", "warmup_errors")}
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        f"script-src 'sha256-{script_hash}'; style-src 'unsafe-inline'; "
        'base-uri \'none\'; form-action \'none\'">'
        '<title>Local AI benchmark report</title><style>'
        'body{background:#0b1214;color:#dceafe;font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}'
        'section,details{border:1px solid #34504b;border-radius:8px;padding:16px;margin:14px 0}'
        'canvas{width:100%;height:auto}table{border-collapse:collapse;width:100%}'
        'th,td{text-align:left;border-bottom:1px solid #34504b;padding:8px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}h1{color:#9ef2ce}'
        '</style><h1>Local AI benchmark</h1>'
        f"<p><strong>{'DEMO DATA — not measured model performance' if result.get('demo') else 'MODEL DATA'}</strong></p>"
        f"<h2>{escaped(result.get('model', {}).get('name', 'Unknown model'))}</h2>"
        f"<p>{escaped(result.get('timestamp', ''))} · {escaped(result.get('id', ''))}</p>"
        f"<p>Suite seed: {escaped(result.get('suite_seed', result.get('seed', 'Unknown')))} · "
        f"Sampling seed: {escaped((result.get('configuration') or {}).get('sampling_seed', 'Unknown'))}</p>"
        '<p>Controlled benchmark results, not universal guarantees. Difficulty tiers are not empirically '
        'calibrated. Token counts and timings may be estimated: see measurement flags below. '
        'All content is embedded; this report works offline.</p>'
        '<canvas id="voxel-lab" width="1000" height="620" role="img" '
        'aria-label="Voxel benchmark scores; exact values in the following table."></canvas>'
        f'<table><caption>Scores / 100 (N/T = not tested)</caption>{score_rows}</table>'
        + "".join(evidence) +
        f"<section><h2>Configuration and measurement provenance</h2><pre>{escaped(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre></section>"
        f'<script id="scores-data" type="application/json">{safe_json}</script>'
        f"<script>{_HTML_VOXEL_SCRIPT}</script></html>", encoding="utf-8")
