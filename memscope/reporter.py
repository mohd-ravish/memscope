import json
import os
import webbrowser
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from .snapshot import parse_snapshot


def generate_report(samples, analysis, snapshot, layer_stats, output_dir) -> str:
    os.makedirs(output_dir, exist_ok=True)

    steps = [s.step for s in samples]
    allocated = [s.allocated_mb for s in samples]
    reserved = [s.reserved_mb for s in samples]
    cpu_mb = [s.cpu_mb for s in samples]

    top_tensors = parse_snapshot(snapshot) if snapshot else []

    layer_data = {
        name: round(sum(sizes) / len(sizes), 3)
        for name, sizes in layer_stats.items()
        if sizes
    }
    sorted_layers = sorted(layer_data.items(), key=lambda x: x[1], reverse=True)[:20]

    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("report.html.j2")

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        analysis=analysis,
        chart_steps=json.dumps(steps),
        chart_allocated=json.dumps(allocated),
        chart_reserved=json.dumps(reserved),
        chart_cpu=json.dumps(cpu_mb),
        top_tensors=top_tensors,
        layer_labels=json.dumps([l for l, _ in sorted_layers]),
        layer_values=json.dumps([v for _, v in sorted_layers]),
        samples=samples,
        total_samples=len(samples),
        peak_allocated=round(max(allocated, default=0), 2),
        peak_cpu=round(max(cpu_mb, default=0), 2),
    )

    report_path = Path(output_dir) / "memscope_report.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"\n[MemScope] Report written to: {report_path.absolute()}")

    try:
        webbrowser.open(report_path.absolute().as_uri())
    except Exception:
        pass

    return str(report_path.absolute())
