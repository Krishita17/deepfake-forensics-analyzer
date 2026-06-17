import json
import os
from datetime import datetime
from pathlib import Path


class ReportGenerator:
    def __init__(self, output_dir="outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_html_report(self, analysis_results, image_paths=None, save_name=None):
        if save_name is None:
            save_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        scores = analysis_results.get("scores", {})
        overall = scores.get("overall_score", 0.5)
        verdict = "FAKE" if overall > 0.5 else "REAL"
        verdict_color = "#FF4757" if overall > 0.5 else "#00D4AA"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepFake Analysis Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0F0F1A; color: #E8E8E8; padding: 40px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{ text-align: center; padding: 40px 0; border-bottom: 2px solid #2D2D44; margin-bottom: 40px; }}
    .header h1 {{ font-size: 2.5em; background: linear-gradient(135deg, #5B8DEF, #00D4AA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .verdict {{ font-size: 4em; font-weight: 900; color: {verdict_color}; margin: 20px 0; }}
    .confidence {{ font-size: 1.4em; color: #AAA; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
    .score-card {{ background: #1A1A2E; border-radius: 12px; padding: 24px; border: 1px solid #2D2D44; }}
    .score-card h3 {{ color: #5B8DEF; margin-bottom: 12px; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
    .score-card .value {{ font-size: 2.2em; font-weight: 700; }}
    .score-bar {{ height: 8px; background: #2D2D44; border-radius: 4px; margin-top: 12px; overflow: hidden; }}
    .score-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
    .section {{ margin: 40px 0; }}
    .section h2 {{ font-size: 1.5em; margin-bottom: 20px; color: #5B8DEF; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
    .image-grid img {{ width: 100%; border-radius: 8px; border: 1px solid #2D2D44; }}
    .details-table {{ width: 100%; border-collapse: collapse; }}
    .details-table td {{ padding: 12px 16px; border-bottom: 1px solid #2D2D44; }}
    .details-table td:first-child {{ color: #888; width: 40%; }}
    .footer {{ text-align: center; margin-top: 60px; padding-top: 20px; border-top: 1px solid #2D2D44; color: #555; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>DeepFake Forensics Analysis Report</h1>
        <div class="verdict">{verdict}</div>
        <div class="confidence">Overall Forgery Probability: {overall:.1%}</div>
        <p style="color:#666; margin-top:10px;">{datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
    </div>

    <div class="score-grid">
"""

        score_configs = [
            ("frequency_score", "Frequency Analysis", "#FF6B6B"),
            ("facial_score", "Facial Forensics", "#FFA502"),
            ("temporal_score", "Temporal Consistency", "#5B8DEF"),
            ("neural_score", "Neural Network", "#A55EEA"),
        ]

        for key, label, color in score_configs:
            val = scores.get(key, 0)
            val_color = "#FF4757" if val > 0.5 else "#00D4AA"
            html += f"""
        <div class="score-card">
            <h3>{label}</h3>
            <div class="value" style="color:{val_color}">{val:.1%}</div>
            <div class="score-bar">
                <div class="score-bar-fill" style="width:{val*100}%; background:{color};"></div>
            </div>
        </div>
"""

        html += """    </div>\n"""

        if image_paths:
            html += """    <div class="section">\n        <h2>Visual Analysis</h2>\n        <div class="image-grid">\n"""
            for img_path in image_paths:
                html += f'            <img src="{img_path}" alt="Analysis visualization">\n'
            html += """        </div>\n    </div>\n"""

        details = analysis_results.get("details", {})
        if details:
            html += """    <div class="section">\n        <h2>Detailed Findings</h2>\n        <table class="details-table">\n"""
            for key, value in details.items():
                display_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    display_val = f"{value:.4f}"
                else:
                    display_val = str(value)
                html += f"            <tr><td>{display_key}</td><td>{display_val}</td></tr>\n"
            html += """        </table>\n    </div>\n"""

        html += f"""
    <div class="footer">
        <p>Generated by DeepFake Forensics Analyzer v1.0.0</p>
        <p>This report is for informational purposes. Results should be interpreted by qualified analysts.</p>
    </div>
</div>
</body>
</html>"""

        path = os.path.join(self.output_dir, save_name)
        with open(path, "w") as f:
            f.write(html)

        return path

    def generate_json_report(self, analysis_results, save_name=None):
        if save_name is None:
            save_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "metadata": {
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "analyzer": "DeepFake Forensics Analyzer",
            },
            "results": self._make_serializable(analysis_results),
        }

        path = os.path.join(self.output_dir, save_name)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return path

    @staticmethod
    def _make_serializable(obj):
        import numpy as np
        import torch

        if isinstance(obj, dict):
            return {k: ReportGenerator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ReportGenerator._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return obj
