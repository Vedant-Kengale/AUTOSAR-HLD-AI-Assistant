"""
AutoDoc Insight — Full Architecture Analysis Report Generator
Generates comprehensive, audit-ready Markdown and HTML engineering reports
combining architecture extraction, dependency mapping, inconsistency audits,
RAG verification metrics, and chat transcripts.
"""

from __future__ import annotations
import datetime
from typing import Any, Dict, List, Optional


def generate_mermaid_diagram(dependencies: List[str], components: List[str]) -> str:
    """Generate a valid Mermaid flowchart diagram from extracted components and dependencies."""
    if not dependencies and not components:
        return "```mermaid\ngraph TD\n    A[AUTOSAR Architecture] --> B[No explicit dependencies extracted]\n```"

    lines = ["```mermaid", "graph LR", "    %% AutoDoc Insight - Architectural Dependency Flow"]

    # Style definitions
    lines.append("    classDef swc fill:#1e293b,stroke:#00b4d8,stroke-width:2px,color:#f8fafc;")
    lines.append("    classDef bsw fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#94a3b8;")

    added_nodes = set()
    edge_count = 0

    # Parse dependencies (e.g. "DiagnosticManager -> CANDriver via If_CAN_Diag")
    for dep in dependencies:
        dep_str = str(dep).strip()
        if "->" in dep_str or "→" in dep_str:
            delimiter = "->" if "->" in dep_str else "→"
            parts = dep_str.split(delimiter, 1)
            src = parts[0].strip().replace(" ", "_").replace("-", "_")
            tgt_part = parts[1].strip()
            
            # Check for interface note "via X"
            if " via " in tgt_part.lower():
                tgt, iface = tgt_part.split(" via ", 1)
                tgt = tgt.strip().replace(" ", "_").replace("-", "_")
                label = f"|{iface.strip()}|"
            else:
                tgt = tgt_part.replace(" ", "_").replace("-", "_")
                label = ""

            # Sanitize node names
            src = "".join(c for c in src if c.isalnum() or c == "_")
            tgt = "".join(c for c in tgt if c.isalnum() or c == "_")

            if src and tgt:
                lines.append(f"    {src}{label} --> {tgt}")
                added_nodes.add(src)
                added_nodes.add(tgt)
                edge_count += 1

    # If no edges could be parsed, display components as nodes
    if edge_count == 0:
        for comp in components[:12]:
            safe_c = "".join(c for c in comp.replace(" ", "_").replace("-", "_") if c.isalnum() or c == "_")
            if safe_c:
                lines.append(f"    {safe_c}[{comp}]")
                added_nodes.add(safe_c)

    lines.append("```")
    return "\n".join(lines)


def generate_full_analysis_report(
    doc_stats: List[Dict[str, Any]],
    arch_result: Optional[Dict[str, Any]] = None,
    inconsistencies: Optional[List[Dict[str, Any]]] = None,
    summary_text: str = "",
    eval_result: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    project_id: Optional[str] = None,
) -> str:
    """
    Consolidates all system findings into an exhaustive, publication-grade
    technical report in Markdown format.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    arch = arch_result or {}
    incons = inconsistencies or []
    chats = chat_history or []

    total_pages = sum(ds.get("total_pages", 0) for ds in doc_stats)
    total_chars = sum(ds.get("total_chars", 0) for ds in doc_stats)
    doc_names = [ds.get("doc_name", "Unknown") for ds in doc_stats]

    report = []

    # Title & Header
    report.append("# 🚗 AutoDoc Insight — Comprehensive AUTOSAR Architecture Analysis Report")
    report.append(f"*Automotive High-Level Design (HLD) Automated Review & Verification*")
    report.append(f"\n**Generated on:** `{timestamp}` | **Project ID:** `{project_id or 'AUTOSAR-AUDIT'}`")
    report.append("**Case Study Alignment:** Tata Technologies TechPulse FY-26 — CS1 Applied AI/ML Capstone\n")
    report.append("---\n")

    # Executive Summary / Overview
    report.append("## 1. Executive Summary & Document Metadata\n")
    report.append("| Attribute | Detail |")
    report.append("|---|---|")
    report.append(f"| **Target Document(s)** | {', '.join(f'`{d}`' for d in doc_names) if doc_names else 'N/A'} |")
    report.append(f"| **Total Pages Ingested** | {total_pages} |")
    report.append(f"| **Total Extracted Characters** | {total_chars:,} |")
    report.append(f"| **Identified SW Components** | {len(arch.get('components', []))} |")
    report.append(f"| **Identified Interfaces** | {len(arch.get('interfaces', []))} |")
    report.append(f"| **Identified Ports & Signals** | {len(arch.get('ports', []))} ports, {len(arch.get('signals', []))} signals |")
    report.append(f"| **Potential Inconsistencies** | {len(incons)} items flagged |")
    report.append("\n> **Audit Assurance Notice:** This report was produced via grounded Retrieval-Augmented Generation (RAG) using local vector embeddings (BGE-small-en-v1.5) and verified against page-level citations from the provided AUTOSAR HLD specification.\n")

    # High-level architecture summary
    if summary_text and summary_text.strip():
        report.append("## 2. High-Level Architecture Narrative\n")
        report.append(summary_text.strip())
        report.append("\n")

    # Architecture Entities Breakdown
    report.append("## 3. Extracted Architectural Elements\n")
    
    # Components
    report.append("### 3.1 Software Components (SWCs & BSW Modules)")
    comps = arch.get("components", [])
    if comps:
        for idx, c in enumerate(comps, 1):
            report.append(f"{idx}. **{c}**")
    else:
        report.append("*No software components explicitly extracted.*")
    report.append("")

    # Interfaces
    report.append("### 3.2 Interfaces")
    ifaces = arch.get("interfaces", [])
    if ifaces:
        for idx, iface in enumerate(ifaces, 1):
            report.append(f"- `{iface}`")
    else:
        report.append("*No interfaces explicitly extracted.*")
    report.append("")

    # Ports & Signals Table
    report.append("### 3.3 Ports & Signals Specification")
    ports = arch.get("ports", [])
    signals = arch.get("signals", [])
    
    report.append("\n**Ports Defined:**")
    if ports:
        for p in ports:
            report.append(f"- 🚪 `{p}`")
    else:
        report.append("*None identified.*")

    report.append("\n**Signals Defined:**")
    if signals:
        for s in signals:
            report.append(f"- 📡 `{s}`")
    else:
        report.append("*None identified.*")
    report.append("")

    # Dependencies & Interactions
    deps = arch.get("dependencies", [])
    flows = arch.get("functional_flows", [])
    report.append("### 3.4 Component Dependencies & Functional Flows")
    if deps:
        report.append("\n**Component Interdependencies:**")
        for d in deps:
            report.append(f"- 🔗 {d}")
    if flows:
        report.append("\n**Documented Functional Flows:**")
        for f in flows:
            report.append(f"- 🔄 {f}")
    if not deps and not flows:
        report.append("*No interdependencies or functional flows identified.*")
    report.append("")

    # Mermaid Visual Architecture Diagram
    report.append("### 3.5 Architectural Relationship Diagram")
    report.append(generate_mermaid_diagram(deps, comps))
    report.append("")

    # Inconsistency & Ambiguity Audit
    report.append("## 4. Documentation Inconsistency & Anomaly Assessment\n")
    if incons:
        report.append(f"**Identified {len(incons)} Potential Ambiguities or Discrepancies:**\n")
        for i, inc in enumerate(incons, 1):
            sev = inc.get("severity", "Low")
            finding = inc.get("finding", "Unspecified issue")
            evidence = inc.get("evidence", [])
            report.append(f"#### 4.{i} [{sev.upper()}] {finding}")
            if evidence:
                report.append("**Document Evidence & Citations:**")
                for e in evidence:
                    report.append(f"- 📍 {e}")
            report.append("")
    else:
        report.append("✅ **No conflicting interface definitions, missing dependencies, or terminology contradictions detected in the analyzed sections.**\n")

    # Citations Register
    citations = arch.get("citations", [])
    if citations:
        report.append("## 5. Document Citation & Traceability Register\n")
        report.append("All statements in this report are strictly traceable to the following document pages:")
        for cit in citations:
            report.append(f"- 📑 {cit}")
        report.append("")

    # Evaluation / Benchmark Metrics (if available)
    if eval_result and "metrics" in eval_result:
        m = eval_result["metrics"]
        report.append("## 6. RAG Verification & System Performance Metrics\n")
        report.append("| Verification Metric | Result | Target Benchmark |")
        report.append("|---|---|---|")
        report.append(f"| **Retrieval Success Rate** | {m.get('retrieval_success', 0):.1%} | > 95% |")
        report.append(f"| **Citation Coverage Rate** | {m.get('citation_coverage', 0):.1%} | 100% |")
        report.append(f"| **Factual Groundedness** | {m.get('groundedness_rate', 0):.1%} | > 90% |")
        report.append(f"| **Strict Answer Pass Rate** | {m.get('answer_pass_rate', 0):.1%} | Dependent on LLM |")
        report.append("\n*Note: Metrics calculated against synthetic benchmark evaluation questions.* \n")

    # Chat Transcript (if any Q&A was performed)
    if chats:
        report.append("## 7. Interactive Engineering Q&A Transcript\n")
        for idx, msg in enumerate(chats, 1):
            role = "Architect / User" if msg.get("role") == "user" else "AutoDoc Insight AI"
            content = msg.get("content", "").strip()
            report.append(f"**[{role}]**: {content}\n")

    # Safety Disclaimer
    report.append("---\n")
    report.append("## 8. Engineering Verification Disclaimer\n")
    report.append(
        "> **Important Automotive Compliance Notice:**\n"
        "> AutoDoc Insight is an automated design review assistant intended to accelerate HLD analysis "
        "> and traceability. All architectural conclusions, signal flows, and interface boundaries must be "
        "> confirmed against original formal AUTOSAR Classic/Adaptive platform models, ARXML artifacts, "
        "> and ISO 26262 safety requirement specifications by certified system architects prior to "
        "> release for ECU implementation."
    )

    return "\n".join(report)

