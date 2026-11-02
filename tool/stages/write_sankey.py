import logging
from collections import Counter

from ..enums import ExportFilter, ExportFormat, Variation
from ..pie import pie_name_helper

logger = logging.getLogger("write_sankey")


def _variation_source(value):
    """Count combined variation flags as one source, not several inflows."""
    names = [flag.name for flag in Variation if flag.value and value & flag.value]
    known = 0
    for flag in Variation:
        known |= flag.value
    if value & ~known:
        names.append(hex(value & ~known))
    return "Variation (" + " + ".join(names) + ")"


def sankey_links(subs_df, status_mask):
    """Account for every exported row once using its exclusive final status.

    Originals enter before isomorphism checks; variations enter before operand
    checks. An earlier variation outcome enters at its own stage. A custom mask
    describes only the exported subset, not all query results.
    """
    stages = [
        ("Isomorphism check", [ExportFilter.ISO]),
        ("Candidate checks", [ExportFilter.FILTERED_IO, ExportFilter.FILTERED_COMPLEX,
                              ExportFilter.FILTERED_SIMPLE, ExportFilter.FILTERED_PRED,
                              ExportFilter.FILTERED_REGS, ExportFilter.FILTERED_MEM,
                              ExportFilter.FILTERED_BRANCH, ExportFilter.INVALID]),
        ("Operand checks", [ExportFilter.FILTERED_OPERANDS]),
        ("Encoding checks", [ExportFilter.FILTERED_ENC]),
        ("Weight checks", [ExportFilter.FILTERED_WEIGHTS]),
        ("Generation checks", [ExportFilter.ERROR, ExportFilter.SYNTAX_ERROR]),
        ("Final outcomes", [ExportFilter.SELECTED]),
    ]
    outcomes = {int(flag): (i, pie_name_helper(flag))
                for i, (_, flags) in enumerate(stages) for flag in flags}
    outcomes[int(ExportFilter.SYNTAX_ERROR)] = (5, "Syntax error")
    mask = int(status_mask)
    query = "query" if mask == int(ExportFilter.ALL) else "query (exported subset)"
    links = Counter()
    for status, variation in subs_df[["Status", "Variations"]].itertuples(index=False, name=None):
        status, variation = int(status), int(variation)
        if not status & mask:
            continue
        # Unknown/composite statuses remain explicit rather than disappearing.
        end, outcome = outcomes.get(status, (6, f"Other status ({status})"))
        start = min(2, end) if variation else 0
        source = _variation_source(variation) if variation else query
        links[source, stages[start][0]] += 1
        for i in range(start, end):
            links[stages[i][0], stages[i + 1][0]] += 1
        links[stages[end][0], outcome] += 1
    return [(source, target, count) for (source, target), count in links.items()]


def write_sankey(settings, subs_df, index_artifacts):
    logger.info("Exporting Sankey chart...")
    if settings.write.sankey_fmt & ExportFormat.MARKDOWN:
        links = sankey_links(subs_df, settings.write.sankey_flt)
        content = """```mermaid
---
config:
  sankey:
    showValues: true
---
sankey-beta

%% source,target,value
"""
        for source, target, value in links:
            content += f"{source},{target},{value}\n"
        content += "```"
        with open(settings.out_dir / "sankey.md", "w") as f:
            f.write(content)
        index_artifacts[None]["sankey"] = settings.out_dir / "sankey.md"
