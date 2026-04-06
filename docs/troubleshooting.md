# Troubleshooting

Common issues and how to fix them when using claw-compactor.

## Low compression ratio on code

**Symptom**: Getting <10% compression on Python/JS/Go code instead of the expected ~25%.

**Likely cause**: tree-sitter is not installed. Neurosyntax falls back to regex-based compression which is much less effective.

**Fix**:
```bash
pip install tree-sitter tree-sitter-python tree-sitter-javascript
```

Verify with:
```python
from claw_compactor.fusion.neurosyntax import Neurosyntax
stage = Neurosyntax()
print(stage._has_tree_sitter)  # Should be True
```

## JSON compression is low (expected ~80%)

**Symptom**: JSON arrays only getting 10-20% compression instead of ~82%.

**Likely cause**: The input is being detected as generic text rather than JSON. Ionizer only fires when Cortex classifies content as JSON.

**Check**:
```python
from claw_compactor.fusion.cortex import Cortex
ctx = FusionContext(content=your_json_string)
result = Cortex().apply(ctx)
# Check result.markers for content type
```

**Fix**: Ensure the JSON is valid and the input starts with `[` or `{`. Wrapped JSON (e.g. inside markdown code fences) may not be detected.

## RewindStore markers not being resolved

**Symptom**: Output contains `[[REWIND:sha256:...]]` markers but the LLM doesn't expand them.

**Likely causes**:
1. The Rewind tool isn't registered with your LLM framework
2. RewindStore TTL expired before the LLM requested expansion
3. The proxy server isn't running

**Fix**: Ensure the proxy is running and the `rewind_retrieve` tool is available:
```bash
cd proxy && npm start
```

Increase TTL if needed:
```python
engine = FusionEngine(rewind_ttl=3600)  # 1 hour
```

## Non-deterministic output between runs

**Symptom**: Same input produces slightly different compressed output each run.

**Likely cause**: Python's hash randomization affects set iteration order in SemanticDedup.

**Fix**: Set `PYTHONHASHSEED=0` for deterministic output:
```bash
PYTHONHASHSEED=0 python your_script.py
```

Note: this is a development/testing concern only. The compression quality is identical regardless of ordering.

## High memory usage on large inputs

**Symptom**: Memory spikes when compressing 500KB+ inputs.

**Cause**: All 14 stages process the full content in memory. Some stages (SemanticDedup, Ionizer) create intermediate data structures proportional to input size.

**Mitigation**:
- Split very large inputs into logical chunks before compression
- Disable stages you don't need:
  ```python
  engine = FusionEngine(disabled_stages=["semantic_dedup", "nexus"])
  ```
- Use streaming for tool outputs where possible

## Compression damages code identifiers

**Symptom**: Variable names or function names are shortened/mangled after compression.

**Likely cause**: Neurosyntax AST parsing failed and fell back to regex, or Abbrev stage is modifying code tokens.

**Check**: Look for `neurosyntax:regex_fallback` in the compression markers.

**Fix**: Ensure tree-sitter is installed for your language. If the language isn't supported, you can disable Neurosyntax for that content:
```python
engine = FusionEngine(disabled_stages=["neurosyntax"])
```

## QuantumLock not detecting cache boundaries

**Symptom**: System prompt isn't being marked for KV-cache alignment.

**Likely cause**: QuantumLock requires the content to include a clear system prompt boundary. It looks for patterns like `<system>`, `System:`, or the Anthropic message format.

**Fix**: Structure your input with explicit role markers:
```
<system>
Your system prompt here.
</system>

<user>
User message here.
</user>
```
