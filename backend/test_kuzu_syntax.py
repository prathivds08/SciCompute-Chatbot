"""Test MERGE ON CREATE SET with parametrized values."""
import kuzu
import tempfile
import os

d = tempfile.mkdtemp()
db = kuzu.Database(os.path.join(d, "test2"))
c = kuzu.Connection(db)

c.execute("CREATE NODE TABLE IF NOT EXISTS W(name STRING, description STRING, PRIMARY KEY(name))")

# Test with simple param names
try:
    c.execute(
        "MERGE (w:W {name: $name}) ON CREATE SET w.description = $desc ON MATCH SET w.description = $desc",
        parameters={"name": "dft", "desc": "DFT convergence tests"}
    )
    print("1. MERGE ON CREATE/MATCH SET with $desc param: OK")
except Exception as e:
    print(f"1. FAILED: {e}")

# Test with different param name
try:
    c.execute(
        "MERGE (w:W {name: $name}) ON CREATE SET w.description = $description ON MATCH SET w.description = $description",
        parameters={"name": "md", "description": "MD simulation"}
    )
    print("2. MERGE with $description param: OK")
except Exception as e:
    print(f"2. FAILED: {e}")

# Test with Step node table (multiple parameters)
try:
    c.execute("CREATE NODE TABLE IF NOT EXISTS Step(name STRING, script_template STRING, default_params STRING, step_order INT64, PRIMARY KEY(name))")
    c.execute(
        "MERGE (s:Step {name: $name}) "
        "ON CREATE SET s.script_template = $description, s.default_params = $params, s.step_order = $step_order "
        "ON MATCH SET s.script_template = $description, s.default_params = $params, s.step_order = $step_order",
        parameters={"name": "dft__parse_structure", "description": "CIF parsing via ASE", "params": "", "step_order": 0}
    )
    print("3. MERGE with $params and $step_order: OK")
except Exception as e:
    print(f"3. FAILED: {e}")

# Read back
res = c.execute("MATCH (w:W) RETURN w.name, w.description")
while res.has_next():
    print(f"   Row: {res.get_next()}")

