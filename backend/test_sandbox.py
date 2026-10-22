import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.sandbox.executor import CodeExecutor

def test_executor():
    executor = CodeExecutor()
    print("Testing basic python execution...")
    res = executor.execute_python("x = 10 * 5\nprint(f'Result={x}')")
    assert res["success"] is True, f"Execution failed: {res}"
    assert "Result=50" in res["stdout"], f"Unexpected stdout: {res['stdout']}"
    print("[OK] Basic execution passed.")

    print("Testing plot generation & artifact capture...")
    plot_script = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure()
plt.plot(x, y, label='sin(x)')
plt.title('Test Sin Wave')
plt.legend()
plt.savefig('my_plot.png')
"""
    res_plot = executor.execute_python(plot_script)
    assert res_plot["success"] is True, f"Plot execution failed: {res_plot}"
    assert len(res_plot["artifacts"]) > 0, f"No artifacts collected: {res_plot}"
    art = res_plot["artifacts"][0]
    assert art["type"] == "image", f"Artifact type not image: {art}"
    assert art["data_url"].startswith("data:image/png;base64,"), "Invalid base64 data url"
    print(f"[OK] Plot captured successfully ({art['name']}, {art['size_bytes']} bytes).")

    print("Testing timeout protection...")
    res_timeout = executor.execute_python("import time\ntime.sleep(5)", timeout=2)
    assert res_timeout["success"] is False, "Expected timeout failure"
    assert "timed out" in res_timeout["stderr"].lower(), f"Unexpected stderr: {res_timeout['stderr']}"
    print("[OK] Timeout protection passed.")

    print("All sandbox tests passed successfully!")

if __name__ == "__main__":
    test_executor()
