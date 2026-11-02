import sys
from pathlib import Path

# `evals/` é uma pasta de scripts (não um pacote instalado): entra no path só nos testes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
