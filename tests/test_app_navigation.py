import ast
from pathlib import Path
import unittest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _sidebar_pages() -> list[str]:
    tree = ast.parse(APP_PATH.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "page" for target in node.targets):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "radio":
            continue
        if len(call.args) < 2 or not isinstance(call.args[1], ast.List):
            continue
        return [
            item.value
            for item in call.args[1].elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
    raise AssertionError("Sidebar navigation radio not found")


class AppNavigationTests(unittest.TestCase):
    def test_contribution_page_is_a_single_sidebar_entry(self):
        pages = _sidebar_pages()

        self.assertIn("🏆 Contribuzione Performance", pages)
        self.assertNotIn("🏆 Contribuzione P&L", pages)
        self.assertNotIn("📊 Performance Contribution", pages)

    def test_contribution_page_keeps_all_subscreens(self):
        source = APP_PATH.read_text()

        self.assertIn('elif page == "🏆 Contribuzione Performance":', source)
        self.assertNotIn('elif page == "🏆 Contribuzione P&L":', source)
        self.assertNotIn('elif page == "📊 Performance Contribution":', source)
        for tab_label in ["Snapshot P&L", "Periodo", "Benchmark", "Lookthrough", "Dettaglio"]:
            self.assertIn(f'"{tab_label}"', source)


if __name__ == "__main__":
    unittest.main()
