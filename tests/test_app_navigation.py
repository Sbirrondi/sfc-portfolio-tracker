import ast
from pathlib import Path
import re
import unittest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
UI_HELPERS_PATH = Path(__file__).resolve().parents[1] / "ui_helpers.py"


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
    def test_app_chrome_is_internal_dashboard_friendly(self):
        source = APP_PATH.read_text()

        self.assertIn('initial_sidebar_state="auto"', source)
        helper_source = UI_HELPERS_PATH.read_text()
        self.assertIn('[data-testid="stToolbar"]', helper_source)
        self.assertIn('[data-testid="stDecoration"]', helper_source)

    def test_positions_page_uses_responsive_kpi_cards(self):
        source = APP_PATH.read_text()
        helper_source = UI_HELPERS_PATH.read_text()

        self.assertIn("position-kpi-grid", helper_source)
        self.assertIn("render_position_kpis", source)
        self.assertNotIn("c1, c2, c3, c4, c5 = st.columns(5)", source)

    def test_management_pages_use_shared_access_gate(self):
        source = APP_PATH.read_text()
        helper_source = UI_HELPERS_PATH.read_text()

        self.assertIn("def render_management_gate(", helper_source)
        self.assertIn("access-panel", helper_source)
        self.assertIn("render_management_gate(\"operazioni\"", source)
        self.assertIn("render_management_gate(\"gestione_info\"", source)

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
        for tab_label in ["Snapshot P&L", "Periodo", "Benchmark", "Driver VNGA60", "Lookthrough", "Dettaglio"]:
            self.assertIn(f'"{tab_label}"', source)

    def test_contribution_page_uses_percent_labels_instead_of_pp(self):
        source = APP_PATH.read_text()
        visible_strings = [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

        for text in visible_strings:
            self.assertIsNone(re.search(r"\bpp\b|\(pp\)", text, flags=re.IGNORECASE))
        self.assertIn("Contributo %", source)
        self.assertIn("Active %", source)


if __name__ == "__main__":
    unittest.main()
