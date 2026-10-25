import tempfile
import unittest
from pathlib import Path

from knowswe.chunking.method_chunker import chunk_python_methods


class TestMethodChunker(unittest.TestCase):

    def check_reconstruction(self, source, max_tokens=300):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "example.py"
            path.write_text(source, encoding="utf-8")

            chunks = chunk_python_methods(
                path,
                root,
                max_tokens=max_tokens,
            )

            self.assertEqual(
                "".join(chunk.content for chunk in chunks),
                source,
            )

            self.assertEqual(
                len({chunk.chunk_id for chunk in chunks}),
                len(chunks),
            )

            return chunks

    def test_decorated_class(self):
        source = (
            "def decorator(cls):\n"
            "    return cls\n"
            "\n"
            "@decorator\n"
            "class Example:\n"
            "    def run(self):\n"
            "        return 1\n"
        )

        chunks = self.check_reconstruction(source)

        self.assertTrue(
            any(chunk.symbol == "Example.run" for chunk in chunks)
        )

    def test_decorated_method(self):
        source = (
            "class Example:\n"
            "    @staticmethod\n"
            "    def run():\n"
            "        return 42\n"
        )

        chunks = self.check_reconstruction(source)

        method_chunks = [
            chunk for chunk in chunks
            if chunk.symbol == "Example.run"
        ]

        self.assertTrue(method_chunks)
        self.assertIn("@staticmethod", method_chunks[0].content)

    def test_nested_class(self):
        source = (
            "class Outer:\n"
            "    class Inner:\n"
            "        def run(self):\n"
            "            return 42\n"
        )

        chunks = self.check_reconstruction(source)

        self.assertTrue(
            any(
                chunk.symbol == "Outer.Inner"
                for chunk in chunks
            )
        )

    def test_long_line_preserves_source(self):
        source = 'VALUE = "' + ("x" * 2000) + '"\n'

        chunks = self.check_reconstruction(
            source,
            max_tokens=100,
        )

        self.assertEqual(
            "".join(chunk.content for chunk in chunks),
            source,
        )

        self.assertTrue(
            any(chunk.estimated_tokens > 100 for chunk in chunks)
        )

    def test_empty_file(self):
        chunks = self.check_reconstruction("")
        self.assertEqual(chunks, [])

    def test_invalid_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "example.py"
            path.write_text("x = 1\n")

            with self.assertRaises(ValueError):
                chunk_python_methods(path, root, max_tokens=0)


if __name__ == "__main__":
    unittest.main()
