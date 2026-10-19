"""Reject article documents from the public benchmark distribution."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PublicContentPolicyTests(unittest.TestCase):
    def test_no_manuscript_or_supplementary_article(self):
        for path in (ROOT/'examples').rglob('*'):
            if not path.is_file():
                continue
            self.assertNotIn(path.suffix.lower(), {'.doc','.docx','.odt','.rtf'}, str(path))
            if path.suffix.lower() == '.pdf':
                self.assertIn('/supplement/figures/',path.as_posix(),str(path))
            if path.suffix.lower() == '.tex':
                text=path.read_text(errors='replace')
                self.assertNotIn('\\documentclass',text,str(path))
                self.assertNotIn('\\begin{document}',text,str(path))
        self.assertFalse((ROOT/'examples/paper').exists())
        self.assertFalse((ROOT/'examples/historical-paper').exists())

if __name__=='__main__':
    unittest.main()
