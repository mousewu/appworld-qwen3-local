from pathlib import Path
import tempfile
import unittest
from slime_runtime import find_training_script


class RuntimeTests(unittest.TestCase):
    def test_installed_source_auto_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'slime').mkdir()
            ((root/'train.py').resolve()).touch()
            self.assertEqual(find_training_script({}, str(root/'slime/__init__.py')), (root/'train.py').resolve())

    def test_explicit_script_without_source_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'custom_train.py'
            path.touch()
            self.assertEqual(find_training_script({'SLIME_TRAIN_SCRIPT':str(path)}), path.resolve())

    def test_existing_checkout_any_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.py'
            path.touch()
            self.assertEqual(find_training_script({'SLIME_DIR':directory}), path.resolve())

    def test_missing_entry_gives_path_help(self):
        with self.assertRaisesRegex(ValueError, 'No version/commit is required'):
            find_training_script({}, '/nonexistent/slime/__init__.py')


if __name__ == '__main__':
    unittest.main()
