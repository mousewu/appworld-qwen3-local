import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from process_logging import ProcessLogRelay
from grpo_tracking import SwanLabBridge


class ProcessLoggingTests(unittest.TestCase):
    def test_subprocess_stdout_stderr_unicode_long_line_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'training.log'
            console=io.StringIO()
            code='import sys; print("模型日志"); print("x"*12000); print("TRACEBACK",file=sys.stderr); sys.exit(3)'
            proc=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,bufsize=0)
            relay=ProcessLogRelay(proc.stdout,path,console).start()
            self.assertEqual(proc.wait(),3)
            relay.finish()
            text=path.read_text()
            self.assertIn('模型日志',text)
            self.assertIn('x'*12000,text)
            self.assertIn('TRACEBACK',text)
            self.assertEqual(console.getvalue(),text)

    def test_file_sync_allowlist_and_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'training.log').write_text('first')
            (root/'secret.txt').write_text('not uploaded')
            (root/'episodes').mkdir()
            (root/'episodes/1.json').write_text('{}')
            saved=[]
            bridge=SwanLabBridge.__new__(SwanLabBridge)
            bridge.root=root;bridge.uploaded_files={};bridge.last_file_sync=0
            bridge.file_sync_interval=60;bridge.warned_no_save=False
            bridge.swanlab=SimpleNamespace(save=lambda path,**kwargs:saved.append((path,kwargs)))
            bridge.sync_files(force=True);bridge.sync_files(force=True)
            self.assertEqual(len(saved),2)
            (root/'training.log').write_text('first\nsecond')
            bridge.sync_files(force=True)
            self.assertEqual(len(saved),3)
            self.assertFalse(any('secret.txt' in item[0] for item in saved))
            self.assertTrue(all(item[1]['policy']=='now' for item in saved))


if __name__=='__main__':
    unittest.main()
